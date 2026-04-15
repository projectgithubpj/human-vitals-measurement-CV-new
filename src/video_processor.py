"""
Video Stream Processor  (v5 – BVP Integration)
===============================================
Changes over v4:
- Adds BVPProcessor to the pipeline.  BVP analysis runs every
  BVP_INTERVAL_FRAMES (same cadence as HRV, ~2 s) since per-beat morphology
  is stable on that timescale.
- The lightweight BVP scalar fields (ibi_ms, perfusion_index, bvp_sqi, etc.)
  are sent every frame from the cache; the bvp_signal waveform array is
  bundled with the HRV heavy-payload refresh to avoid excess bandwidth.
- Everything else is unchanged from v4.
"""

import cv2
import threading
import time
import base64
import numpy as np
from typing import Callable, Optional

from .face_roi import FaceROIExtractor
from .rppg_processor import RPPGProcessor
from .respiratory_processor import RespiratoryProcessor
from .hrv_processor import HRVProcessor
from .bvp_processor import BVPProcessor

# ── Tunable constants ────────────────────────────────────────────────────────
HRV_INTERVAL_FRAMES     = 60    # run full HRV pipeline every ~2 s
HRV_PAYLOAD_INTERVAL_FRAMES = 90  # send large spectral arrays every ~3 s

# BVP morphology analysis cadence — same as HRV (beat morphology changes slowly)
BVP_INTERVAL_FRAMES     = 60

ENCODE_FPS              = 15   # frame encoding rate
SYNC_INTERVAL_FRAMES    = 30   # sub-processor FPS sync
# ────────────────────────────────────────────────────────────────────────────


class VideoProcessor:
    """
    Manages webcam capture and coordinates rPPG + respiratory + HRV + BVP pipeline.

    Usage:
        vp = VideoProcessor(on_result=my_callback)
        vp.start()
        ...
        vp.stop()
    """

    NOSE_TIP_LANDMARK = 1

    def __init__(
        self,
        camera_index: int = 0,
        target_fps:   float = 30.0,
        on_result:    Optional[Callable[[dict], None]] = None,
        jpeg_quality: int = 65,
    ):
        self.camera_index = camera_index
        self.target_fps   = target_fps
        self.on_result    = on_result
        self.jpeg_quality = jpeg_quality

        self._cap:    Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # --- Processors ---
        self.roi_extractor = FaceROIExtractor()
        self.rppg          = RPPGProcessor(fps=target_fps, window_sec=15.0)
        self.respiratory   = RespiratoryProcessor(fps=target_fps, window_sec=30.0)
        self.hrv           = HRVProcessor(fps=target_fps, window_sec=60.0)
        self.bvp           = BVPProcessor(fps=target_fps, window_sec=10.0)

        # Diagnostics
        self.frame_count  = 0
        self.detected_fps = 0.0
        self._fps_times: list = []
        self.is_running   = False

        # ── Performance: cached state ─────────────────────────────────
        self._hrv_result_cache: dict = {}
        self._hrv_heavy_cache:  dict = {}
        self._bvp_result_cache: dict = {}   # BVP light fields
        self._bvp_heavy_cache:  dict = {}   # BVP waveform (sent periodically)
        self._last_encode_time: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        import sys
        if sys.platform == 'win32':
            self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(self.camera_index)

        if not self._cap.isOpened():
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS,          self.target_fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.is_running = True
        print(f"[VideoProcessor] Pipeline started on camera {self.camera_index}")
        return True

    def stop(self):
        if not self.is_running:
            return

        self.is_running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()
        self.roi_extractor.release()
        print(f"[VideoProcessor] Pipeline stopped")

    # ------------------------------------------------------------------
    # Main loop (background thread)
    # ------------------------------------------------------------------

    def _run(self):
        frame_interval  = 1.0 / self.target_fps
        encode_interval = 1.0 / ENCODE_FPS
        last_frame_time = 0.0

        while not self._stop_event.is_set():
            now = time.time()
            if now - last_frame_time < frame_interval:
                import eventlet
                eventlet.sleep(0)
                time.sleep(0.001)
                continue

            ret, frame = self._cap.read()
            if not ret:
                print(f"[VideoProcessor] Warning: Camera read failed on index {self.camera_index}")
                time.sleep(0.1)
                continue

            last_frame_time = now
            self.frame_count += 1
            fc = self.frame_count

            # FPS measurement
            self._fps_times.append(now)
            self._fps_times = [t for t in self._fps_times if now - t <= 2.0]
            self.detected_fps = max(len(self._fps_times) / 2.0, 1.0)

            frame = cv2.flip(frame, 1)

            # ---- Face ROI extraction ----
            face_roi = self.roi_extractor.extract(frame)

            # ---- Sync FPS to sub-processors (throttled) ----
            if fc % SYNC_INTERVAL_FRAMES == 0:
                self.respiratory.update_fps(self.detected_fps)
                self.hrv.update_fps(self.detected_fps)
                self.bvp.update_fps(self.detected_fps)

            if face_roi is None:
                # Feed zeros to keep buffers in sync
                rr_result = self.respiratory.add_frame(rppg_pulse_value=0.0, nose_y=None)

                if fc % HRV_INTERVAL_FRAMES == 0:
                    hrv_raw = self.hrv.add_pulse_sample(0.0)
                    self._update_hrv_cache(hrv_raw, fc)

                if fc % BVP_INTERVAL_FRAMES == 0:
                    bvp_raw = self.bvp.add_pulse_sample(0.0)
                    self._update_bvp_cache(bvp_raw, fc)
                else:
                    self.bvp.pulse_buf.append(0.0)

                result = {
                    "heart_rate":     0,
                    "confidence":     0.0,
                    "bpm_min":        0,
                    "bpm_max":        0,
                    "pulse_signal":   [],
                    "signal_quality": 0.0,
                    "status":         "No face detected – position yourself in frame",
                    "face_detected":  False,
                    **rr_result,
                    **self._build_hrv_payload(fc),
                    **self._build_bvp_payload(fc),
                    "fps":            round(self.detected_fps, 1),
                    "frame_b64":      self._maybe_encode(frame, now, encode_interval),
                }
            else:
                # ---- Heart rate (rPPG) — every frame ----
                hr_result = self.rppg.add_frame(face_roi.combined_rgb)

                # ---- Nose-tip Y for respiratory ----
                nose_y = self._extract_nose_y(face_roi.landmarks, frame.shape[0])

                # ---- Respiratory rate — every frame ----
                rr_result = self.respiratory.add_frame(
                    rppg_pulse_value=self.rppg.latest_pulse_sample,
                    nose_y=nose_y,
                )

                # ---- HRV — throttled ----
                if fc % HRV_INTERVAL_FRAMES == 0:
                    hrv_raw = self.hrv.add_pulse_sample(self.rppg.latest_pulse_sample)
                    self._update_hrv_cache(hrv_raw, fc)
                else:
                    self.hrv.pulse_buf.append(float(self.rppg.latest_pulse_sample))

                # ---- BVP — throttled ----
                if fc % BVP_INTERVAL_FRAMES == 0:
                    bvp_raw = self.bvp.add_pulse_sample(self.rppg.latest_pulse_sample)
                    self._update_bvp_cache(bvp_raw, fc)
                else:
                    self.bvp.pulse_buf.append(float(self.rppg.latest_pulse_sample))

                result = {
                    **hr_result,
                    **rr_result,
                    **self._build_hrv_payload(fc),
                    **self._build_bvp_payload(fc),
                    "face_detected": True,
                    "face_bbox":     face_roi.face_bbox,
                    "face_quality":  round(face_roi.quality_score, 2),
                    "fps":           round(self.detected_fps, 1),
                    "frame_b64":     self._maybe_encode(face_roi.annotated_frame, now, encode_interval),
                }

            if self.on_result:
                self.on_result(result)

    # ------------------------------------------------------------------
    # HRV cache helpers  (unchanged from v4)
    # ------------------------------------------------------------------

    def _update_hrv_cache(self, hrv_raw: dict, fc: int):
        HEAVY_KEYS = {'psd_freqs', 'psd_values', 'wt_lf_hf_trend', 'rr_intervals'}
        light, heavy = {}, {}
        for k, v in hrv_raw.items():
            (heavy if k in HEAVY_KEYS else light)[k] = v
        self._hrv_result_cache = light
        if fc % HRV_PAYLOAD_INTERVAL_FRAMES == 0:
            self._hrv_heavy_cache = heavy

    def _build_hrv_payload(self, fc: int) -> dict:
        payload = {**self._hrv_result_cache, **self._hrv_heavy_cache}
        return self._prefix_hrv(payload)

    # ------------------------------------------------------------------
    # BVP cache helpers
    # ------------------------------------------------------------------

    def _update_bvp_cache(self, bvp_raw: dict, fc: int):
        """Split BVP result into light (sent every cycle) and heavy (waveform)."""
        HEAVY_KEYS = {'bvp_signal', 'ibi_arr'}
        light, heavy = {}, {}
        for k, v in bvp_raw.items():
            (heavy if k in HEAVY_KEYS else light)[k] = v
        self._bvp_result_cache = light
        # Refresh waveform on same schedule as HRV heavy payload
        if fc % HRV_PAYLOAD_INTERVAL_FRAMES == 0:
            self._bvp_heavy_cache = heavy

    def _build_bvp_payload(self, fc: int) -> dict:
        """Return the BVP dict to merge into the result."""
        return {**self._bvp_result_cache, **self._bvp_heavy_cache}

    # ------------------------------------------------------------------
    # Frame encoding — throttled to ENCODE_FPS
    # ------------------------------------------------------------------

    def _maybe_encode(self, frame_bgr: np.ndarray, now: float,
                      encode_interval: float) -> str:
        if now - self._last_encode_time >= encode_interval:
            self._last_encode_time = now
            self._last_frame_b64 = self._encode_frame(frame_bgr)
        return getattr(self, '_last_frame_b64', '')

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prefix_hrv(hrv_result: dict) -> dict:
        PASSTHROUGH = {
            "hrv_confidence", "hrv_status",
            "sdnn", "rmssd", "pnn50", "cv",
            "sd1", "sd2",
            "mean_rr", "mean_hr",
            "vlf_power", "lf_power", "hf_power",
            "lf_hf", "lf_nu", "hf_nu", "total_power",
            "psd_freqs", "psd_values",
            "wt_lf_hf", "wt_lf_hf_trend", "wt_confidence",
            "rr_intervals",
            "interpretation",
        }
        out = {}
        for k, v in hrv_result.items():
            if k == "status":
                out["hrv_status_msg"] = v
            else:
                out[k] = v
        return out

    @staticmethod
    def _extract_nose_y(landmarks: np.ndarray, frame_height: int) -> Optional[float]:
        try:
            nose_y_px = landmarks[VideoProcessor.NOSE_TIP_LANDMARK, 1]
            return float(nose_y_px / frame_height)
        except (IndexError, TypeError):
            return None

    def _encode_frame(self, frame_bgr: np.ndarray) -> str:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        _, buf = cv2.imencode('.jpg', frame_bgr, encode_params)
        return base64.b64encode(buf).decode('utf-8')