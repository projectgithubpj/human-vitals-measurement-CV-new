"""
Video Stream Processor  (v4 – Performance Optimized)
=====================================================
Key optimizations over v3:
- HRV computed every HRV_INTERVAL_FRAMES (60 frames = ~2 s) instead of every frame.
  The wavelet CWT in hrv_processor is expensive (~50-100 ms); running it at 30 fps
  was the primary cause of lag.
- Heavy HRV payload fields (psd_freqs, psd_values, wt_lf_hf_trend, rr_intervals)
  are sent only every HRV_PAYLOAD_INTERVAL_FRAMES to further cut WebSocket traffic.
- Frame encoding drops to ENCODE_FPS (15 fps) — the human eye cannot perceive the
  difference at this latency but it halves the WebSocket bandwidth.
- FPS sync to sub-processors happens every SYNC_INTERVAL_FRAMES, not every frame.
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

# ── Tunable constants ────────────────────────────────────────────────────────
# How often (in frames) to run the full HRV pipeline (bandpass + peaks + Welch + CWT).
# At 30 fps: 60 frames = every ~2 seconds. HRV metrics change slowly so this is fine.
HRV_INTERVAL_FRAMES = 60

# How often to send the large HRV spectral arrays (PSD, wavelet trend, RR intervals).
# These are large lists — sending every frame wastes bandwidth.
HRV_PAYLOAD_INTERVAL_FRAMES = 90   # every ~3 s

# Encode and send frames at this rate (fps). Halving from 30→15 cuts frame bandwidth 50%.
ENCODE_FPS = 15

# Sync sub-processor FPS every N frames (cheap but no need to do it every frame).
SYNC_INTERVAL_FRAMES = 30
# ────────────────────────────────────────────────────────────────────────────


class VideoProcessor:
    """
    Manages webcam capture and coordinates rPPG + respiratory + HRV pipeline.

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

        # Diagnostics
        self.frame_count  = 0
        self.detected_fps = 0.0
        self._fps_times: list = []
        self.is_running   = False

        # ── Performance: cached HRV state ────────────────────────────────
        self._hrv_result_cache: dict = {}          # last computed HRV result
        self._hrv_heavy_cache:  dict = {}          # last large-payload HRV fields
        self._last_encode_time: float = 0.0        # for ENCODE_FPS throttle

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
            fc = self.frame_count   # local alias

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

            if face_roi is None:
                # Feed zeros to keep buffers in sync
                rr_result = self.respiratory.add_frame(rppg_pulse_value=0.0, nose_y=None)

                # Only run HRV on schedule
                if fc % HRV_INTERVAL_FRAMES == 0:
                    hrv_raw = self.hrv.add_pulse_sample(0.0)
                    self._update_hrv_cache(hrv_raw, fc)

                result = {
                    "heart_rate":    0,
                    "confidence":    0.0,
                    "bpm_min":       0,
                    "bpm_max":       0,
                    "pulse_signal":  [],
                    "signal_quality": 0.0,
                    "status":        "No face detected – position yourself in frame",
                    "face_detected": False,
                    **rr_result,
                    **self._build_hrv_payload(fc),
                    "fps":           round(self.detected_fps, 1),
                    "frame_b64":     self._maybe_encode(frame, now, encode_interval),
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

                # ---- HRV — throttled to HRV_INTERVAL_FRAMES ----
                if fc % HRV_INTERVAL_FRAMES == 0:
                    hrv_raw = self.hrv.add_pulse_sample(self.rppg.latest_pulse_sample)
                    self._update_hrv_cache(hrv_raw, fc)
                else:
                    # Still feed the sample so the buffer stays current, but skip compute
                    self.hrv.pulse_buf.append(float(self.rppg.latest_pulse_sample))

                result = {
                    **hr_result,
                    **rr_result,
                    **self._build_hrv_payload(fc),
                    "face_detected": True,
                    "face_bbox":     face_roi.face_bbox,
                    "face_quality":  round(face_roi.quality_score, 2),
                    "fps":           round(self.detected_fps, 1),
                    "frame_b64":     self._maybe_encode(face_roi.annotated_frame, now, encode_interval),
                }

            if self.on_result:
                self.on_result(result)

    # ------------------------------------------------------------------
    # HRV cache helpers
    # ------------------------------------------------------------------

    def _update_hrv_cache(self, hrv_raw: dict, fc: int):
        """Split HRV result into lightweight (sent every cycle) and heavy (sent periodically)."""
        HEAVY_KEYS = {'psd_freqs', 'psd_values', 'wt_lf_hf_trend', 'rr_intervals'}

        light = {}
        heavy = {}
        for k, v in hrv_raw.items():
            if k in HEAVY_KEYS:
                heavy[k] = v
            else:
                light[k] = v

        self._hrv_result_cache = light
        # Only refresh heavy payload on schedule
        if fc % HRV_PAYLOAD_INTERVAL_FRAMES == 0:
            self._hrv_heavy_cache = heavy

    def _build_hrv_payload(self, fc: int) -> dict:
        """Return the prefixed HRV dict to merge into the result."""
        payload = {**self._hrv_result_cache}
        # Always include heavy keys (may be empty dict until first compute)
        payload.update(self._hrv_heavy_cache)
        return self._prefix_hrv(payload)

    # ------------------------------------------------------------------
    # Frame encoding — throttled to ENCODE_FPS
    # ------------------------------------------------------------------

    def _maybe_encode(self, frame_bgr: np.ndarray, now: float,
                      encode_interval: float) -> str:
        """Only re-encode if enough time has passed since last encode."""
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
            elif k in PASSTHROUGH:
                out[k] = v
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