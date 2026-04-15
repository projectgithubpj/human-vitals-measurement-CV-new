import numpy as np
from collections import deque
from scipy.signal import butter, filtfilt, welch, detrend, hilbert
import time


class RespiratoryProcessor:
    # Physiological bounds
    RR_LOW_BPM  = 6
    RR_HIGH_BPM = 30

    FREQ_LOW  = RR_LOW_BPM / 60.0   # 0.10 Hz
    FREQ_HIGH = RR_HIGH_BPM / 60.0  # 0.50 Hz

    def __init__(self, fps: float = 30.0, window_sec: float = 30.0):
        self.fps        = fps
        self.window_sec = window_sec
        self._update_window_len()

        # Buffers
        self.rppg_buf   = deque(maxlen=self.window_len)
        self.nose_y_buf = deque(maxlen=self.window_len)

        # Timing
        self._frame_times = deque(maxlen=60)

        # Output
        self.respiratory_rate = 0.0
        self.confidence       = 0.0
        self.resp_signal      = np.array([])
        self._rr_history      = deque(maxlen=8)

        self.MIN_FRAMES = int(fps * 12)

    # --------------------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------------------

    def update_fps(self, fps: float) -> None:
        if abs(fps - self.fps) > 1.0 and 5 < fps < 120:
            self.fps = fps
            self._update_window_len()

            self.rppg_buf   = deque(self.rppg_buf,   maxlen=self.window_len)
            self.nose_y_buf = deque(self.nose_y_buf, maxlen=self.window_len)

            self.MIN_FRAMES = int(fps * 12)

    def add_frame(self, rppg_pulse_value: float, nose_y: float | None = None) -> dict:
        now = time.time()
        self._frame_times.append(now)

        # FPS estimation
        if len(self._frame_times) >= 10:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            if elapsed > 0:
                measured_fps = (len(self._frame_times) - 1) / elapsed
                measured_fps = np.clip(measured_fps, 5, 120)
                self.update_fps(float(measured_fps))

        # Store signals
        self.rppg_buf.append(float(rppg_pulse_value))
        if nose_y is not None:
            self.nose_y_buf.append(float(nose_y))

        # Not enough data
        if len(self.rppg_buf) < self.MIN_FRAMES:
            return self._pending_result(self.MIN_FRAMES - len(self.rppg_buf))

        return self._estimate_rr()

    # --------------------------------------------------------------
    # CORE
    # --------------------------------------------------------------

    def _estimate_rr(self) -> dict:
        rppg_arr = np.array(self.rppg_buf, dtype=np.float64)

        nose_arr = (
            np.array(self.nose_y_buf, dtype=np.float64)
            if len(self.nose_y_buf) >= self.MIN_FRAMES
            else None
        )

        # Signal A
        rr_rppg, conf_rppg, sig_rppg = self._rr_from_rppg(rppg_arr)

        # Signal B
        rr_nose, conf_nose = 0.0, 0.0
        if nose_arr is not None:
            rr_nose, conf_nose, _ = self._rr_from_landmark(nose_arr)

        # Fusion
        rr_final, conf_final = self._fuse(rr_rppg, conf_rppg, rr_nose, conf_nose)

        # Outlier rejection
        if len(self._rr_history) > 0:
            last_rr = self._rr_history[-1]
            if abs(rr_final - last_rr) > 8:
                rr_final = last_rr

        # Smoothing
        if rr_final > 0:
            self._rr_history.append(rr_final)

        if len(self._rr_history) >= 2:
            recent  = list(self._rr_history)[-6:]
            weights = np.linspace(0.5, 1.0, len(recent))
            smoothed = float(np.average(recent, weights=weights))
        else:
            smoothed = rr_final

        self.respiratory_rate = smoothed
        self.confidence       = conf_final

        # Display signal
        if sig_rppg is not None:
            tail = min(len(sig_rppg), int(self.fps * 5))
            self.resp_signal = sig_rppg[-tail:]

        rr_min, rr_max = self._halfpower_band(smoothed)

        return {
            "respiratory_rate": round(smoothed, 1),
            "rr_confidence":    round(conf_final, 3),
            "rr_status":        "Measuring" if smoothed > 0 else "Estimating…",
            "resp_signal":      self.resp_signal.tolist() if len(self.resp_signal) else [],
            "rr_range":         (rr_min, rr_max),
        }

    # --------------------------------------------------------------
    # SIGNAL METHODS
    # --------------------------------------------------------------

    def _rr_from_rppg(self, rppg: np.ndarray):
        sig = detrend(rppg)

        try:
            envelope = np.abs(hilbert(sig))
        except:
            envelope = np.abs(sig)

        # Smooth
        window = int(self.fps * 0.5)
        if window > 3:
            envelope = np.convolve(envelope, np.ones(window)/window, mode='same')

        envelope = detrend(envelope)

        # Normalize
        std = np.std(envelope)
        if std < 1e-6:
            return 0.0, 0.0, None

        envelope = (envelope - np.mean(envelope)) / std

        resp_sig = self._bandpass(envelope)
        if resp_sig is None or np.std(resp_sig) < 0.01:
            return 0.0, 0.0, None

        rr, conf = self._welch_rr(resp_sig)
        return rr, conf, resp_sig

    def _rr_from_landmark(self, nose_y: np.ndarray):
        sig = detrend(nose_y)

        # Motion rejection
        diff = np.abs(np.diff(sig))
        if np.median(diff) > 0.02:
            return 0.0, 0.0, None

        std = np.std(sig)
        if std < 1e-6:
            return 0.0, 0.0, None

        sig = (sig - np.mean(sig)) / std

        resp_sig = self._bandpass(sig)
        if resp_sig is None or np.std(resp_sig) < 0.01:
            return 0.0, 0.0, None

        rr, conf = self._welch_rr(resp_sig)
        return rr, conf, resp_sig

    # --------------------------------------------------------------
    # FUSION
    # --------------------------------------------------------------

    def _fuse(self, rr_a, conf_a, rr_b, conf_b):
        if conf_a + conf_b == 0:
            return 0.0, 0.0

        w_a = conf_a / (conf_a + conf_b + 1e-8)
        w_b = conf_b / (conf_a + conf_b + 1e-8)

        rr = rr_a * w_a + rr_b * w_b
        conf = max(conf_a, conf_b)

        rr = float(np.clip(rr, self.RR_LOW_BPM, self.RR_HIGH_BPM))
        return rr, conf

    # --------------------------------------------------------------
    # SIGNAL PROCESSING
    # --------------------------------------------------------------

    def _bandpass(self, signal: np.ndarray):
        nyq  = self.fps / 2.0
        low  = np.clip(self.FREQ_LOW  / nyq, 0.01, 0.49)
        high = np.clip(self.FREQ_HIGH / nyq, 0.01, 0.49)

        if low >= high:
            return None

        try:
            b, a = butter(4, [low, high], btype='band')
            padlen = min(3 * max(len(a), len(b)), len(signal) - 1)
            if padlen < 1:
                return None
            return filtfilt(b, a, signal, padlen=padlen)
        except:
            return None

    def _welch_rr(self, signal: np.ndarray):
        nperseg = min(len(signal), int(self.fps * 15))

        freqs, psd = welch(
            signal,
            fs=self.fps,
            nperseg=nperseg,
            noverlap=nperseg // 2,
            window='hann'
        )

        mask = (freqs >= self.FREQ_LOW) & (freqs <= self.FREQ_HIGH)
        if not mask.any():
            return 0.0, 0.0

        psd_m = psd[mask]
        freqs_m = freqs[mask]

        # Top 3 peaks
        idx = np.argsort(psd_m)[-3:]
        peak_freqs = freqs_m[idx]
        peak_powers = psd_m[idx]

        rr_bpm = np.sum(peak_freqs * peak_powers) / np.sum(peak_powers) * 60
        conf = float(np.max(peak_powers) / (np.sum(psd_m) + 1e-12))

        return rr_bpm, conf

    def _halfpower_band(self, rr_center: float, margin: float = 2.0):
        lo = max(int(rr_center - margin), self.RR_LOW_BPM)
        hi = min(int(rr_center + margin), self.RR_HIGH_BPM)
        return lo, hi

    # --------------------------------------------------------------
    # INTERNAL
    # --------------------------------------------------------------

    def _update_window_len(self):
        self.window_len = int(self.fps * self.window_sec)

    def _pending_result(self, remaining: int):
        return {
            "respiratory_rate": 0,
            "rr_confidence": 0.0,
            "rr_status": f"Collecting data… {remaining} frames remaining",
            "resp_signal": [],
            "rr_range": (0, 0),
        }