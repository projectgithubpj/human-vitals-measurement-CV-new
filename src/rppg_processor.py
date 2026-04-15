"""
rPPG (Remote Photoplethysmography) Signal Processor
Uses the CHROM (Chrominance-based) method for heart rate estimation.
Reference: De Haan & Jeanne (2013) - Robust Pulse Rate From Chrominance-Based rPPG

v2 changes
----------
- Exposes `latest_pulse_sample` (float) so VideoProcessor can feed
  each frame's pulse value into RespiratoryProcessor without re-processing.
- Everything else is unchanged.
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch
from collections import deque
from scipy.signal import detrend

import time


class RPPGProcessor:
    """
    Implements the CHROM rPPG algorithm for contactless heart rate estimation.

    Algorithm overview:
    1. Extract mean RGB values from face ROI each frame
    2. Normalize RGB signals to remove illumination variation
    3. Apply CHROM projection to isolate pulse signal
    4. Bandpass filter (0.75–3 Hz → 45–180 BPM)
    5. FFT/Welch PSD to find dominant frequency = heart rate
    """

    # Physiological BPM bounds
    BPM_LOW  = 45
    BPM_HIGH = 180

    # Bandpass filter frequencies (Hz)
    FREQ_LOW  = BPM_LOW  / 60.0   # 0.75 Hz
    FREQ_HIGH = BPM_HIGH / 60.0   # 3.00 Hz

    def __init__(self, fps: float = 30.0, window_sec: float = 10.0):
        """
        Args:
            fps:        Camera frames per second (updated dynamically)
            window_sec: Sliding window length in seconds for HR estimation
        """
        self.fps        = fps
        self.window_sec = window_sec
        self.window_len = int(fps * window_sec)

        # RGB signal buffers
        self.r_buf = deque(maxlen=self.window_len)
        self.g_buf = deque(maxlen=self.window_len)
        self.b_buf = deque(maxlen=self.window_len)

        self.alpha_history = deque(maxlen=10)

        # Timing for dynamic FPS estimation
        self._frame_times: deque = deque(maxlen=60)

        # Output state
        self.heart_rate   = 0.0
        self.confidence   = 0.0
        self.pulse_signal = np.array([])
        self.bpm_history: deque = deque(maxlen=10)

        # NEW: latest single pulse sample for respiratory processor
        self.latest_pulse_sample: float = 0.0

        # Minimum frames before we attempt estimation
        self.MIN_FRAMES = int(fps * 3)   # 3 seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_fps(self, fps: float) -> None:
        """Dynamically update FPS and resize buffers accordingly."""
        if abs(fps - self.fps) > 1.0 and 5 < fps < 120:
            self.fps = fps
            new_len  = int(fps * self.window_sec)
            self.window_len = new_len
            self.r_buf = deque(self.r_buf, maxlen=new_len)
            self.g_buf = deque(self.g_buf, maxlen=new_len)
            self.b_buf = deque(self.b_buf, maxlen=new_len)
            self.MIN_FRAMES = int(fps * 3)

    def add_frame(self, roi_rgb: np.ndarray) -> dict:
        """
        Process a single face ROI frame.

        Args:
            roi_rgb: H×W×3 uint8 array (RGB order)

        Returns:
            dict with keys: heart_rate, confidence, bpm_min, bpm_max,
                            pulse_signal, signal_quality, status,
                            latest_pulse_sample
        """
        now = time.time()
        self._frame_times.append(now)
        if len(self._frame_times) >= 10:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            if elapsed > 0:
                measured_fps = (len(self._frame_times) - 1) / elapsed
                measured_fps = np.clip(measured_fps, 24, 35)
                self.update_fps(measured_fps)

        # Append mean RGB of ROI
        mean_rgb = roi_rgb.mean(axis=(0, 1)).astype(np.float64)
        self.r_buf.append(mean_rgb[0])
        self.g_buf.append(mean_rgb[1])
        self.b_buf.append(mean_rgb[2])

        n = len(self.r_buf)
        if n < self.MIN_FRAMES:
            remaining = self.MIN_FRAMES - n
            self.latest_pulse_sample = 0.0
            return {
                "heart_rate":          0,
                "confidence":          0.0,
                "bpm_min":             0,
                "bpm_max":             0,
                "pulse_signal":        [],
                "signal_quality":      0.0,
                "latest_pulse_sample": 0.0,
                "status": f"Collecting data… {remaining} frames remaining",
            }

        return self._estimate_hr()

    # ------------------------------------------------------------------
    # Core rPPG: CHROM method
    # ------------------------------------------------------------------

    def _estimate_hr(self) -> dict:
        R = np.array(self.r_buf, dtype=np.float64)
        G = np.array(self.g_buf, dtype=np.float64)
        B = np.array(self.b_buf, dtype=np.float64)

        # 1. Temporal normalisation
        R_n, G_n, B_n = self._normalise(R, G, B)

        R_n = detrend(R_n)
        G_n = detrend(G_n)
        B_n = detrend(B_n)

        R_n = self._standardise(R_n)
        G_n = self._standardise(G_n)
        B_n = self._standardise(B_n)

        # 2. CHROM projection
        X = 3*R_n - 2*G_n
        Y = 1.5*R_n + G_n - 1.5*B_n

        # 3. Bandpass filter
        Xf = self._bandpass(X)
        Yf = self._bandpass(Y)

        if Xf is None or Yf is None:
            self.latest_pulse_sample = 0.0
            return self._error_result("Signal too noisy")

        # 4. Alpha blending ratio
        alpha = Xf.std() / (Yf.std() + 1e-9)
        self.alpha_history.append(alpha)
        alpha = np.mean(self.alpha_history)

        # 5. Extract pulse waveform
        pulse_filt = Xf - alpha * Yf
        self.pulse_signal = pulse_filt

        # Store latest sample for RespiratoryProcessor
        self.latest_pulse_sample = float(pulse_filt[-1])

        # 6. Frequency analysis (Welch PSD)
        hr_bpm, confidence, bpm_min, bpm_max = self._welch_hr(pulse_filt)

        # Time-domain HR using peaks
        peaks, _ = find_peaks(pulse_filt, distance=self.fps / 2)
        if len(peaks) > 1:
            intervals = np.diff(peaks) / self.fps
            hr_time   = 60.0 / np.mean(intervals)
            hr_bpm    = 0.7 * hr_bpm + 0.3 * hr_time

        # 7. Temporal smoothing
        self.bpm_history.append(hr_bpm)
        if len(self.bpm_history) > 1:
            recent  = list(self.bpm_history)[-6:]
            weights = np.linspace(0.5, 1.0, len(recent))
            smoothed_hr = float(np.average(recent, weights=weights))
        else:
            smoothed_hr = hr_bpm

        self.heart_rate = smoothed_hr
        self.confidence = confidence

        quality = self._signal_quality(pulse_filt)

        if confidence < 0.3:
            smoothed_hr = self.heart_rate

        return {
            "heart_rate":          round(smoothed_hr, 1),
            "confidence":          round(confidence, 3),
            "bpm_min":             bpm_min,
            "bpm_max":             bpm_max,
            "pulse_signal":        pulse_filt[-60:].tolist(),
            "signal_quality":      round(quality, 2),
            "latest_pulse_sample": self.latest_pulse_sample,
            "status":              "Measuring",
        }

    # ------------------------------------------------------------------
    # Signal processing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(R, G, B, win: int = 32):
        def _local_mean_norm(x, w):
            pad_w    = w // 2
            x_padded = np.pad(x, (pad_w, w - pad_w - 1), mode='edge')
            kernel   = np.ones(w) / w
            local_mean = np.convolve(x_padded, kernel, mode='valid')
            local_mean = np.clip(local_mean, 1e-6, None)
            return x / local_mean
        return (_local_mean_norm(R, win),
                _local_mean_norm(G, win),
                _local_mean_norm(B, win))

    @staticmethod
    def _standardise(x: np.ndarray) -> np.ndarray:
        std = x.std()
        if std < 1e-9:
            return x
        return (x - x.mean()) / std

    def _bandpass(self, signal: np.ndarray) -> np.ndarray | None:
        nyq = self.fps / 2.0
        if self.heart_rate > 0:
            center = self.heart_rate / 60.0
            low    = max(self.FREQ_LOW,  center - 0.5)
            high   = min(self.FREQ_HIGH, center + 0.5)
        else:
            low  = self.FREQ_LOW
            high = self.FREQ_HIGH

        low  = low  / nyq
        high = high / nyq
        low  = np.clip(low,  0.01, 0.99)
        high = np.clip(high, 0.01, 0.99)
        if low >= high:
            return None
        try:
            b, a   = butter(4, [low, high], btype='band')
            padlen = min(3 * max(len(a), len(b)), len(signal) - 1)
            return filtfilt(b, a, signal, padlen=padlen)
        except Exception:
            return None

    def _welch_hr(self, signal: np.ndarray):
        nperseg = min(len(signal), int(self.fps * 6))
        freqs, psd = welch(signal, fs=self.fps, nperseg=nperseg,
                           nfft=2048, window='hann')

        mask = (freqs >= self.FREQ_LOW) & (freqs <= self.FREQ_HIGH)
        if not mask.any():
            return 0.0, 0.0, 0, 0

        freqs_m  = freqs[mask]
        psd_m    = psd[mask]
        peak_idx = np.argmax(psd_m)
        hr_hz    = freqs_m[peak_idx]
        hr_bpm   = hr_hz * 60.0

        total_power = psd_m.sum() + 1e-12
        confidence  = float(psd_m[peak_idx] / total_power)

        half_power = psd_m[peak_idx] / 2
        band_mask  = psd_m >= half_power
        band_freqs = freqs_m[band_mask]
        bpm_min = int(band_freqs.min() * 60) if len(band_freqs) else int(hr_bpm - 3)
        bpm_max = int(band_freqs.max() * 60) if len(band_freqs) else int(hr_bpm + 3)

        return float(hr_bpm), confidence, bpm_min, bpm_max

    @staticmethod
    def _signal_quality(signal: np.ndarray) -> float:
        ac  = signal - signal.mean()
        snr = (ac ** 2).mean() / ((signal ** 2).mean() + 1e-9)
        return float(np.clip(snr * 10, 0.0, 1.0))

    @staticmethod
    def _error_result(msg: str) -> dict:
        return {
            "heart_rate": 0, "confidence": 0.0,
            "bpm_min": 0,    "bpm_max": 0,
            "pulse_signal": [], "signal_quality": 0.0,
            "latest_pulse_sample": 0.0,
            "status": msg,
        }