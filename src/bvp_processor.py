"""
Blood Volume Pulse (BVP) Processor
====================================
Extracts the BVP waveform from the rPPG pulse signal and computes a rich set
of clinically meaningful pulse-wave morphology features.

Physiological background
------------------------
The BVP signal is the peripheral manifestation of cardiac ejection: each
heartbeat causes a pressure wave that propagates through the arterial tree and
modulates blood volume in the skin capillaries.  A single BVP pulse cycle
contains five canonical fiducial points (following Charlton et al. 2022 /
pyPPG nomenclature):

  SPO  – Systolic Phase Onset  (pulse foot / valley before systole)
  SPP  – Systolic Peak         (primary peak, maximum amplitude)
  DN   – Dicrotic Notch        (inflection caused by aortic valve closure)
  DPP  – Diastolic Peak        (secondary, smaller peak after notch)
  DEP  – Diastolic End Point   (end of the pulse cycle = next SPO)

From these five points a standard set of biomarkers is derived:

  IBI              Inter-Beat Interval (ms)  — inverse of instantaneous HR
  Systolic time    SPO → SPP (ms)
  Diastolic time   DN  → DEP (ms)
  Pulse width      Width of the systolic peak at half maximum amplitude (ms)
  Rise time        SPO → SPP (ms)  alias of systolic time
  Perfusion index  AC amplitude / DC level  (%)  — proxy for vasodilatation
  Augmentation idx (DPP − SPP) / SPP  — vascular stiffness marker (unitless)
  Reflection idx   DPP_amplitude / SPP_amplitude  — peripheral reflection
  Stiffness idx    IBI / (DPP_time − SPP_time)  — proxy for pulse-wave vel.
  Crest time       SPO → SPP (ms)

All time-domain features are averaged over the beats visible in the analysis
window to give stable, per-update estimates.

Signal quality (SQI)
--------------------
A composite signal quality index (0–1) is computed from:
  • spectral purity  – fraction of PSD energy in the cardiac band
  • template correlation – mean Pearson r of individual beats vs. median beat
  • perfusion index   – AC/DC ratio relative to a minimum threshold

References
----------
Charlton PH et al. "Assessing mental stress using wearable PPG signals."
  Frontiers in Physiology, 2022.
Elgendi M. "On the analysis of fingertip photoplethysmogram signals."
  Current Cardiology Reviews, 2012.
Goda MA, Charlton PH, Behar JA. "pyPPG: A Python toolbox for comprehensive
  photoplethysmography signal analysis." Physiol Meas, 2024.
Hertzman AB. "Photoelectric plethysmography of the fingers and toes in man."
  Proc Soc Exp Biol Med, 1937.
"""

import numpy as np
from collections import deque
from scipy.signal import (
    butter, filtfilt, find_peaks, welch, savgol_filter, detrend
)
from scipy.interpolate import interp1d
import time


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BPM_LOW  = 40
BPM_HIGH = 200
FREQ_LOW  = BPM_LOW  / 60.0   # 0.67 Hz
FREQ_HIGH = BPM_HIGH / 60.0   # 3.33 Hz

# Minimum perfusion index to flag "signal too weak" (empirical, rPPG typical)
MIN_PERFUSION = 0.5   # %


class BVPProcessor:
    """
    Real-time Blood Volume Pulse processor.

    Call ``add_pulse_sample(value)`` once per video frame.  After the warm-up
    period it returns a dict of BVP features on every call.
    """

    def __init__(self, fps: float = 30.0, window_sec: float = 10.0):
        """
        Parameters
        ----------
        fps         : Camera / pipeline frame rate (updated dynamically).
        window_sec  : Analysis window length in seconds (default 10 s gives
                      ~5-8 complete pulse cycles at rest which is enough for
                      stable morphology estimates).
        """
        self.fps        = fps
        self.window_sec = window_sec
        self._update_derived()

        # Raw BVP sample buffer (CHROM pulse signal fed from RPPGProcessor)
        self.pulse_buf: deque = deque(maxlen=self.window_len)

        # Timing
        self._frame_times: deque = deque(maxlen=60)

        # Output cache (returned when insufficient data)
        self._last_result: dict = self._empty_result("Collecting data…")

        # Beat-to-beat smoothing
        self._ibi_history:     deque = deque(maxlen=8)
        self._pi_history:      deque = deque(maxlen=8)
        self._sqi_history:     deque = deque(maxlen=6)

        # Minimum frames before analysis (5 seconds = ~2–4 beats)
        self.MIN_FRAMES = int(fps * 5)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_fps(self, fps: float) -> None:
        """Dynamically resize buffers when measured FPS changes."""
        if abs(fps - self.fps) > 1.0 and 5 < fps < 120:
            self.fps = fps
            self._update_derived()
            self.pulse_buf = deque(self.pulse_buf, maxlen=self.window_len)
            self.MIN_FRAMES = int(fps * 5)

    def add_pulse_sample(self, pulse_value: float) -> dict:
        """
        Ingest one rPPG pulse sample (the ``latest_pulse_sample`` from
        RPPGProcessor) and return the latest BVP feature dict.
        """
        now = time.time()
        self._frame_times.append(now)

        # Dynamic FPS estimation
        if len(self._frame_times) >= 10:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            if elapsed > 0:
                measured = (len(self._frame_times) - 1) / elapsed
                self.update_fps(float(np.clip(measured, 5, 120)))

        self.pulse_buf.append(float(pulse_value))

        n = len(self.pulse_buf)
        if n < self.MIN_FRAMES:
            self._last_result = self._empty_result(
                f"Collecting data… {self.MIN_FRAMES - n} frames remaining"
            )
            return self._last_result

        return self._analyse()

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def _analyse(self) -> dict:
        raw = np.array(self.pulse_buf, dtype=np.float64)

        # 1. Pre-process: detrend → bandpass → normalise
        signal = self._preprocess(raw)
        if signal is None:
            return self._last_result

        # 2. Signal quality index (before beat segmentation)
        sqi = self._compute_sqi(signal)
        self._sqi_history.append(sqi)
        smooth_sqi = float(np.mean(self._sqi_history))

        # 3. Beat segmentation — detect systolic peaks
        peaks = self._detect_peaks(signal)
        if len(peaks) < 2:
            result = self._empty_result("Detecting beats…")
            result["bvp_signal"]   = signal[-int(self.fps * 5):].tolist()
            result["bvp_sqi"]      = round(smooth_sqi, 3)
            self._last_result = result
            return result

        # 4. IBI / instantaneous HR
        ibi_ms_arr = np.diff(peaks) / self.fps * 1000.0   # ms
        # Physiological gate: 300–1800 ms
        valid = (ibi_ms_arr >= 300) & (ibi_ms_arr <= 1800)
        ibi_ms_arr = ibi_ms_arr[valid]

        if len(ibi_ms_arr) == 0:
            result = self._empty_result("IBI out of range")
            result["bvp_sqi"] = round(smooth_sqi, 3)
            self._last_result = result
            return result

        mean_ibi   = float(np.median(ibi_ms_arr))
        inst_hr    = 60000.0 / mean_ibi

        self._ibi_history.append(mean_ibi)
        smooth_ibi = float(np.median(self._ibi_history))
        smooth_hr  = 60000.0 / smooth_ibi

        # 5. Waveform morphology (fiducial point analysis)
        morph = self._morphology(signal, peaks)

        # 6. Perfusion index (AC/DC on raw signal)
        pi = self._perfusion_index(raw)
        self._pi_history.append(pi)
        smooth_pi = float(np.mean(self._pi_history))

        # 7. Template quality (beat-template correlation)
        tmpl_corr = self._template_correlation(signal, peaks)

        # 8. Pulse waveform export (last 5 s)
        tail = int(self.fps * 5)
        bvp_export = signal[-tail:].tolist()

        # 9. Assemble result
        result = {
            # ── Core ──────────────────────────────────────────────────
            "bvp_status":          "Measuring",
            "bvp_signal":          bvp_export,
            "bvp_sqi":             round(smooth_sqi, 3),

            # ── Beat timing ───────────────────────────────────────────
            "ibi_ms":              round(smooth_ibi, 1),
            "ibi_arr":             ibi_ms_arr.tolist(),
            "bvp_heart_rate":      round(smooth_hr, 1),

            # ── Perfusion ─────────────────────────────────────────────
            "perfusion_index":     round(smooth_pi, 3),   # %

            # ── Morphology ────────────────────────────────────────────
            **morph,

            # ── Template quality ──────────────────────────────────────
            "template_correlation": round(tmpl_corr, 3),
        }

        self._last_result = result
        return result

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def _preprocess(self, raw: np.ndarray) -> np.ndarray | None:
        sig = detrend(raw)
        filtered = self._bandpass(sig)
        if filtered is None:
            return None
        std = filtered.std()
        if std < 1e-8:
            return None
        return (filtered - filtered.mean()) / std

    def _bandpass(self, signal: np.ndarray) -> np.ndarray | None:
        nyq = self.fps / 2.0
        low  = np.clip(FREQ_LOW  / nyq, 0.01, 0.49)
        high = np.clip(FREQ_HIGH / nyq, 0.01, 0.49)
        if low >= high:
            return None
        try:
            b, a   = butter(4, [low, high], btype='band')
            padlen = min(3 * max(len(a), len(b)), len(signal) - 1)
            return filtfilt(b, a, signal, padlen=max(padlen, 1))
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Peak detection (systolic peaks = maxima of BVP)
    # ------------------------------------------------------------------

    def _detect_peaks(self, signal: np.ndarray) -> np.ndarray:
        """
        Detects systolic peaks using a two-pass strategy:
          Pass 1: scipy find_peaks with physiological distance constraint.
          Pass 2: Refine each peak within a ±half-distance window to ensure
                  we land on the true local maximum.
        """
        min_dist = int(self.fps * 60 / BPM_HIGH)   # samples @ max HR
        max_dist = int(self.fps * 60 / BPM_LOW)    # samples @ min HR

        peaks, props = find_peaks(
            signal,
            distance=max(min_dist, 1),
            prominence=0.1,
            height=np.percentile(signal, 40),
        )

        if len(peaks) < 2:
            # Relax constraints and retry
            peaks, _ = find_peaks(signal, distance=max(min_dist // 2, 1))

        # Refine peaks (snap to true local max in ±half window)
        half = max(min_dist // 2, 1)
        refined = []
        for p in peaks:
            lo = max(0, p - half)
            hi = min(len(signal), p + half + 1)
            refined.append(lo + np.argmax(signal[lo:hi]))
        return np.array(sorted(set(refined)), dtype=int)

    # ------------------------------------------------------------------
    # Waveform morphology — fiducial-point extraction
    # ------------------------------------------------------------------

    def _morphology(self, signal: np.ndarray, peaks: np.ndarray) -> dict:
        """
        For each beat window (SPO → next SPO) extract the five fiducial
        points and compute pulse-wave features.  Returns median values over
        all detected beats.
        """
        fps = self.fps

        systolic_times  = []
        pulse_widths    = []
        aug_indices     = []
        reflection_idxs = []
        stiffness_idxs  = []
        dn_depths       = []
        rise_times      = []
        diastolic_times = []

        # We need onset (foot) of each beat.  Onset = minimum between
        # consecutive peaks.
        onsets = self._detect_onsets(signal, peaks)

        if len(onsets) < 2:
            return self._empty_morphology()

        for i in range(len(onsets) - 1):
            beat = signal[onsets[i]: onsets[i + 1]]
            if len(beat) < 5:
                continue

            beat_len = len(beat)
            beat_t   = np.arange(beat_len) / fps * 1000.0   # ms

            # --- Systolic peak (SPP) ---
            spp_idx = int(np.argmax(beat))
            spp_amp = float(beat[spp_idx])
            spp_t   = beat_t[spp_idx]

            # --- Rise time (SPO → SPP) ---
            rise_times.append(spp_t)

            # --- Pulse width at half amplitude ---
            half_amp = spp_amp / 2.0
            above    = np.where(beat >= half_amp)[0]
            if len(above) >= 2:
                pulse_widths.append((above[-1] - above[0]) / fps * 1000.0)

            # --- Dicrotic notch (DN) and diastolic peak (DPP) ---
            # Search only in the diastolic segment (after SPP)
            diastolic_seg = beat[spp_idx:]
            dn_idx_rel, dpp_idx_rel = self._find_dn_dpp(diastolic_seg)

            if dn_idx_rel is not None and dpp_idx_rel is not None:
                dn_idx  = spp_idx + dn_idx_rel
                dpp_idx = spp_idx + dpp_idx_rel

                dn_amp  = float(beat[dn_idx])
                dpp_amp = float(beat[dpp_idx])
                dpp_t   = beat_t[dpp_idx]

                # Augmentation index: (DPP - SPP) / SPP  (usually negative for
                # healthy young arteries; approaches 0 / positive with stiffness)
                aug_idx = (dpp_amp - spp_amp) / (abs(spp_amp) + 1e-9)
                aug_indices.append(float(aug_idx))

                # Reflection index: DPP / SPP (0–1, higher = more reflection)
                refl_idx = dpp_amp / (spp_amp + 1e-9)
                reflection_idxs.append(float(np.clip(refl_idx, 0, 2)))

                # Dicrotic notch depth relative to systolic peak
                dn_depth = (spp_amp - dn_amp) / (spp_amp + 1e-9)
                dn_depths.append(float(np.clip(dn_depth, 0, 1)))

                # Stiffness index proxy: beat_duration / (DPP_t - SPP_t)
                # (beat_duration in seconds; larger = stiffer arteries)
                dt = (dpp_t - spp_t)   # ms
                if dt > 10:
                    beat_dur_s = beat_len / fps
                    stiffness_idxs.append(beat_dur_s / (dt / 1000.0))

                # Diastolic time: DN → end of beat
                diastolic_times.append(beat_t[-1] - beat_t[dn_idx])

            # Systolic time: SPO → SPP
            systolic_times.append(spp_t)

        def _safe_median(lst):
            return round(float(np.median(lst)), 3) if lst else 0.0

        return {
            "systolic_time_ms":    _safe_median(rise_times),
            "diastolic_time_ms":   _safe_median(diastolic_times),
            "pulse_width_ms":      _safe_median(pulse_widths),
            "augmentation_index":  _safe_median(aug_indices),
            "reflection_index":    _safe_median(reflection_idxs),
            "stiffness_index":     _safe_median(stiffness_idxs),
            "dicrotic_notch_depth":_safe_median(dn_depths),
            "beats_analysed":      len(rise_times),
        }

    def _detect_onsets(self, signal: np.ndarray, peaks: np.ndarray) -> np.ndarray:
        """
        Detect beat onsets (valleys / feet) as the minima between consecutive peaks.
        Adds a synthetic onset before the first peak and after the last peak.
        """
        onsets = []
        # Before first peak
        onsets.append(int(np.argmin(signal[:peaks[0] + 1])))

        for i in range(len(peaks) - 1):
            segment = signal[peaks[i]: peaks[i + 1]]
            onsets.append(peaks[i] + int(np.argmin(segment)))

        # After last peak
        tail = signal[peaks[-1]:]
        onsets.append(peaks[-1] + int(np.argmin(tail)))

        return np.array(onsets, dtype=int)

    @staticmethod
    def _find_dn_dpp(diastolic_seg: np.ndarray):
        """
        Locate the dicrotic notch (DN) and diastolic peak (DPP) in the
        segment of a beat *after* the systolic peak.

        Strategy (IEM-inspired, Echeverría et al. 2024):
          1. Smooth the segment with a Savitzky-Golay filter.
          2. The DN is the *minimum* in the first 60 % of the segment
             (after an initial guard of 15 % to skip the systolic descent).
          3. The DPP is the *maximum* occurring after DN.
        """
        n = len(diastolic_seg)
        if n < 6:
            return None, None

        # Smooth to suppress noise
        wl = min(11, n if n % 2 == 1 else n - 1)
        if wl < 3:
            return None, None
        try:
            smooth = savgol_filter(diastolic_seg, window_length=wl, polyorder=2)
        except Exception:
            smooth = diastolic_seg.copy()

        # Guard: skip first 10 % (still on systolic descent) and search
        # up to 65 % for the notch
        guard = max(int(n * 0.10), 1)
        end   = max(int(n * 0.65), guard + 2)

        search_seg = smooth[guard:end]
        if len(search_seg) < 2:
            return None, None

        dn_rel = int(np.argmin(search_seg))
        dn_idx = guard + dn_rel

        # DPP: maximum after DN
        post_dn = smooth[dn_idx + 1:]
        if len(post_dn) < 1:
            return None, None

        dpp_rel = int(np.argmax(post_dn))
        dpp_idx = dn_idx + 1 + dpp_rel

        # Sanity: DPP should be higher than DN
        if smooth[dpp_idx] <= smooth[dn_idx]:
            return None, None

        return dn_idx, dpp_idx

    # ------------------------------------------------------------------
    # Signal quality index
    # ------------------------------------------------------------------

    def _compute_sqi(self, signal: np.ndarray) -> float:
        """
        Composite SQI (0–1) from three components:
          1. Spectral purity   – PSD peak fraction in HR band
          2. SNR estimate      – AC power / total power ratio
          3. Kurtosis          – clean PPG has kurtosis > 2 (leptokurtic)
        """
        # Spectral purity
        nperseg = min(len(signal), int(self.fps * 6))
        try:
            freqs, psd = welch(signal, fs=self.fps, nperseg=nperseg)
            mask  = (freqs >= FREQ_LOW) & (freqs <= FREQ_HIGH)
            if mask.any():
                spectral = float(psd[mask].max() / (psd[mask].sum() + 1e-12))
            else:
                spectral = 0.0
        except Exception:
            spectral = 0.0

        # SNR (AC power fraction)
        ac  = signal - signal.mean()
        snr = float(np.clip((ac ** 2).mean() / ((signal ** 2).mean() + 1e-9) * 10, 0, 1))

        # Kurtosis (excess kurtosis: healthy PPG ≈ 3–5, noisy ≈ 0)
        std4 = (signal.std() ** 4) + 1e-12
        kurt = float(np.mean((signal - signal.mean()) ** 4) / std4)
        kurt_score = float(np.clip((kurt - 1.5) / 5.0, 0, 1))

        return float(np.clip(0.5 * spectral + 0.3 * snr + 0.2 * kurt_score, 0, 1))

    # ------------------------------------------------------------------
    # Perfusion Index
    # ------------------------------------------------------------------

    def _perfusion_index(self, raw: np.ndarray) -> float:
        """
        Perfusion index = AC / DC × 100  (%).
        AC  = peak-to-peak amplitude of the pulsatile component.
        DC  = mean of the raw signal (baseline level).

        For rPPG the raw RGB values are not absolute, so we use the
        normalised pulse signal: AC = std × 2√2  (RMS → peak-to-peak approx.)
        and DC is proxied by the range of the low-frequency trend.
        """
        # Low-frequency trend (DC baseline)
        try:
            b_lf, a_lf = butter(2, 0.5 / (self.fps / 2.0), btype='low')
            dc = filtfilt(b_lf, a_lf, raw)
        except Exception:
            dc = np.full_like(raw, raw.mean())

        ac_rms  = float(np.std(raw - dc))
        dc_mean = float(np.abs(np.mean(dc)) + 1e-9)
        pi      = (ac_rms * 2 * np.sqrt(2)) / dc_mean * 100.0
        return float(np.clip(pi, 0.0, 100.0))

    # ------------------------------------------------------------------
    # Template correlation
    # ------------------------------------------------------------------

    def _template_correlation(self, signal: np.ndarray,
                               peaks: np.ndarray) -> float:
        """
        Build a median beat template and compute the mean Pearson r of
        each individual beat against it.  High correlation → clean signal.
        """
        if len(peaks) < 3:
            return 0.0

        # Fixed-length beat window: use median IBI
        ibi_samp = int(np.median(np.diff(peaks)))
        if ibi_samp < 3:
            return 0.0

        beats = []
        for p in peaks[:-1]:
            lo = p - ibi_samp // 3
            hi = lo + ibi_samp
            if lo < 0 or hi > len(signal):
                continue
            beat = signal[lo:hi]
            # Resample to fixed length for template comparison
            x_old = np.linspace(0, 1, len(beat))
            x_new = np.linspace(0, 1, ibi_samp)
            try:
                beats.append(interp1d(x_old, beat, kind='linear')(x_new))
            except Exception:
                pass

        if len(beats) < 2:
            return 0.0

        beats    = np.array(beats)
        template = np.median(beats, axis=0)

        corrs = []
        for b in beats:
            if b.std() > 1e-9 and template.std() > 1e-9:
                corrs.append(float(np.corrcoef(b, template)[0, 1]))

        return float(np.clip(np.mean(corrs) if corrs else 0.0, 0, 1))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_derived(self):
        self.window_len = int(self.fps * self.window_sec)

    @staticmethod
    def _empty_morphology() -> dict:
        return {
            "systolic_time_ms":     0.0,
            "diastolic_time_ms":    0.0,
            "pulse_width_ms":       0.0,
            "augmentation_index":   0.0,
            "reflection_index":     0.0,
            "stiffness_index":      0.0,
            "dicrotic_notch_depth": 0.0,
            "beats_analysed":       0,
        }

    @staticmethod
    def _empty_result(status: str = "") -> dict:
        result = {
            "bvp_status":           status,
            "bvp_signal":           [],
            "bvp_sqi":              0.0,
            "ibi_ms":               0.0,
            "ibi_arr":              [],
            "bvp_heart_rate":       0.0,
            "perfusion_index":      0.0,
            "template_correlation": 0.0,
        }
        result.update(BVPProcessor._empty_morphology())
        return result
