# Create /mnt/data/running_baseline.py again
from dataclasses import dataclass
import numpy as np
from scipy.signal import stft

@dataclass
class STFTConfig:
    fs: float               # sampling rate (Hz)
    nperseg: int = 320      # window length (samples)
    noverlap: int = 240     # overlap (samples)
    window: str = "hann"    # window type
    onesided: bool = True

class RunningBaseline:
    """
    Maintain a running baseline of spectral power per-frequency using an EMA,
    and provide baseline-relative contrast (dB) or Z-score.
    """
    def __init__(self, n_freq: int, alpha: float = 0.95):
        self.alpha = float(alpha)
        self.n_freq = int(n_freq)
        self._mean = None
        self._var = None

    def update(self, P_lin: np.ndarray):
        """Update the running baseline with a new power spectrum (linear)."""
        P_lin = np.asarray(P_lin, dtype=float)
        if self._mean is None:
            eps = 1e-20
            self._mean = P_lin.copy()
            self._var = np.ones_like(P_lin) * (np.var(P_lin) + eps)
        else:
            a = self.alpha
            self._mean = a * self._mean + (1.0 - a) * P_lin
            self._var = a * self._var + (1.0 - a) * (P_lin - self._mean) ** 2 + 1e-20

    def contrast_db(self, P_lin: np.ndarray) -> np.ndarray:
        eps = 1e-20
        return 10.0 * np.log10((np.asarray(P_lin) + eps) / (self._mean + eps))

    def zscore(self, P_lin: np.ndarray) -> np.ndarray:
        eps = 1e-20
        return (np.asarray(P_lin) - self._mean) / (np.sqrt(self._var) + eps)

def stft_power_1shot(x: np.ndarray, cfg: STFTConfig):
    """Compute a power spectrum from STFT of chunk `x` by averaging frames."""
    f, t, Z = stft(
        x, fs=cfg.fs, nperseg=cfg.nperseg, noverlap=cfg.noverlap,
        window=cfg.window, return_onesided=cfg.onesided, boundary=None
    )
    P = (np.abs(Z) ** 2)
    P_lin = P.mean(axis=1) if P.ndim == 2 else P
    return f, P_lin

def band_indices(freqs: np.ndarray, bands: dict):
    """Return dict of band -> frequency index array."""
    idx = {}
    for name, (fmin, fmax) in bands.items():
        idx[name] = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    return idx

def aggregate_bands(metric_f: np.ndarray, band_idx: dict):
    """Aggregate per-frequency metric to per-band (mean)."""
    out = {}
    for name, inds in band_idx.items():
        out[name] = float(np.nanmean(metric_f[inds])) if len(inds) else np.nan
    return out