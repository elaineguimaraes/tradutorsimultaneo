"""Utilitários de DSP compartilhados: downmix e resample."""

from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly


def to_mono(pcm: np.ndarray) -> np.ndarray:
    """(n, ch) ou (n,) float32 -> (n,) float32."""
    if pcm.ndim == 1:
        return pcm
    return pcm.mean(axis=1).astype(np.float32)


def resample(pcm: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Resample polifásico de alta qualidade (mono float32)."""
    if sr_from == sr_to:
        return pcm.astype(np.float32, copy=False)
    g = np.gcd(sr_from, sr_to)
    out = resample_poly(pcm.astype(np.float32, copy=False), sr_to // g, sr_from // g)
    return out.astype(np.float32, copy=False)
