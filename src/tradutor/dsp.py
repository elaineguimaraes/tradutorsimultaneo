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


class StreamResampler:
    """Resample polifásico em blocos, idêntico ao do sinal inteiro.

    `resample()` por trecho cria transitórios nas bordas (o filtro vê zeros
    onde o sinal continua). Aqui cada bloco é processado junto de um contexto
    do bloco anterior e só as saídas cujo filtro já "enxergou" amostras
    suficientes dos dois lados são entregues; o resto espera o próximo bloco
    (ou `flush()`). As fronteiras ficam alinhadas à grade de saída: as
    entregas são sempre múltiplos de `down` amostras de entrada.
    """

    def __init__(self, sr_from: int, sr_to: int) -> None:
        self.sr_from = int(sr_from)
        self.sr_to = int(sr_to)
        g = int(np.gcd(self.sr_from, self.sr_to))
        self._up = self.sr_to // g
        self._down = self.sr_from // g
        # alcance do filtro (resample_poly: meia-largura 10*max(up, down), na
        # taxa interpolada) convertido para amostras de entrada, com folga
        reach = (10 * max(self._up, self._down)) // self._up + 8
        self._ctx_len = -(-reach // self._down) * self._down
        self._buf = np.zeros(0, dtype=np.float32)
        self._ctx = 0   # amostras de contexto (já entregues) no início de _buf

    def process(self, pcm: np.ndarray) -> np.ndarray:
        """Entrega as saídas já definitivas deste bloco (pode ser vazio)."""
        x = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if self.sr_from == self.sr_to:
            return x
        self._buf = np.concatenate((self._buf, x)) if self._buf.size else x.copy()
        free = len(self._buf) - self._ctx - self._ctx_len   # margem à direita
        n_in = (free // self._down) * self._down
        if n_in <= 0:
            return np.zeros(0, dtype=np.float32)
        y = resample_poly(self._buf, self._up, self._down).astype(np.float32,
                                                                  copy=False)
        a = self._ctx * self._up // self._down
        b = (self._ctx + n_in) * self._up // self._down
        out = y[a:b]
        keep = min(self._ctx + n_in, self._ctx_len)   # contexto para o próximo
        cut = self._ctx + n_in - keep
        self._buf = self._buf[cut:]
        self._ctx = keep
        return out

    def flush(self) -> np.ndarray:
        """Fim do sinal: entrega o que restou (borda final igual à do inteiro)."""
        if self.sr_from == self.sr_to or len(self._buf) <= self._ctx:
            self._buf = np.zeros(0, dtype=np.float32)
            self._ctx = 0
            return np.zeros(0, dtype=np.float32)
        y = resample_poly(self._buf, self._up, self._down).astype(np.float32,
                                                                  copy=False)
        out = y[self._ctx * self._up // self._down:]
        self._buf = np.zeros(0, dtype=np.float32)
        self._ctx = 0
        return out
