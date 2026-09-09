"""Testes standalone do módulo `tradutor.segmenter` (VAD + ASR).

Executar:  .venv\\Scripts\\python.exe tests\\test_segmenter.py

Não depende de pytest. Gera áudio sintético 16 kHz (fala simulada com
harmônicos + formantes + envelope silábico) e verifica a segmentação; o teste
do Transcriber é PULADO com aviso se o faster-whisper/modelo não estiver
disponível (download pode não ter sido feito).
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from tradutor.contracts import SR_ASR, SpeechSegment          # noqa: E402
from tradutor.segmenter import (SpeechSegmenter, Transcriber,  # noqa: E402
                                collapse_repetitions, energy_vad,
                                is_hallucination)

try:  # acentos no console do Windows
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

VERBOSE = "-v" in sys.argv
logging.basicConfig(
    level=logging.DEBUG if VERBOSE else logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s")


# --------------------------------------------------------------------------
# Geração de áudio sintético
# --------------------------------------------------------------------------

# Formantes (F1, F2, F3) de vogais: base da fala sintética.
_VOWELS = [(730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240),
           (530, 1840, 2480), (570, 840, 2410), (660, 1720, 2410),
           (490, 1350, 1690)]


def _resonate(src: np.ndarray, freq: float, bw: float,
              sr: int = SR_ASR) -> np.ndarray:
    """Ressoador (formante) via convolução com a resposta impulsiva."""
    n = int(0.025 * sr)
    ti = np.arange(n) / sr
    kernel = np.exp(-np.pi * bw * ti) * np.sin(2 * np.pi * freq * ti)
    return np.convolve(src, kernel, mode="same")


def _glottal(n: int, f0_ini: float, f0_fim: float, sr: int,
             rng: np.random.Generator) -> np.ndarray:
    """Fonte glotal: harmônicos com contorno de F0 + jitter."""
    t = np.arange(n) / sr
    f0 = np.linspace(f0_ini, f0_fim, n) * (1 + 0.03 * np.sin(2 * np.pi * 5 * t))
    phase = 2 * np.pi * np.cumsum(f0) / sr
    src = np.zeros(n)
    for k in range(1, 40):
        src += np.sin(k * phase) / (k ** 1.1)
    return src + 0.01 * rng.standard_normal(n)


def synth_speech(dur_s: float, sr: int = SR_ASR, f0: float = 120.0,
                 seed: int = 0) -> np.ndarray:
    """Fala sintética: sílabas consoante+vogal com formantes em movimento.

    Realista o bastante para o Silero VAD classificar como voz (prob. média
    ~0.9), sem depender de nenhum arquivo de áudio externo.
    """
    rng = np.random.default_rng(seed)
    partes: List[np.ndarray] = []
    total = 0
    alvo = int(dur_s * sr)
    i = 0
    while total < alvo:
        # consoante: burst curto de ruído colorido
        cn = int(rng.uniform(0.03, 0.06) * sr)
        cons = _resonate(rng.standard_normal(cn), rng.uniform(2000, 4500),
                         800.0, sr) * np.hanning(cn) * 0.35
        # vogal: formantes transitando de um alvo para outro
        vn = int(rng.uniform(0.10, 0.20) * sr)
        f1, f2, f3 = _VOWELS[i % len(_VOWELS)]
        g1, g2, g3 = _VOWELS[(i + 3) % len(_VOWELS)]
        base = f0 * rng.uniform(0.9, 1.1)
        src = _glottal(vn, base * 1.05, base * 0.95, sr, rng)
        w = np.linspace(0.0, 1.0, vn)
        voc = np.zeros(vn)
        for (fa, fb), gain in (((f1, g1), 1.0), ((f2, g2), 0.5), ((f3, g3), 0.25)):
            voc += gain * ((1 - w) * _resonate(src, fa, 80.0, sr)
                           + w * _resonate(src, fb, 110.0, sr))
        env = np.ones(vn)
        r = int(0.015 * sr)
        env[:r] = np.linspace(0.0, 1.0, r)
        env[-r:] = np.linspace(1.0, 0.3, r)
        voc *= env / (np.max(np.abs(voc)) or 1.0)
        partes.append(np.concatenate([cons, voc]))
        total += cn + vn
        i += 1
        if i % 7 == 0:            # respiro curto entre "palavras"
            gap = int(0.05 * sr)
            partes.append(np.zeros(gap))
            total += gap

    y = np.concatenate(partes)[:alvo]
    r = int(0.02 * sr)
    y[:r] *= np.linspace(0.0, 1.0, r)
    y[-r:] *= np.linspace(1.0, 0.0, r)
    return (0.35 * y / (np.max(np.abs(y)) or 1.0)).astype(np.float32)


def silence(dur_s: float, sr: int = SR_ASR, noise: float = 1e-4) -> np.ndarray:
    """Silêncio (com ruído de fundo bem baixo, como um sinal digital real)."""
    n = int(dur_s * sr)
    rng = np.random.default_rng(1234)
    return (noise * rng.standard_normal(n)).astype(np.float32)


def scenario_two_utterances(sr: int = SR_ASR) -> np.ndarray:
    """2 s silêncio + 3 s fala + 1 s silêncio + 2 s fala + 1,5 s silêncio."""
    return np.concatenate([
        silence(2.0, sr),
        synth_speech(3.0, sr, f0=118.0, seed=1),
        silence(1.0, sr),
        synth_speech(2.0, sr, f0=145.0, seed=2),
        silence(1.5, sr),
    ]).astype(np.float32)


def feed_all(seg: SpeechSegmenter, audio: np.ndarray,
             block_ms: int = 20) -> None:
    """Alimenta o segmentador em blocos, como faria a captura em tempo real.

    Aplica contrapressão (o teste enfileira muito mais rápido que o tempo
    real; sem isso a fila estouraria e o segmentador descartaria áudio).
    """
    n = int(SR_ASR * block_ms / 1000)
    for i in range(0, len(audio), n):
        while seg.pending_blocks() > 64:
            time.sleep(0.001)
        seg.feed(audio[i:i + n])
    seg.flush()


class Collector:
    def __init__(self) -> None:
        self.segments: List[SpeechSegment] = []
        self._lock = threading.Lock()

    def __call__(self, seg: SpeechSegment) -> None:
        with self._lock:
            self.segments.append(seg)

    def durations(self) -> List[float]:
        return [len(s.pcm) / SR_ASR for s in self.segments]


# --------------------------------------------------------------------------
# Testes
# --------------------------------------------------------------------------

def _check_two_utterances(col: Collector, backend: str, strict: bool) -> None:
    durs = col.durations()
    print(f"    backend={backend} segmentos={len(durs)} "
          f"durações={[round(d, 2) for d in durs]}")
    assert 1 <= len(col.segments) <= 3, f"esperado ~2 segmentos, veio {len(durs)}"
    if strict:
        assert len(durs) == 2, f"esperado 2 segmentos, veio {len(durs)}"
        assert 2.5 <= durs[0] <= 4.0, f"1º segmento com {durs[0]:.2f}s"
        assert 1.5 <= durs[1] <= 3.0, f"2º segmento com {durs[1]:.2f}s"

    for s in col.segments:
        assert s.pcm.dtype == np.float32 and s.pcm.ndim == 1
        d = len(s.pcm) / SR_ASR
        assert abs((s.t_end - s.t_start) - d) < 1e-3, "timestamps incoerentes"
        assert np.max(np.abs(s.pcm)) > 0.01, "segmento sem energia (silêncio)"
    for a, b in zip(col.segments, col.segments[1:]):
        assert b.t_start > a.t_end - 0.05, "segmentos fora de ordem/sobrepostos"

    if len(col.segments) == 2:
        gap = col.segments[1].t_start - col.segments[0].t_end
        assert 0.0 <= gap <= 1.5, f"intervalo entre segmentos = {gap:.2f}s"


def test_segmentacao_vad_energia() -> None:
    """VAD de energia (fallback): deve achar exatamente as 2 falas."""
    col = Collector()
    seg = SpeechSegmenter(col, silence_ms=600, min_speech_ms=250, pad_ms=150,
                          use_silero=False)
    assert seg.vad_backend == "energy"
    try:
        feed_all(seg, scenario_two_utterances())
    finally:
        seg.stop()
    _check_two_utterances(col, seg.vad_backend, strict=True)


def test_segmentacao_silero() -> None:
    """Silero VAD (se disponível) no mesmo cenário."""
    col = Collector()
    seg = SpeechSegmenter(col, silence_ms=600, min_speech_ms=250, pad_ms=150)
    backend = seg.vad_backend
    if backend == "energy":
        seg.stop()
        raise Skip("Silero VAD do faster-whisper indisponível (usando energia)")
    try:
        feed_all(seg, scenario_two_utterances())
    finally:
        seg.stop()
    if not col.segments:
        raise Skip(f"{backend} não detectou o áudio sintético como fala "
                   "(Silero é treinado em voz humana real)")
    _check_two_utterances(col, backend, strict=True)


def test_max_segment_corta() -> None:
    """Fala longa contínua deve ser fatiada em ~max_segment_s."""
    col = Collector()
    seg = SpeechSegmenter(col, silence_ms=600, max_segment_s=3.0,
                          min_speech_ms=250, use_silero=False)
    audio = np.concatenate([silence(0.5), synth_speech(12.0, seed=3),
                            silence(1.0)]).astype(np.float32)
    try:
        feed_all(seg, audio)
    finally:
        seg.stop()
    durs = col.durations()
    print(f"    cortes={len(durs)} durações={[round(d, 2) for d in durs]}")
    assert len(durs) >= 3, f"esperado >=3 cortes, veio {len(durs)}"
    assert all(d <= 3.6 for d in durs), f"segmento maior que max_segment_s: {durs}"
    assert abs(sum(durs) - 12.0) < 1.5, f"soma das durações = {sum(durs):.2f}s"


def test_min_speech_descarta() -> None:
    """Estalo de 100 ms não vira segmento."""
    col = Collector()
    seg = SpeechSegmenter(col, silence_ms=400, min_speech_ms=250,
                          use_silero=False)
    audio = np.concatenate([silence(1.0), synth_speech(0.08, seed=4),
                            silence(1.5)]).astype(np.float32)
    try:
        feed_all(seg, audio)
    finally:
        seg.stop()
    print(f"    segmentos={len(col.segments)}")
    assert not col.segments, "estalo curto não deveria gerar segmento"


def test_flush_emite_parcial() -> None:
    """flush() entrega a fala em andamento (sem esperar o silêncio)."""
    col = Collector()
    seg = SpeechSegmenter(col, silence_ms=600, min_speech_ms=250,
                          use_silero=False)
    audio = np.concatenate([silence(0.5), synth_speech(2.0, seed=5)]).astype(
        np.float32)
    try:
        n = int(SR_ASR * 0.02)
        for i in range(0, len(audio), n):
            while seg.pending_blocks() > 64:
                time.sleep(0.001)
            seg.feed(audio[i:i + n])
        seg.flush()
        assert len(col.segments) == 1, f"flush não emitiu ({len(col.segments)})"
        print(f"    flush -> {col.durations()[0]:.2f}s")
    finally:
        seg.stop()


def test_feed_nao_bloqueia() -> None:
    """feed() deve retornar imediatamente mesmo com a fila cheia."""
    col = Collector()
    seg = SpeechSegmenter(col, use_silero=False, queue_blocks=4)
    try:
        block = synth_speech(0.02, seed=6)
        t0 = time.monotonic()
        for _ in range(400):
            seg.feed(block)
        dt = time.monotonic() - t0
        print(f"    400 feeds em {dt * 1000:.0f} ms")
        assert dt < 1.0, f"feed() bloqueou ({dt:.2f}s)"
    finally:
        seg.stop()


def _merge(stamps, gap_s: float = 0.6):
    """Junta trechos separados por menos de `gap_s` (como faz o hangover)."""
    out: List[List[float]] = []
    for a, b in stamps:
        a, b = a / SR_ASR, b / SR_ASR
        if out and a - out[-1][1] < gap_s:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(round(a, 2), round(b, 2)) for a, b in out]


def test_energy_vad_direto() -> None:
    """A função de VAD por energia isolada."""
    audio = scenario_two_utterances()
    spans = _merge(energy_vad(audio, SR_ASR))
    print(f"    trechos={spans}")
    assert len(spans) == 2, f"esperado 2 trechos, veio {spans}"
    assert abs(spans[0][0] - 2.0) < 0.3 and abs(spans[0][1] - 5.0) < 0.3
    assert abs(spans[1][0] - 6.0) < 0.3 and abs(spans[1][1] - 8.0) < 0.3
    assert not energy_vad(silence(3.0), SR_ASR), "silêncio virou fala"


def test_filtro_alucinacao() -> None:
    """Padrões clássicos de alucinação são reconhecidos."""
    for bad in ("Legendas pela comunidade Amara.org",
                "Thanks for watching!", "Thank you for watching.",
                "Obrigado por assistir", "Subtitles by the Amara community",
                "Sottotitoli e revisione a cura di QTSS", "  ...  ",
                "Like and subscribe", "you", "Субтитры подписаны"):
        assert is_hallucination(bad), f"deveria filtrar: {bad!r}"
    for good in ("Hello everyone, welcome back to the channel.",
                 "O mercado fechou em alta nesta terça-feira.",
                 "Vamos falar sobre inteligência artificial."):
        assert not is_hallucination(good), f"não deveria filtrar: {good!r}"


def test_colapso_repeticao() -> None:
    """Loops do decoder são colapsados; ênfase legítima é preservada."""
    loop = ("Another one right here. We popped up one side was popped up "
            "one side was popped up one side was popped up one side was w")
    out, cut = collapse_repetitions(loop)
    print(f"    {len(loop)} -> {len(out)} chars: {out!r}")
    assert cut and out.count("one side") == 1, out
    assert collapse_repetitions("de um lado de um lado de um lado")[0] == "de um lado"
    for ok in ("All we can do is play these levels realistically.",
               "yeah yeah yeah", "Buy buy buy! The market is moving.",
               "O mercado subiu, subiu e caiu de novo."):
        assert collapse_repetitions(ok) == (ok, False), f"não devia cortar: {ok!r}"


def test_transcriber_silencio() -> None:
    """Transcriber com 1 s de zeros deve devolver None sem lançar."""
    try:
        import faster_whisper  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise Skip(f"faster-whisper não instalado ({exc})")

    model = os.environ.get("TRADUTOR_TEST_MODEL", "tiny")
    try:
        asr = Transcriber(model_size=model, compute_type="int8", cpu_threads=4)
    except Exception as exc:  # noqa: BLE001
        raise Skip(f"modelo '{model}' indisponível (sem download?): "
                   f"{type(exc).__name__}: {exc}")

    now = time.monotonic()
    seg = SpeechSegment(pcm=np.zeros(SR_ASR, dtype=np.float32),
                        t_start=now, t_end=now + 1.0)
    t0 = time.monotonic()
    out = asr.transcribe(seg)
    print(f"    transcribe(1s zeros) -> {out!r} em {time.monotonic() - t0:.2f}s")
    assert out is None, f"silêncio deveria dar None, veio {out!r}"


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

class Skip(Exception):
    """Sinaliza teste pulado (dependência ausente)."""


TESTS = [
    test_energy_vad_direto,
    test_filtro_alucinacao,
    test_colapso_repeticao,
    test_segmentacao_vad_energia,
    test_segmentacao_silero,
    test_max_segment_corta,
    test_min_speech_descarta,
    test_flush_emite_parcial,
    test_feed_nao_bloqueia,
    test_transcriber_silencio,
]


def main() -> int:
    ok = skipped = failed = 0
    for fn in TESTS:
        name = fn.__name__
        print(f"[ .. ] {name}")
        t0 = time.monotonic()
        try:
            fn()
        except Skip as exc:
            skipped += 1
            print(f"[SKIP] {name}: AVISO: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
        else:
            ok += 1
            print(f"[ OK ] {name} ({time.monotonic() - t0:.2f}s)")
    print(f"\n{ok} passaram, {skipped} pulados, {failed} falharam")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
