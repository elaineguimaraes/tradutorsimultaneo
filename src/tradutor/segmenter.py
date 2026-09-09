"""VAD + ASR do Tradutor Simultâneo.

Implementa dois componentes de `contracts.py`:

* :class:`SpeechSegmenter`: segmentação de fala em streaming (Silero VAD do
  pacote ``faster_whisper``, com fallback para VAD de energia RMS).
* :class:`Transcriber`: ASR síncrono com ``faster_whisper.WhisperModel``
  (CPU, int8) e filtro anti-alucinação.

Todo o PCM é ``numpy.float32`` mono em [-1, 1] @ 16 kHz (``SR_ASR``).
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
import queue
import re
import threading
import time
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

try:  # pacote instalado / import relativo normal
    from .contracts import SR_ASR, SpeechSegment, Transcript
except ImportError:  # pragma: no cover - execução direta a partir de src/
    from tradutor.contracts import SR_ASR, SpeechSegment, Transcript  # type: ignore

_LOG_VAD = logging.getLogger("tradutor.vad")
_LOG_ASR = logging.getLogger("tradutor.asr")

# Tipo do detector: recebe áudio mono float32 e devolve [(ini, fim)] em amostras.
VadFn = Callable[[np.ndarray], List[Tuple[int, int]]]


# --------------------------------------------------------------------------
# Filtro anti-alucinação do Whisper
# --------------------------------------------------------------------------

#: Substrings (minúsculas) típicas de alucinação do Whisper em silêncio/música.
HALLUCINATION_PATTERNS: frozenset[str] = frozenset({
    "legendas pela comunidade",
    "amara.org",
    "thanks for watching",
    "thank you for watching",
    "obrigado por assistir",
    "subtitles by",
    "subtitled by",
    "sottotitoli",
    "sous-titres",
    "untertitel",
    "подпис",
    "share this video",
    "like and subscribe",
    "please subscribe",
    "subscribe to my channel",
    "字幕",
    "www.",
})

#: Textos que, sozinhos, quase sempre são alucinação de silêncio.
HALLUCINATION_EXACT: frozenset[str] = frozenset({
    "you", "thank you", "thanks", "bye", "bye.", "obrigado", "obrigada",
    "gracias", "merci", "。", ".", "..", "...", "!", "?", "-", "♪", "♪♪",
    "[music]", "[música]", "(music)", "(música)", "[applause]",
})

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[\s\.\,\!\?\-\–\—\"'…]+")


def _normalize(text: str) -> str:
    """Minúsculas + espaços colapsados (para comparação de alucinações)."""
    return _WS_RE.sub(" ", text.strip().lower())


def is_hallucination(text: str) -> bool:
    """True se o texto for um padrão clássico de alucinação do Whisper."""
    norm = _normalize(text)
    if not norm:
        return True
    if _PUNCT_RE.sub("", norm) == "":       # só pontuação
        return True
    if norm in HALLUCINATION_EXACT:
        return True
    stripped = norm.strip(" .,!?-…\"'")
    if stripped in HALLUCINATION_EXACT:
        return True
    return any(p in norm for p in HALLUCINATION_PATTERNS)


def _key(word: str) -> str:
    """Palavra normalizada para comparar repetições (sem pontuação/caixa)."""
    return _PUNCT_RE.sub("", word.lower())


def collapse_repetitions(text: str, *, max_ngram: int = 8,
                         min_repeats: int = 3) -> Tuple[str, bool]:
    """Colapsa loops de repetição do decoder do Whisper.

    Quando o Whisper entra em loop ele repete o mesmo n-grama várias vezes
    ("popped up one side was popped up one side was ..."). Reduz cada
    sequência repetida a uma única ocorrência e devolve ``(texto, houve_corte)``.
    Um n-grama de uma palavra só é colapsado a partir de ``min_repeats + 1``
    ocorrências (repetir uma palavra três vezes é ênfase legítima).
    """
    words = text.split()
    if len(words) < min(min_repeats + 1, 2 * min_repeats):
        return text, False

    keys = [_key(w) for w in words]
    out: List[str] = []
    i = 0
    cut = False
    while i < len(words):
        best_n = best_reps = 0
        for n in range(1, max_ngram + 1):
            need = min_repeats + 1 if n == 1 else min_repeats
            if i + n * need > len(words):
                break
            gram = keys[i:i + n]
            if not any(gram):                      # só pontuação: ignora
                continue
            reps, j = 1, i + n
            while j + n <= len(words) and keys[j:j + n] == gram:
                reps, j = reps + 1, j + n
            if reps >= need and reps * n > best_reps * best_n:
                best_n, best_reps = n, reps
        if best_n:
            out.extend(words[i:i + best_n])
            i += best_n * best_reps
            cut = True
        else:
            out.append(words[i])
            i += 1
    return " ".join(out), cut


# --------------------------------------------------------------------------
# VAD de energia (fallback), RMS com histerese
# --------------------------------------------------------------------------

def _frame_db(audio: np.ndarray, frame: int) -> np.ndarray:
    """dBFS por quadro (RMS), com piso em -100 dB."""
    n = len(audio) // frame
    if n == 0:
        return np.empty(0, dtype=np.float32)
    blocks = audio[: n * frame].reshape(n, frame).astype(np.float32, copy=False)
    rms = np.sqrt(np.mean(blocks * blocks, axis=1) + 1e-20)
    return 20.0 * np.log10(np.maximum(rms, 1e-5))


def energy_vad(audio: np.ndarray, samplerate: int = SR_ASR,
               frame_ms: int = 30, open_offset_db: float = 12.0,
               hysteresis_db: float = 8.0, min_open_db: float = -45.0,
               ) -> List[Tuple[int, int]]:
    """VAD por energia com histerese, calibrado pelo piso de ruído da janela.

    O limiar de abertura é ``piso_de_ruído + open_offset_db`` (nunca abaixo de
    ``min_open_db`` dBFS, tipicamente ~-45..-38 dBFS), limitado a 20 dB abaixo
    do nível alto da janela para funcionar também em fala contínua (sem
    silêncio na janela para estimar o piso). Fecha ``hysteresis_db`` abaixo.
    """
    frame = max(1, int(samplerate * frame_ms / 1000))
    db = _frame_db(audio, frame)
    if db.size == 0:
        return []
    noise = float(np.percentile(db, 10.0))
    loud = float(np.percentile(db, 95.0))
    open_db = max(min(noise + open_offset_db, loud - 20.0), min_open_db)
    close_db = open_db - hysteresis_db

    segs: List[Tuple[int, int]] = []
    active = False
    start = 0
    for i, val in enumerate(db):
        if not active and val >= open_db:
            active = True
            start = i
        elif active and val < close_db:
            active = False
            segs.append((start * frame, i * frame))
    if active:
        segs.append((start * frame, len(db) * frame))
    return segs


# --------------------------------------------------------------------------
# Silero VAD do faster-whisper: detecção da API disponível
# --------------------------------------------------------------------------

def _make_vad_options(vad_options_cls, threshold: float,
                      min_silence_ms: int) -> object:
    """Instancia VadOptions passando só os campos que a versão suporta."""
    desired = {
        "threshold": threshold,
        "neg_threshold": max(0.05, threshold - 0.15),
        "min_speech_duration_ms": 60,
        "min_silence_duration_ms": min_silence_ms,
        "speech_pad_ms": 0,
        "window_size_samples": 512,
    }
    try:
        names = {f.name for f in dataclasses.fields(vad_options_cls)}
    except TypeError:  # não é dataclass
        try:
            names = set(inspect.signature(vad_options_cls).parameters)
        except (TypeError, ValueError):
            names = set()
    kwargs = {k: v for k, v in desired.items() if k in names} if names else {}
    try:
        return vad_options_cls(**kwargs)
    except TypeError:
        return vad_options_cls()


def _build_silero_timestamps(samplerate: int, threshold: float,
                             min_silence_ms: int) -> Optional[VadFn]:
    """Tenta a API ``get_speech_timestamps`` + ``VadOptions``."""
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except Exception as exc:  # noqa: BLE001
        _LOG_VAD.debug("get_speech_timestamps indisponível: %s", exc)
        return None

    opts = _make_vad_options(VadOptions, threshold, min_silence_ms)
    try:
        params = set(inspect.signature(get_speech_timestamps).parameters)
    except (TypeError, ValueError):
        params = set()
    pass_sr = "sampling_rate" in params

    def _run(audio: np.ndarray) -> List[Tuple[int, int]]:
        buf = np.ascontiguousarray(audio, dtype=np.float32)
        kwargs = {"vad_options": opts}
        if pass_sr:
            kwargs["sampling_rate"] = samplerate
        try:
            raw = get_speech_timestamps(buf, **kwargs)
        except TypeError:
            raw = get_speech_timestamps(buf, opts)
        return _normalize_timestamps(raw, len(buf))

    if _smoke_test(_run, samplerate, "faster_whisper.vad.get_speech_timestamps"):
        return _run
    return None


def _build_silero_raw(samplerate: int, threshold: float) -> Optional[VadFn]:
    """Tenta a API de baixo nível (``get_vad_model`` / ``SileroVADModel``)."""
    model = None
    try:
        from faster_whisper.vad import get_vad_model  # type: ignore
        model = get_vad_model()
    except Exception as exc:  # noqa: BLE001
        _LOG_VAD.debug("get_vad_model indisponível: %s", exc)
    if model is None:
        try:
            import os

            from faster_whisper.utils import get_assets_path  # type: ignore
            from faster_whisper.vad import SileroVADModel  # type: ignore
            assets = get_assets_path()
            onnx = sorted(f for f in os.listdir(assets) if f.endswith(".onnx"))
            if not onnx:
                return None
            enc = [f for f in onnx if "encoder" in f]
            dec = [f for f in onnx if "decoder" in f]
            if enc and dec:   # versões com modelo dividido (v5)
                model = SileroVADModel(os.path.join(assets, enc[0]),
                                       os.path.join(assets, dec[0]))
            else:
                model = SileroVADModel(os.path.join(assets, onnx[0]))
        except Exception as exc:  # noqa: BLE001
            _LOG_VAD.debug("SileroVADModel indisponível: %s", exc)
            return None

    win = 512
    ctx_n = 64
    style = {"mode": None}   # descoberto na 1ª chamada e memorizado

    def _probs_whole(audio: np.ndarray) -> np.ndarray:
        """faster-whisper >= 1.2: model(audio_1d múltiplo de 512) -> probs."""
        n = len(audio) // win
        return np.asarray(model(audio[: n * win]), dtype=np.float32).reshape(-1)

    def _probs_chunked(audio: np.ndarray) -> np.ndarray:
        """faster-whisper 1.0/1.1: chamada por janela, com estado."""
        n = len(audio) // win
        chunks = audio[: n * win].reshape(n, win)
        state = (model.get_initial_state(batch_size=1)
                 if hasattr(model, "get_initial_state") else None)
        context = np.zeros((1, ctx_n), dtype=np.float32)
        out: List[float] = []
        for i in range(n):
            x = chunks[i:i + 1]
            res = None
            for args in ((x, state, context, samplerate), (x, state, samplerate),
                         (x, state), (x,)):
                try:
                    res = model(*args)
                    break
                except TypeError:
                    continue
            if res is None:
                raise RuntimeError("assinatura de SileroVADModel desconhecida")
            if isinstance(res, tuple):
                prob = res[0]
                if len(res) > 1:
                    state = res[1]
                if len(res) > 2:
                    context = res[2]
            else:
                prob = res
            out.append(float(np.asarray(prob).reshape(-1)[0]))
        return np.asarray(out, dtype=np.float32)

    def _probs(audio: np.ndarray) -> np.ndarray:
        n = len(audio) // win
        if n == 0:
            return np.empty(0, dtype=np.float32)
        if style["mode"] is None:
            for mode, fn in (("whole", _probs_whole), ("chunked", _probs_chunked)):
                try:
                    p = fn(audio)
                except Exception:  # noqa: BLE001
                    continue
                if p.size:
                    style["mode"] = mode
                    return p
            raise RuntimeError("nenhuma convenção de chamada do Silero funcionou")
        return (_probs_whole if style["mode"] == "whole" else _probs_chunked)(audio)

    def _run(audio: np.ndarray) -> List[Tuple[int, int]]:
        buf = np.ascontiguousarray(audio, dtype=np.float32)
        p = _probs(buf)
        if p.size == 0:
            return []
        segs: List[Tuple[int, int]] = []
        active = False
        start = 0
        neg = max(0.05, threshold - 0.15)
        for i, val in enumerate(p):
            if not active and val >= threshold:
                active, start = True, i
            elif active and val < neg:
                active = False
                segs.append((start * win, i * win))
        if active:
            segs.append((start * win, len(p) * win))
        return segs

    if _smoke_test(_run, samplerate, "faster_whisper.vad.SileroVADModel"):
        return _run
    return None


def _smoke_test(fn: VadFn, samplerate: int, name: str) -> bool:
    """Roda o detector em 1 s de silêncio para validar a API."""
    try:
        out = fn(np.zeros(samplerate, dtype=np.float32))
        list(out)
    except Exception as exc:  # noqa: BLE001
        _LOG_VAD.warning("VAD %s falhou na verificação inicial: %s", name, exc)
        return False
    _LOG_VAD.info("VAD Silero ativo via %s", name)
    return True


def _normalize_timestamps(raw: Sequence, n_samples: int) -> List[Tuple[int, int]]:
    """Normaliza a saída do VAD para [(ini, fim)] em amostras, ordenada."""
    out: List[Tuple[int, int]] = []
    for item in raw or []:
        if isinstance(item, dict):
            s, e = item.get("start", 0), item.get("end", 0)
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            s, e = item[0], item[1]
        else:
            s = getattr(item, "start", None)
            e = getattr(item, "end", None)
            if s is None or e is None:
                continue
        s, e = int(s), int(e)
        if e > s:
            out.append((max(0, s), min(n_samples, e)))
    out.sort()
    return out


def build_vad(samplerate: int = SR_ASR, threshold: float = 0.5,
              min_silence_ms: int = 80, use_silero: bool = True,
              ) -> Tuple[VadFn, str]:
    """Devolve ``(detector, nome_do_backend)``, preferindo o Silero.

    Ordem: ``get_speech_timestamps`` -> ``SileroVADModel`` -> energia RMS.
    """
    if use_silero:
        fn = _build_silero_timestamps(samplerate, threshold, min_silence_ms)
        if fn is not None:
            return fn, "silero:get_speech_timestamps"
        fn = _build_silero_raw(samplerate, threshold)
        if fn is not None:
            return fn, "silero:model"
        _LOG_VAD.warning(
            "Silero VAD do faster-whisper indisponível; usando VAD de energia "
            "RMS (qualidade inferior).")
    else:
        _LOG_VAD.info("VAD de energia RMS solicitado explicitamente.")

    def _run(audio: np.ndarray) -> List[Tuple[int, int]]:
        return energy_vad(audio, samplerate)

    return _run, "energy"


# --------------------------------------------------------------------------
# SpeechSegmenter
# --------------------------------------------------------------------------

class _Flush:
    """Sentinela de flush na fila interna."""

    def __init__(self) -> None:
        self.done = threading.Event()


_STOP = object()


class SpeechSegmenter:
    """Segmentador de fala streaming (implementa ``SpeechSegmenterProtocol``).

    ``feed()`` só enfileira (nunca bloqueia); uma thread própria roda o VAD
    sobre um buffer rolante a cada ``tick_ms`` de áudio novo. A fala abre
    quando o VAD detecta voz (com pré-roll de ``pad_ms``) e fecha após
    ``silence_ms`` contínuos sem voz. Segmentos com menos de ``min_speech_ms``
    são descartados; ao atingir ``max_segment_s`` o corte é feito no vale de
    menor energia dos últimos 800 ms.

    Os timestamps são derivados do índice absoluto da amostra a partir do
    ``time.monotonic()`` do primeiro bloco. Em tempo real coincidem com o
    relógio, e ``t_end - t_start`` é sempre a duração real do PCM.
    """

    def __init__(self, on_segment: Callable[[SpeechSegment], None],
                 samplerate: int = SR_ASR, silence_ms: int = 600,
                 max_segment_s: float = 12.0, min_speech_ms: int = 250,
                 pad_ms: int = 150, *, use_silero: bool = True,
                 vad_threshold: float = 0.5, tick_ms: int = 300,
                 lookback_s: float = 2.0, queue_blocks: int = 512) -> None:
        self.on_segment = on_segment
        self.samplerate = int(samplerate)
        self.silence_ms = int(silence_ms)
        self.max_segment_s = float(max_segment_s)
        self.min_speech_ms = int(min_speech_ms)
        self.pad_ms = int(pad_ms)

        sr = self.samplerate
        self._pad = int(sr * self.pad_ms / 1000)
        self._tick = max(1, int(sr * tick_ms / 1000))
        self._lookback = max(int(sr * lookback_s), self._tick + self._pad)
        self._min_speech = int(sr * self.min_speech_ms / 1000)
        self._silence = int(sr * self.silence_ms / 1000)
        self._max_seg = int(sr * self.max_segment_s)
        self._valley_win = int(sr * 0.8)
        self._min_vad = max(512, int(sr * 0.5))   # janela mínima p/ rodar o VAD

        self._vad, self.vad_backend = build_vad(
            sr, vad_threshold, use_silero=use_silero)

        self._queue: "queue.Queue[object]" = queue.Queue(maxsize=queue_blocks)
        self._buf = np.zeros(0, dtype=np.float32)
        self._buf_start = 0          # índice absoluto de _buf[0]
        self._total = 0              # total de amostras já recebidas
        self._processed = 0          # último _total processado pelo VAD
        self._t_origin: Optional[float] = None
        self._in_speech = False
        self._speech_start = 0     # início do PCM emitido (com pré-roll)
        self._voice_start = 0      # início da VOZ detectada (sem pré-roll)
        self._last_voice = 0
        self._ignore_before = 0
        self._dropped = 0

        self._running = True
        self._thread = threading.Thread(
            target=self._worker, name="SpeechSegmenter", daemon=True)
        self._thread.start()

    # -- API pública -------------------------------------------------------

    def feed(self, pcm_mono_16k: np.ndarray) -> None:
        """Enfileira um bloco mono float32 @ samplerate. Nunca bloqueia."""
        if pcm_mono_16k is None or len(pcm_mono_16k) == 0 or not self._running:
            return
        block = np.asarray(pcm_mono_16k, dtype=np.float32).reshape(-1)
        try:
            self._queue.put_nowait(block)
        except queue.Full:
            self._dropped += 1
            if self._dropped % 50 == 1:
                _LOG_VAD.warning("fila do segmentador cheia; descartando áudio "
                                 "(%d blocos)", self._dropped)
            try:  # descarta o mais antigo e mantém o novo
                self._queue.get_nowait()
                self._queue.put_nowait(block)
            except (queue.Empty, queue.Full):
                pass

    def pending_blocks(self) -> int:
        """Blocos ainda não processados (monitoramento de sobrecarga)."""
        return self._queue.qsize()

    def flush(self, timeout: float = 5.0) -> None:
        """Força a emissão do que estiver acumulado (usado ao pausar)."""
        if not self._running:
            return
        token = _Flush()
        try:
            self._queue.put_nowait(token)
        except queue.Full:
            return
        token.done.wait(timeout)

    def stop(self) -> None:
        """Encerra a thread de processamento."""
        if not self._running:
            return
        self._running = False
        try:
            self._queue.put_nowait(_STOP)
        except queue.Full:
            pass
        self._thread.join(timeout=3.0)

    # -- interno -----------------------------------------------------------

    def _worker(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                if not self._running:
                    return
                continue
            if item is _STOP:
                return
            if isinstance(item, _Flush):
                self._do_flush(item)
                continue

            blocks = [item]
            while True:  # drena o que já chegou (menos concatenações)
                try:
                    nxt = self._queue.get_nowait()
                except queue.Empty:
                    break
                if nxt is _STOP:
                    self._append(blocks)
                    return
                if isinstance(nxt, _Flush):
                    self._append(blocks)
                    blocks = []
                    self._pump()
                    self._do_flush(nxt)
                    continue
                blocks.append(nxt)
            self._append(blocks)
            self._pump()

    def _do_flush(self, token: _Flush) -> None:
        try:
            self._process(self._total, force=True)
            self._flush_pending()
        except Exception:  # noqa: BLE001
            _LOG_VAD.exception("erro no flush do segmentador")
        finally:
            token.done.set()

    def _pump(self) -> None:
        """Avança o VAD em passos de ``tick`` (robusto a rajadas na fila)."""
        while self._total - self._processed >= self._tick:
            try:
                self._process(self._processed + self._tick)
            except Exception:  # noqa: BLE001
                _LOG_VAD.exception("erro no processamento do VAD")
                self._processed = self._total

    def _append(self, blocks: List[np.ndarray]) -> None:
        if not blocks:
            return
        if self._t_origin is None:
            n_first = sum(len(b) for b in blocks)
            self._t_origin = time.monotonic() - n_first / self.samplerate
        chunk = blocks[0] if len(blocks) == 1 else np.concatenate(blocks)
        self._buf = np.concatenate((self._buf, chunk)) if self._buf.size else chunk
        self._total += len(chunk)

    def _t(self, idx: int) -> float:
        base = self._t_origin if self._t_origin is not None else time.monotonic()
        return base + idx / self.samplerate

    def _slice(self, start: int, end: int) -> np.ndarray:
        a = max(0, start - self._buf_start)
        b = max(a, min(len(self._buf), end - self._buf_start))
        return self._buf[a:b]

    def _process(self, now: int, force: bool = False) -> None:
        """Roda o VAD na janela que termina em ``now`` (índice absoluto).

        ``now`` é o "agora" virtual do stream: o processamento avança em
        passos de ``tick`` mesmo que a fila entregue uma rajada de áudio,
        preservando a semântica de tempo real.
        """
        now = min(now, self._total)
        if now - self._processed < self._tick and not force:
            return
        self._processed = now

        total = now
        if self._in_speech:
            win_start = max(self._buf_start, self._speech_start - self._pad)
        else:
            win_start = max(self._buf_start, self._ignore_before,
                            total - self._lookback)
        win = self._slice(win_start, total)
        if len(win) >= min(self._min_vad, 512):
            stamps = self._vad(win)
        else:
            stamps = []

        if stamps:
            first_abs = win_start + stamps[0][0]
            last_abs = win_start + stamps[-1][1]
            if not self._in_speech:
                self._in_speech = True
                self._voice_start = first_abs
                self._speech_start = max(first_abs - self._pad,
                                         self._ignore_before, self._buf_start)
                self._last_voice = last_abs
            else:
                self._last_voice = max(self._last_voice, last_abs)

        if self._in_speech:
            silence = total - self._last_voice
            if silence >= self._silence:
                end = min(self._last_voice + self._pad, total)
                # a duração útil é a de VOZ (sem o pad), p/ não passar blips
                self._emit(self._speech_start, end,
                           voiced=self._last_voice - self._voice_start)
                self._reset_speech(end)
            elif total - self._speech_start >= self._max_seg:
                cut = self._valley_cut(total)
                self._emit(self._speech_start, cut)
                self._speech_start = cut
                self._voice_start = cut
                self._last_voice = max(self._last_voice, cut)
                self._ignore_before = min(self._ignore_before, cut)
                _LOG_VAD.debug("corte por max_segment_s em %.2fs", self._t(cut))

        self._trim(total)

    def _valley_cut(self, total: int) -> int:
        """Índice absoluto do vale de menor energia nos últimos 800 ms."""
        start = max(self._speech_start + self._min_speech,
                    total - self._valley_win)
        if start >= total:
            return total
        win = self._slice(start, total)
        frame = max(1, int(self.samplerate * 0.02))
        db = _frame_db(win, frame)
        if db.size == 0:
            return total
        return start + int(np.argmin(db)) * frame

    def _emit(self, start: int, end: int, voiced: Optional[int] = None) -> None:
        start = max(start, self._buf_start)
        end = min(end, self._total)
        n = end - start
        useful = n if voiced is None else min(voiced, n)
        if useful < self._min_speech:
            _LOG_VAD.debug("segmento curto descartado (%.0f ms de voz)",
                           1000.0 * useful / self.samplerate)
            return
        pcm = np.array(self._slice(start, end), dtype=np.float32, copy=True)
        seg = SpeechSegment(pcm=pcm, t_start=self._t(start), t_end=self._t(end))
        _LOG_VAD.debug("segmento %.2fs (%s)", n / self.samplerate,
                       self.vad_backend)
        try:
            self.on_segment(seg)
        except Exception:  # noqa: BLE001
            _LOG_VAD.exception("on_segment lançou exceção")

    def _reset_speech(self, end: int) -> None:
        self._in_speech = False
        self._speech_start = 0
        self._voice_start = 0
        self._last_voice = 0
        self._ignore_before = max(self._ignore_before, end)

    def _flush_pending(self) -> None:
        if self._in_speech:
            end = min(self._last_voice + self._pad, self._total)
            self._emit(self._speech_start, end,
                       voiced=self._last_voice - self._voice_start)
            self._reset_speech(end)

    def _trim(self, now: int) -> None:
        if self._in_speech:
            keep = self._speech_start - self._pad
        else:
            keep = now - (self._lookback + self._pad)
        keep = max(self._buf_start, min(keep, self._total))
        if keep > self._buf_start:
            self._buf = np.array(self._buf[keep - self._buf_start:],
                                 dtype=np.float32, copy=True)
            self._buf_start = keep


# --------------------------------------------------------------------------
# Transcriber
# --------------------------------------------------------------------------

class Transcriber:
    """ASR síncrono com faster-whisper (implementa ``TranscriberProtocol``).

    O modelo é carregado no ``__init__`` (bloqueante, pode baixar do
    HuggingFace na primeira vez). ``transcribe()`` devolve ``None`` quando o
    segmento não tem fala útil (texto vazio, alta ``no_speech_prob`` ou
    alucinação clássica do Whisper em silêncio).
    """

    def __init__(self, model_size: str = "small", compute_type: str = "int8",
                 cpu_threads: int = 4, *, device: str = "cpu",
                 num_workers: int = 1, beam_size: int = 1,
                 no_speech_threshold: float = 0.6,
                 logprob_threshold: float = -1.0,
                 temperatures: Sequence[float] = (0.0, 0.2, 0.4),
                 compression_ratio_threshold: float = 2.4,
                 repetition_penalty: float = 1.1) -> None:
        from faster_whisper import WhisperModel  # import tardio (pesado)

        self.model_size = model_size
        self.beam_size = int(beam_size)
        self.no_speech_threshold = float(no_speech_threshold)
        self.logprob_threshold = float(logprob_threshold)
        # temperatura em escada: se o decoder entrar em loop (razão de
        # compressão alta) o faster-whisper redecodifica mais quente
        self.temperatures = [float(t) for t in temperatures]
        self.compression_ratio_threshold = float(compression_ratio_threshold)
        self.repetition_penalty = float(repetition_penalty)

        t0 = time.monotonic()
        self._model = WhisperModel(
            model_size, device=device, compute_type=compute_type,
            cpu_threads=cpu_threads, num_workers=num_workers)
        _LOG_ASR.info("modelo whisper '%s' carregado (%s/%s) em %.1fs",
                      model_size, device, compute_type, time.monotonic() - t0)

    def transcribe(self, segment: SpeechSegment) -> Optional[Transcript]:
        """Transcreve um segmento; ``None`` se não houver fala útil."""
        pcm = np.ascontiguousarray(
            np.asarray(segment.pcm, dtype=np.float32).reshape(-1))
        dur = len(pcm) / SR_ASR
        if len(pcm) == 0:
            return None

        t0 = time.monotonic()
        try:
            segments, info = self._model.transcribe(
                pcm,
                language=None,
                beam_size=self.beam_size,
                vad_filter=False,
                condition_on_previous_text=False,
                without_timestamps=True,
                temperature=self.temperatures,
                compression_ratio_threshold=self.compression_ratio_threshold,
                log_prob_threshold=self.logprob_threshold,
                repetition_penalty=self.repetition_penalty,
            )
            parts = list(segments)
        except Exception:  # noqa: BLE001
            _LOG_ASR.exception("falha na transcrição (%.2fs de áudio)", dur)
            return None
        elapsed = time.monotonic() - t0

        text = _WS_RE.sub(" ", " ".join(
            (getattr(p, "text", "") or "").strip() for p in parts)).strip()
        lang = getattr(info, "language", "") or ""
        lang_prob = float(getattr(info, "language_probability", 0.0) or 0.0)

        if not text:
            _LOG_ASR.info("descartado: vazio | %.2fs áudio | %.2fs proc", dur,
                          elapsed)
            return None

        # rede de segurança: a escada de temperatura reduz os loops, não os
        # elimina; o que passar é colapsado aqui antes de ir para MT/TTS
        collapsed, cut = collapse_repetitions(text)
        if cut:
            _LOG_ASR.info("loop de repetição colapsado (%d->%d chars) | %r",
                          len(text), len(collapsed), text[:120])
            text = collapsed

        no_speech = _mean_attr(parts, "no_speech_prob", 0.0)
        avg_logprob = _mean_attr(parts, "avg_logprob", 0.0)

        if no_speech > self.no_speech_threshold and avg_logprob < self.logprob_threshold:
            _LOG_ASR.info("descartado: no_speech=%.2f logprob=%.2f | %r",
                          no_speech, avg_logprob, text[:60])
            return None
        if is_hallucination(text):
            _LOG_ASR.info("descartado: alucinação | %r", text[:60])
            return None

        _LOG_ASR.info("asr lang=%s (%.2f) | áudio %.2fs | proc %.2fs "
                      "(rtf %.2f) | %s", lang, lang_prob, dur, elapsed,
                      elapsed / dur if dur else 0.0, text[:120])
        return Transcript(text=text, lang=lang, lang_prob=lang_prob,
                          segment=segment)


def _mean_attr(items: Sequence, name: str, default: float) -> float:
    vals = [float(getattr(it, name)) for it in items
            if getattr(it, name, None) is not None]
    return sum(vals) / len(vals) if vals else default
