"""Camada de áudio: captura WASAPI loopback (`soundcard`) e mixer de saída
(`sounddevice`).

Implementa os contratos ``LoopbackCaptureProtocol`` e ``OutputMixerProtocol``
de :mod:`tradutor.contracts`, além das funções utilitárias de listagem de
dispositivos.

Convenções (ver contracts.py):
    - PCM sempre ``numpy.ndarray`` float32 em [-1, 1].
    - Taxa nativa 48 kHz; blocos de 960 quadros (20 ms).
"""

from __future__ import annotations

import logging
import os
import threading
from collections import deque
from typing import Callable, Deque, List, Optional, Tuple

import numpy as np

try:  # pragma: no cover - depende do ambiente
    from .contracts import SR_NATIVE
except Exception:  # pragma: no cover - execução fora do pacote
    SR_NATIVE = 48000

try:  # pragma: no cover
    import soundcard as _sc
except Exception as _e:  # pragma: no cover
    _sc = None
    _SC_ERR = _e
else:  # pragma: no cover
    _SC_ERR = None

try:  # pragma: no cover
    import sounddevice as _sd
except Exception as _e:  # pragma: no cover
    _sd = None
    _SD_ERR = _e
else:  # pragma: no cover
    _SD_ERR = None

log = logging.getLogger("tradutor.audio")

BLOCKSIZE = 960                 # 20 ms @ 48 kHz
_RAMP_SECONDS = 0.05            # tempo para percorrer todo o range de ganho
_DUCK_HOLD_SECONDS = 0.25       # mantém o ducking um pouco após o fim do TTS
_PASSTHROUGH_MAX_SECONDS = 0.20  # teto do ring de passthrough (fica "ao vivo")
_PASSTHROUGH_PREFILL_SECONDS = 0.06  # colchão antes de começar a tocar (anti-jitter)
_TTS_EDGE_FADE_SECONDS = 0.003  # fade nas bordas de cada fala (anti-estalo)


# --------------------------------------------------------------------------
# COM (Windows): soundcard usa WASAPI via COM e exige inicialização por thread
# --------------------------------------------------------------------------

def _com_initialize() -> bool:
    """Inicializa COM (MTA) na thread atual. Silencioso fora do Windows."""
    if os.name != "nt":
        return False
    try:
        import ctypes
        # COINIT_MULTITHREADED = 0x0, mesmo modo usado internamente pelo
        # soundcard; S_FALSE/RPC_E_CHANGED_MODE são inofensivos.
        ctypes.windll.ole32.CoInitializeEx(None, 0x0)
        return True
    except Exception as exc:  # pragma: no cover
        log.debug("CoInitializeEx falhou (seguindo mesmo assim): %s", exc)
        return False


def _com_uninitialize(initialized: bool) -> None:
    if not initialized or os.name != "nt":
        return
    try:
        import ctypes
        ctypes.windll.ole32.CoUninitialize()
    except Exception:  # pragma: no cover
        pass


# --------------------------------------------------------------------------
# Utilitários de dispositivo
# --------------------------------------------------------------------------

def _loopback_mics() -> list:
    """Lista os objetos de microfone de loopback do soundcard."""
    if _sc is None:
        log.warning("soundcard indisponível: %s", _SC_ERR)
        return []
    try:
        mics = _sc.all_microphones(include_loopback=True)
    except Exception as exc:
        log.warning("falha ao listar microfones: %s", exc)
        return []
    loop = [m for m in mics if getattr(m, "isloopback", False)]
    return loop or mics


def default_loopback_name() -> Optional[str]:
    """Nome do dispositivo de saída padrão (fonte do loopback padrão)."""
    if _sc is None:
        return None
    try:
        spk = _sc.default_speaker()
        return spk.name if spk is not None else None
    except Exception as exc:
        log.warning("falha ao obter o speaker padrão: %s", exc)
        return None


def list_loopback_devices() -> List[str]:
    """Nomes dos dispositivos de loopback disponíveis (padrão do sistema primeiro)."""
    names: List[str] = []
    for mic in _loopback_mics():
        name = getattr(mic, "name", None)
        if name and name not in names:
            names.append(name)
    default = default_loopback_name()
    if default:
        for i, name in enumerate(names):
            if default.lower() in name.lower() or name.lower() in default.lower():
                names.insert(0, names.pop(i))
                break
    return names


def list_output_devices() -> List[Tuple[int, str]]:
    """Dispositivos de reprodução do sounddevice como (índice, nome).

    O PortAudio expõe cada endpoint várias vezes (MME, DirectSound, WASAPI,
    WDM-KS), e o MME ainda trunca o nome em 31 caracteres. Filtramos para as
    entradas WASAPI: uma por dispositivo, nome completo e abertura direta no
    endpoint certo. Se não houver WASAPI (outra plataforma), devolve tudo.
    """
    if _sd is None:
        log.warning("sounddevice indisponível: %s", _SD_ERR)
        return []
    try:
        devices = _sd.query_devices()
        hostapis = _sd.query_hostapis()
    except Exception as exc:
        log.warning("falha ao listar dispositivos de saída: %s", exc)
        return []
    wasapi_ids = {i for i, h in enumerate(hostapis)
                  if "wasapi" in str(h.get("name", "")).lower()}
    all_out: List[Tuple[int, str, int]] = []
    for idx, dev in enumerate(devices):
        if int(dev.get("max_output_channels", 0)) > 0:
            all_out.append((idx, str(dev.get("name", f"device {idx}")),
                            int(dev.get("hostapi", -1))))
    wasapi = [(i, n) for i, n, h in all_out if h in wasapi_ids]
    return wasapi or [(i, n) for i, n, _ in all_out]


def find_cable_device() -> Optional[str]:
    """Nome do loopback do VB-CABLE, se instalado."""
    for name in list_loopback_devices():
        if "cable" in name.lower():
            return name
    return None


# --------------------------------------------------------------------------
# Captura (WASAPI loopback)
# --------------------------------------------------------------------------

class LoopbackCapture:
    """Captura o áudio que o sistema está reproduzindo (WASAPI loopback).

    Roda em thread própria; para cada bloco chama
    ``on_block(pcm (n, canais) float32, samplerate)``. Se o dispositivo
    desaparecer, espera 0,5 s, redescobre e reabre, avisando ``on_error`` uma
    vez por falha.
    """

    def __init__(self, device_hint: Optional[str] = None,
                 samplerate: int = SR_NATIVE, blocksize: int = BLOCKSIZE) -> None:
        self.device_hint = device_hint
        self.samplerate = int(samplerate)
        self.blocksize = int(blocksize)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._on_block: Optional[Callable[[np.ndarray, int], None]] = None
        self._on_error: Optional[Callable[[Exception], None]] = None
        self.device_name: Optional[str] = None

    # -- descoberta ---------------------------------------------------------

    def _resolve_mic(self):
        """Devolve o objeto de microfone de loopback a ser aberto."""
        if _sc is None:
            raise RuntimeError(f"soundcard indisponível: {_SC_ERR}")
        hint = self.device_hint
        if not hint:
            name = default_loopback_name()
            if not name:
                raise RuntimeError("nenhum dispositivo de saída padrão encontrado")
            mic = _sc.get_microphone(id=str(name), include_loopback=True)
            self.device_name = getattr(mic, "name", name)
            return mic
        low = hint.lower()
        for mic in _loopback_mics():
            if low in str(getattr(mic, "name", "")).lower():
                self.device_name = mic.name
                return mic
        # último recurso: deixa o próprio soundcard resolver a substring
        mic = _sc.get_microphone(id=str(hint), include_loopback=True)
        self.device_name = getattr(mic, "name", hint)
        return mic

    # -- ciclo de vida ------------------------------------------------------

    def start(self, on_block: Callable[[np.ndarray, int], None],
              on_error: Optional[Callable[[Exception], None]] = None) -> None:
        """Inicia a thread de captura (idempotente)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._on_block = on_block
        self._on_error = on_error
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="loopback-capture",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Sinaliza parada e aguarda a thread encerrar."""
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self._thread = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- laço ---------------------------------------------------------------

    def _report(self, exc: Exception) -> None:
        log.warning("captura: %s", exc)
        cb = self._on_error
        if cb is not None:
            try:
                cb(exc)
            except Exception:  # pragma: no cover
                log.exception("on_error levantou exceção")

    def _run(self) -> None:
        com = _com_initialize()
        try:
            while not self._stop.is_set():
                try:
                    mic = self._resolve_mic()
                    log.info("captura aberta em %r @ %d Hz", self.device_name,
                             self.samplerate)
                    with mic.recorder(samplerate=self.samplerate,
                                      blocksize=self.blocksize) as rec:
                        while not self._stop.is_set():
                            data = rec.record(numframes=self.blocksize)
                            if data is None:
                                continue
                            pcm = np.asarray(data, dtype=np.float32)
                            if pcm.size == 0:
                                continue
                            if pcm.ndim == 1:
                                pcm = pcm.reshape(-1, 1)
                            if self._stop.is_set():
                                # `record()` pode ter demorado e esta captura já
                                # ter sido substituída: não entregar o bloco
                                break
                            cb = self._on_block
                            if cb is not None:
                                try:
                                    cb(pcm, self.samplerate)
                                except Exception:
                                    # erro do consumidor não derruba a captura
                                    log.exception("on_block levantou exceção")
                except Exception as exc:
                    if self._stop.is_set():
                        break
                    self._report(exc)
                    self._stop.wait(0.5)   # espera antes de redescobrir
        finally:
            _com_uninitialize(com)
            log.info("thread de captura encerrada")


# --------------------------------------------------------------------------
# Mixer de saída
# --------------------------------------------------------------------------

# -- AGC (normalização do áudio original) -----------------------------------
# Nivela o passthrough para um nível de referência ANTES do slider de volume,
# para que "Volume original" soe igual com o YouTube a 100% ou a 40%.
_AGC_TARGET_RMS = 0.08          # ~-22 dBFS de referência
_AGC_MIN_GAIN = 0.2             # nunca atenua mais que 5x
_AGC_MAX_GAIN = 6.0             # nunca amplifica mais que 6x (evita soprar ruído)
_AGC_SILENCE_RMS = 1e-4         # abaixo disso é silêncio: não adapta
_AGC_ATTACK_SECONDS = 0.25      # reação quando o som fica ALTO demais (rápida)
_AGC_RELEASE_SECONDS = 2.0      # reação quando o som fica baixo (lenta)


class OutputMixer:
    """Mixa o passthrough do áudio original com a fala TTS em pt-BR.

    O passthrough vive num ring com teto de ~200 ms (descarta o mais antigo
    para nunca acumular atraso); o TTS vive numa fila mono sem teto. Enquanto
    houver TTS pendente, o passthrough é atenuado até ``duck_level`` com rampa
    linear de ~50 ms (nunca degrau, que estala).
    """

    def __init__(self, output_device: Optional[object] = None,
                 samplerate: int = SR_NATIVE, channels: int = 2,
                 blocksize: int = BLOCKSIZE,
                 gain_original: float = 0.30, gain_tts: float = 1.00,
                 duck_level: float = 0.15) -> None:
        self.output_device = output_device
        self.samplerate = int(samplerate)
        self.channels = int(channels)
        self.blocksize = int(blocksize)

        self._lock = threading.Lock()
        self._gain_original = float(gain_original)
        self._gain_tts = float(gain_tts)
        self._duck_level = float(duck_level)

        self._pass: Deque[np.ndarray] = deque()
        self._pass_off = 0
        self._pass_remaining = 0
        self._pass_max = int(_PASSTHROUGH_MAX_SECONDS * self.samplerate)
        self._pass_prefill = int(_PASSTHROUGH_PREFILL_SECONDS * self.samplerate)
        self._pass_priming = True   # espera o colchão encher antes de tocar

        self._tts: Deque[np.ndarray] = deque()
        self._tts_off = 0
        self._tts_remaining = 0

        self._cur_gain = float(gain_original)   # estado da rampa de ducking
        self._agc_gain = 1.0                    # normalização do passthrough
        self.agc_enabled = False                # o app liga; testes ficam determinísticos
        # Segura o ducking por um tempo após a última amostra de TTS, para não
        # oscilar entre falas próximas. Contado em quadros (relógio de áudio),
        # não em tempo de parede, assim o render offline se comporta igual.
        self._duck_hold_frames = 0
        self._stream = None
        self._underruns = 0

    # -- propriedades thread-safe -------------------------------------------

    @property
    def gain_original(self) -> float:
        """Ganho do áudio original quando não há TTS tocando."""
        with self._lock:
            return self._gain_original

    @gain_original.setter
    def gain_original(self, value: float) -> None:
        with self._lock:
            self._gain_original = float(np.clip(value, 0.0, 1.5))

    @property
    def gain_tts(self) -> float:
        """Ganho aplicado à voz traduzida."""
        with self._lock:
            return self._gain_tts

    @gain_tts.setter
    def gain_tts(self, value: float) -> None:
        with self._lock:
            self._gain_tts = float(np.clip(value, 0.0, 1.5))

    @property
    def duck_level(self) -> float:
        """Ganho do original enquanto a tradução fala."""
        with self._lock:
            return self._duck_level

    @duck_level.setter
    def duck_level(self, value: float) -> None:
        with self._lock:
            self._duck_level = float(np.clip(value, 0.0, 1.5))

    # -- ciclo de vida ------------------------------------------------------

    def start(self) -> None:
        """Abre o OutputStream (idempotente)."""
        if self._stream is not None:
            return
        if _sd is None:
            raise RuntimeError(f"sounddevice indisponível: {_SD_ERR}")
        stream = _sd.OutputStream(
            device=self.output_device,
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.blocksize,
            latency="low",
            callback=self._callback,
        )
        stream.start()
        self._stream = stream
        log.info("mixer aberto: device=%r sr=%d ch=%d block=%d",
                 self.output_device, self.samplerate, self.channels, self.blocksize)

    def stop(self) -> None:
        """Fecha o stream e limpa os buffers."""
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            except Exception:  # pragma: no cover
                log.exception("erro ao parar o stream")
            try:
                stream.close()
            except Exception:  # pragma: no cover
                log.exception("erro ao fechar o stream")
        with self._lock:
            self._pass.clear()
            self._pass_off = self._pass_remaining = 0
            self._pass_priming = True
            self._tts.clear()
            self._tts_off = self._tts_remaining = 0
        log.info("mixer fechado (underruns=%d)", self._underruns)

    def __enter__(self) -> "OutputMixer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- entrada de áudio ---------------------------------------------------

    def _to_channels(self, pcm: np.ndarray) -> np.ndarray:
        """Normaliza (n,), (n,1) ou (n,k) para (n, self.channels) float32."""
        arr = np.asarray(pcm, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim != 2:
            raise ValueError(f"PCM com dimensões inesperadas: {arr.shape}")
        ch = arr.shape[1]
        if ch == self.channels:
            return np.ascontiguousarray(arr)
        if ch == 1:
            return np.repeat(arr, self.channels, axis=1)
        mono = arr.mean(axis=1, dtype=np.float32).reshape(-1, 1)
        return np.repeat(mono, self.channels, axis=1)

    def feed_passthrough(self, pcm: np.ndarray) -> None:
        """Enfileira um bloco do áudio original (já na taxa do mixer)."""
        block = self._to_channels(pcm)
        if block.shape[0] == 0:
            return
        with self._lock:
            self._pass.append(block)
            self._pass_remaining += block.shape[0]
            # teto: descarta o mais antigo para o passthrough ficar ao vivo
            dropped = 0
            while self._pass_remaining > self._pass_max and self._pass:
                head = self._pass[0]
                avail = head.shape[0] - self._pass_off
                excess = self._pass_remaining - self._pass_max
                if avail <= excess:
                    self._pass.popleft()
                    self._pass_off = 0
                    self._pass_remaining -= avail
                    dropped += avail
                else:
                    self._pass_off += excess
                    self._pass_remaining -= excess
                    dropped += excess
        if dropped:
            log.debug("passthrough: %d quadros antigos descartados", dropped)

    def enqueue_tts(self, pcm: np.ndarray) -> None:
        """Enfileira uma fala traduzida (mono float32, taxa do mixer)."""
        arr = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return
        arr = arr.copy()
        # fade curtíssimo nas bordas: evita estalo no início/fim da fala
        n = min(int(_TTS_EDGE_FADE_SECONDS * self.samplerate), arr.size // 2)
        if n > 1:
            ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
            arr[:n] *= ramp
            arr[-n:] *= ramp[::-1]
        with self._lock:
            self._tts.append(arr)
            self._tts_remaining += arr.size

    # -- estado -------------------------------------------------------------

    def tts_backlog_seconds(self) -> float:
        """Segundos de fala traduzida ainda não reproduzidos."""
        with self._lock:
            return self._tts_remaining / float(self.samplerate)

    def is_tts_active(self) -> bool:
        """True se há TTS tocando ou pendente."""
        with self._lock:
            return self._tts_remaining > 0

    def clear_tts(self) -> None:
        """Descarta a fila de TTS ('ir para o ao vivo') com fade de saída."""
        with self._lock:
            tail = None
            if self._tts:
                head = self._tts[0][self._tts_off:]
                n = min(head.size, int(0.010 * self.samplerate))
                if n > 1:
                    tail = head[:n] * np.linspace(1.0, 0.0, n, dtype=np.float32)
            self._tts.clear()
            self._tts_off = 0
            self._tts_remaining = 0
            if tail is not None:
                self._tts.append(np.ascontiguousarray(tail, dtype=np.float32))
                self._tts_remaining = tail.size

    @property
    def underruns(self) -> int:
        """Quantos blocos faltaram passthrough (diagnóstico)."""
        return self._underruns

    def passthrough_seconds(self) -> float:
        """Segundos de áudio original em buffer (diagnóstico/latência)."""
        with self._lock:
            return self._pass_remaining / float(self.samplerate)

    # -- render -------------------------------------------------------------

    def _pull_passthrough(self, frames: int) -> Tuple[np.ndarray, bool]:
        """Retira `frames` quadros do ring de passthrough (zeros se faltar).

        Enquanto estiver "primando" (logo após abrir ou após um underrun),
        devolve silêncio até acumular ~60 ms. Sem esse colchão o ring fica
        rente ao vazio e o jitter do produtor vira falha a cada bloco.
        """
        out = np.zeros((frames, self.channels), dtype=np.float32)
        if self._pass_priming:
            if self._pass_remaining < max(self._pass_prefill, frames):
                return out, False
            self._pass_priming = False
        filled = 0
        while filled < frames and self._pass:
            head = self._pass[0]
            avail = head.shape[0] - self._pass_off
            take = min(avail, frames - filled)
            out[filled:filled + take] = head[self._pass_off:self._pass_off + take]
            filled += take
            self._pass_off += take
            self._pass_remaining -= take
            if self._pass_off >= head.shape[0]:
                self._pass.popleft()
                self._pass_off = 0
        if filled < frames:
            self._pass_priming = True   # esvaziou: reconstrói o colchão
            return out, True
        return out, False

    def _pull_tts(self, frames: int) -> Tuple[np.ndarray, int]:
        """Retira `frames` quadros mono da fila de TTS (zeros se faltar)."""
        out = np.zeros(frames, dtype=np.float32)
        filled = 0
        while filled < frames and self._tts:
            head = self._tts[0]
            avail = head.size - self._tts_off
            take = min(avail, frames - filled)
            out[filled:filled + take] = head[self._tts_off:self._tts_off + take]
            filled += take
            self._tts_off += take
            self._tts_remaining -= take
            if self._tts_off >= head.size:
                self._tts.popleft()
                self._tts_off = 0
        return out, filled

    def render_block(self, frames: int) -> np.ndarray:
        """Renderiza `frames` quadros (n, canais), usado pelo callback e por testes."""
        with self._lock:
            g_orig = self._gain_original
            g_tts = self._gain_tts
            duck = self._duck_level
            passthrough, underrun = self._pull_passthrough(frames)
            tts, tts_filled = self._pull_tts(frames)
            if tts_filled > 0:
                self._duck_hold_frames = int(_DUCK_HOLD_SECONDS * self.samplerate)
            else:
                self._duck_hold_frames = max(0, self._duck_hold_frames - frames)
            target = duck if self._duck_hold_frames > 0 else g_orig

            # rampa linear limitada: o range completo leva _RAMP_SECONDS
            cur = self._cur_gain
            max_step = frames / (_RAMP_SECONDS * self.samplerate)
            delta = float(np.clip(target - cur, -max_step, max_step))
            nxt = cur + delta
            self._cur_gain = nxt

        if underrun:
            self._underruns += 1

        # AGC: nivela o original para _AGC_TARGET_RMS antes do slider,
        # tornando "Volume original" independente do volume da fonte.
        if self.agc_enabled and frames > 0:
            rms = float(np.sqrt(np.mean(np.square(passthrough))))
            if rms > _AGC_SILENCE_RMS:
                desired = float(np.clip(_AGC_TARGET_RMS / rms,
                                        _AGC_MIN_GAIN, _AGC_MAX_GAIN))
                tc = (_AGC_ATTACK_SECONDS if desired < self._agc_gain
                      else _AGC_RELEASE_SECONDS)
                alpha = min(1.0, frames / (tc * self.samplerate))
                self._agc_gain += (desired - self._agc_gain) * alpha

        env = np.linspace(cur, nxt, frames, endpoint=False, dtype=np.float32)
        out = passthrough * (env * self._agc_gain)[:, None]
        if tts_filled > 0:
            out += (tts * g_tts)[:, None]
        np.clip(out, -1.0, 1.0, out=out)
        return out

    def _callback(self, outdata, frames, time_info, status) -> None:
        """Callback do sounddevice: nunca levanta exceção (silêncio no pior caso)."""
        try:
            if status:
                log.debug("status do stream: %s", status)
            outdata[:] = self.render_block(frames)
        except Exception:  # pragma: no cover
            log.exception("erro no callback do mixer")
            try:
                outdata[:] = 0
            except Exception:
                pass


__all__ = [
    "LoopbackCapture",
    "OutputMixer",
    "list_loopback_devices",
    "list_output_devices",
    "find_cable_device",
    "default_loopback_name",
    "SR_NATIVE",
    "BLOCKSIZE",
]
