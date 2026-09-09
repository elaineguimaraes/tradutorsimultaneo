"""Contratos entre os módulos do Tradutor Simultâneo.

Este arquivo é a fonte de verdade da arquitetura. Cada módulo implementa as
interfaces daqui e NADA além delas é usado pela integração (main.py).

Fluxo de dados (threads separadas, conectadas por queue.Queue):

    LoopbackCapture (audio_io)  --bloco 48k float32-->  main
        main: downmix mono + resample 16k  -->  SpeechSegmenter.feed()
    SpeechSegmenter (segmenter)  --SpeechSegment-->  fila ASR
    Transcriber (segmenter)      --Transcript-->     fila MT
    Translator (translate_tts)   --Translation-->    fila TTS + legenda (GUI)
    TtsSpeaker (translate_tts)   --TtsAudio-->       OutputMixer.enqueue_tts()
    OutputMixer (audio_io): passthrough do original (com ducking) + voz pt

Convenções de áudio:
    - Todo PCM trafega como numpy.ndarray float32 em [-1, 1].
    - Captura/saída: SR_NATIVE = 48000 Hz. ASR/VAD: SR_ASR = 16000 Hz mono.
    - TtsSpeaker retorna o PCM na taxa nativa dele (campo samplerate);
      a integração resampleia para SR_NATIVE antes do mixer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

import numpy as np

SR_NATIVE = 48000   # captura e mixer de saída
SR_ASR = 16000      # VAD e Whisper (mono)


# --------------------------------------------------------------------------
# Mensagens que trafegam nas filas
# --------------------------------------------------------------------------

@dataclass
class SpeechSegment:
    """Trecho contínuo de fala detectado pelo VAD, pronto para o Whisper."""
    pcm: np.ndarray          # float32 mono @ SR_ASR
    t_start: float           # time.monotonic() do início da fala
    t_end: float             # time.monotonic() do fim da fala


@dataclass
class Transcript:
    text: str                # texto no idioma original
    lang: str                # código ISO detectado pelo Whisper ("en", "es", "pt", ...)
    lang_prob: float         # confiança da detecção de idioma (0-1)
    segment: SpeechSegment   # segmento de origem (para timestamps)


@dataclass
class Translation:
    text_pt: str             # texto traduzido para pt-BR
    source: Transcript       # transcrição de origem


@dataclass
class TtsAudio:
    pcm: np.ndarray          # float32 mono na taxa `samplerate`
    samplerate: int
    source: Translation


# --------------------------------------------------------------------------
# audio_io.py (especialista de áudio)
# --------------------------------------------------------------------------

class LoopbackCaptureProtocol(Protocol):
    """Captura o áudio de saída do sistema (WASAPI loopback) via `soundcard`.

    Implementação: classe LoopbackCapture(device_hint: str | None = None,
    samplerate: int = SR_NATIVE, blocksize: int = 960).
    - device_hint: substring do nome do dispositivo de loopback a capturar
      (ex.: "CABLE"). None => dispositivo de saída padrão do sistema.
    - Roda em thread própria; para cada bloco capturado chama
      on_block(pcm: np.ndarray float32 shape (n, canais), samplerate: int).
    - Deve ser resiliente: se o dispositivo sumir, tenta reabrir e reporta
      via on_error(exc) sem derrubar a thread.
    """

    def start(self, on_block: Callable[[np.ndarray, int], None],
              on_error: Optional[Callable[[Exception], None]] = None) -> None: ...
    def stop(self) -> None: ...


class OutputMixerProtocol(Protocol):
    """Mixa passthrough do áudio original + fala TTS num OutputStream (sounddevice).

    Implementação: classe OutputMixer(output_device: int | str | None = None,
    samplerate: int = SR_NATIVE, channels: int = 2).

    Ducking: enquanto houver TTS tocando, o passthrough é multiplicado por
    `duck_level` (com rampa suave de ~50 ms para não estalar); fora isso,
    por `gain_original`. O TTS toca com `gain_tts`.

    Atributos ajustáveis a quente (thread-safe, floats 0.0-1.5):
        gain_original, gain_tts, duck_level
    """

    gain_original: float
    gain_tts: float
    duck_level: float

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def feed_passthrough(self, pcm: np.ndarray) -> None:
        """Recebe bloco float32 (n, canais) @ samplerate do mixer (sem resample)."""
    def enqueue_tts(self, pcm: np.ndarray) -> None:
        """Enfileira fala pt-BR: float32 mono @ samplerate do mixer."""
    def tts_backlog_seconds(self) -> float:
        """Segundos de TTS ainda não reproduzidos (para taxa adaptativa)."""
    def is_tts_active(self) -> bool: ...
    def clear_tts(self) -> None:
        """Descarta toda a fila de TTS (botão 'ir para o ao vivo')."""


# Funções utilitárias esperadas em audio_io.py:
#   list_loopback_devices() -> list[str]
#   list_output_devices() -> list[tuple[int, str]]   (índice sounddevice, nome)
#   find_cable_device() -> str | None   (nome do loopback do VB-CABLE, se houver)


# --------------------------------------------------------------------------
# segmenter.py (especialista VAD + ASR)
# --------------------------------------------------------------------------

class SpeechSegmenterProtocol(Protocol):
    """Segmentador de fala streaming baseado em Silero VAD.

    Implementação: classe SpeechSegmenter(on_segment: Callable[[SpeechSegment], None],
    samplerate: int = SR_ASR, silence_ms: int = 600, max_segment_s: float = 12.0,
    min_speech_ms: int = 250, pad_ms: int = 150).

    - feed() recebe blocos mono float32 @ SR_ASR de qualquer tamanho, em
      qualquer thread; o processamento pesado deve ocorrer em thread própria.
    - Emite SpeechSegment quando: (a) detectou fim de fala (silence_ms de
      silêncio após fala) ou (b) a fala corrente atingiu max_segment_s
      (corta no melhor vale de energia recente para não partir palavra).
    - Usa o Silero VAD embutido no pacote faster_whisper (ver
      scratch/env_report.md para a API exata da versão instalada); fallback
      para VAD de energia RMS com histerese se o import falhar.
    """

    def feed(self, pcm_mono_16k: np.ndarray) -> None: ...
    def flush(self) -> None:
        """Força emissão do que estiver acumulado (usado ao pausar)."""
    def stop(self) -> None: ...


class TranscriberProtocol(Protocol):
    """ASR com faster-whisper. SÍNCRONO (a integração chama em thread própria).

    Implementação: classe Transcriber(model_size: str = "small",
    compute_type: str = "int8", cpu_threads: int = 4).
    - transcribe() retorna None se o segmento não contém fala útil
      (texto vazio, só música, alucinação típica de Whisper: filtrar
      no_speech_prob alto e textos-alucinação conhecidos tipo "Legendas pela
      comunidade Amara.org", "Thank you for watching", etc.).
    - Detecção de idioma automática; preencher Transcript.lang/lang_prob.
    - Parâmetros importantes: beam_size=1 ou 2 (CPU!), vad_filter=False
      (já segmentamos antes), condition_on_previous_text=False,
      without_timestamps=True.
    """

    def transcribe(self, segment: SpeechSegment) -> Optional[Transcript]: ...


# --------------------------------------------------------------------------
# translate_tts.py (especialista tradução + voz)
# --------------------------------------------------------------------------

class TranslatorProtocol(Protocol):
    """Tradução de texto para pt-BR, offline (Opus-MT via CTranslate2; Argos Translate como reserva).

    Implementação: classe Translator(target: str = "pt").
    - ensure_ready(langs: list[str]) baixa/instala pacotes na primeira vez
      (chamada na inicialização, pode demorar; ver env_report.md para o
      backend que funcionou no Python 3.13).
    - translate() SÍNCRONO. Se src_lang == "pt", retorna o texto original.
      Se não houver pacote para src_lang, retorna o texto original (a GUI
      mostra o cru, nunca lançar exceção por idioma não suportado).
    """

    def ensure_ready(self, langs: list[str]) -> None: ...
    def translate(self, text: str, src_lang: str) -> str: ...


class TtsSpeakerProtocol(Protocol):
    """Síntese de voz pt-BR via edge-tts (nuvem, gratuito).

    Implementação: classe TtsSpeaker(voice: str = "pt-BR-FranciscaNeural").
    - synth() SÍNCRONO (a integração chama em thread própria); internamente
      gerencia seu próprio loop asyncio (thread dedicada) e decodifica o mp3
      retornado para float32 mono (ver env_report.md: soundfile ou miniaudio).
    - rate_pct: aceleração da fala em % (0 = normal, 25 = +25%). Passar ao
      edge-tts como rate="+25%".
    - Em erro de rede: tentar 2x com backoff curto; se falhar, retornar None
      (a integração loga e segue, pois a legenda já foi mostrada).
    """

    def synth(self, text: str, rate_pct: int = 0) -> Optional[TtsAudio]: ...


# --------------------------------------------------------------------------
# gui.py (especialista de interface; Tkinter, roda na MAIN thread)
# --------------------------------------------------------------------------

@dataclass
class UiState:
    """Snapshot periódico do pipeline -> GUI (a GUI faz polling via after())."""
    running: bool = False
    detected_lang: str = ""          # último idioma detectado, ex. "en"
    backlog_seconds: float = 0.0     # atraso da fila de TTS
    rate_pct: int = 0                # aceleração atual do TTS
    status: str = ""                 # mensagem livre ("ouvindo…", "erro: …")


class ControllerProtocol(Protocol):
    """O que a GUI pode pedir ao pipeline (implementado em main.py).

    Todos os métodos devem ser thread-safe e retornar imediatamente.
    """

    def start_pipeline(self) -> None: ...
    def stop_pipeline(self) -> None: ...
    def set_gain_original(self, value: float) -> None: ...   # 0.0-1.0
    def set_gain_tts(self, value: float) -> None: ...        # 0.0-1.5
    def set_duck_level(self, value: float) -> None: ...      # 0.0-1.0
    def set_tts_speed(self, value: float) -> None: ...      # 1.0/1.25/1.5
    def set_tts_voice(self, name: str) -> None: ...         # id edge-tts (pt-BR-*Neural)
    def set_capture_device(self, name: str) -> None: ...
    def set_output_device(self, index: int) -> None: ...
    def skip_to_live(self) -> None: ...                      # limpa fila TTS
    def get_ui_state(self) -> UiState: ...
    def list_capture_devices(self) -> list[str]: ...
    def list_output_devices(self) -> list[tuple[int, str]]: ...


class GuiProtocol(Protocol):
    """Implementação: classe App(controller: ControllerProtocol, config: AppConfig).

    Componentes:
    1. Painel de controle (janela normal): botão Iniciar/Pausar, botão
       "Ao vivo" (skip_to_live), sliders (volume original, volume tradução,
       nível de ducking), combos de dispositivo (captura/saída), status
       (idioma detectado, atraso em s, taxa TTS), fonte da legenda +/-.
    2. Overlay de legenda: janela sem borda (overrideredirect), sempre no
       topo (-topmost), fundo escuro semi-transparente (-alpha ~0.85),
       arrastável com o mouse, botão/atalho para "clique atravessa"
       (WS_EX_TRANSPARENT via ctypes/win32), mostra as 2 últimas traduções
       (a atual em fonte maior, a anterior menor e acinzentada) e, em fonte
       pequena acima, o texto original. Some sozinho após ~6 s sem texto
       novo (alpha 0), reaparece ao chegar texto.
    - push_subtitle() pode ser chamado de QUALQUER thread: deve apenas
      enfileirar; o consumo acontece no loop Tk via after(100).
    - run() bloqueia (mainloop). Ao fechar a janela de controle: chama
      controller.stop_pipeline() e salva a config (posição do overlay,
      volumes, fonte) via config.save().
    """

    def push_subtitle(self, original: str, translated: str) -> None: ...
    def run(self) -> None: ...


# --------------------------------------------------------------------------
# config.py (escrito pela integração, disponível para todos)
# --------------------------------------------------------------------------

@dataclass
class AppConfig:
    """Persistida em config.json na raiz do projeto."""
    capture_device_hint: Optional[str] = None   # None => saída padrão; "CABLE" p/ VB-Cable
    output_device: Optional[int] = None         # None => padrão do sounddevice
    gain_original: float = 0.30
    gain_tts: float = 1.00
    duck_level: float = 0.15
    whisper_model: str = "small"
    tts_voice: str = "pt-BR-FranciscaNeural"
    silence_ms: int = 600
    max_segment_s: float = 12.0
    # taxa adaptativa: acima de cada limiar de backlog (s), usar a taxa (%)
    rate_ladder: list[tuple[float, int]] = field(
        default_factory=lambda: [(4.0, 10), (8.0, 25), (14.0, 40)])
    max_backlog_s: float = 20.0                 # acima disso, descarta áudio antigo
    subtitle_font_size: int = 18
    subtitle_pos: Optional[tuple[int, int]] = None
    overlay_click_through: bool = False

    def save(self, path: str = "config.json") -> None: ...
    @staticmethod
    def load(path: str = "config.json") -> "AppConfig": ...
