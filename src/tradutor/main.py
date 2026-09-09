"""Orquestração do Tradutor Simultâneo.

Liga captura -> VAD -> ASR -> tradução -> TTS -> mixer + GUI, conforme
contracts.py. Executar:  python -m tradutor.main  (com src/ no PYTHONPATH)
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Optional

import numpy as np

from tradutor import dsp
from tradutor.config import AppConfig
from tradutor.contracts import (SR_ASR, SR_NATIVE, SpeechSegment, Transcript,
                                Translation, UiState)

log = logging.getLogger("tradutor.main")


class Pipeline:
    """Dono das threads e filas; implementa ControllerProtocol para a GUI."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._lock = threading.Lock()
        self._running = False
        self._starting = False   # carregamento em andamento (impede start duplo)
        self._stopping = False   # parada em andamento (roda fora da thread da GUI)
        # cada start ganha uma geração; workers de gerações antigas morrem
        # sozinhos (antes, um worker preso num `put` sobrevivia ao stop e
        # voltava a rodar em paralelo com o novo, causando voz duplicada)
        self._generation = 0
        self._teardown: Optional[threading.Thread] = None
        self._state = UiState()
        self._gui = None  # injetado por attach_gui()

        # filas entre estágios (maxsize evita crescimento sem limite)
        self._q_asr: queue.Queue = queue.Queue(maxsize=8)
        self._q_mt: queue.Queue = queue.Queue(maxsize=16)
        self._q_tts: queue.Queue = queue.Queue(maxsize=16)

        # componentes pesados são criados sob demanda em _ensure_components()
        self._capture = None
        self._mixer = None
        self._segmenter = None
        self._transcriber = None
        self._translator = None
        self._speaker = None
        self._glossary = None
        self._tts_pool: Optional[ThreadPoolExecutor] = None
        self._threads: list[threading.Thread] = []
        self._components_ready = False
        self._capture_lock = threading.Lock()   # serializa trocas de captura
        self._last_block_ts = 0.0               # relógio do watchdog de captura

    # ------------------------------------------------------------------ GUI
    def attach_gui(self, gui) -> None:
        self._gui = gui

    # ------------------------------------------------- criação dos componentes
    def _ensure_components(self) -> None:
        if self._components_ready:
            return
        self._set_status("carregando modelos… (primeira vez demora)")
        from tradutor.audio_io import LoopbackCapture, OutputMixer
        from tradutor.segmenter import SpeechSegmenter, Transcriber
        from tradutor.translate_tts import Translator, TtsSpeaker

        cfg = self.config
        # Se o VB-CABLE está instalado e nenhum dispositivo foi escolhido,
        # capturar dele por padrão (setup recomendado: apps -> CABLE; o app
        # devolve o som nos alto-falantes reais). Evita realimentação do TTS.
        if cfg.capture_device_hint is None:
            from tradutor.audio_io import find_cable_device
            try:
                cable = find_cable_device()
            except Exception:
                cable = None
            if cable:
                cfg.capture_device_hint = cable
                log.info("VB-CABLE detectado, capturando de %r", cable)

        self._mixer = OutputMixer(output_device=self._resolve_output_device(),
                                  samplerate=SR_NATIVE)
        self._mixer.agc_enabled = True   # "Volume original" independe do volume da fonte
        self._mixer.gain_original = cfg.gain_original
        self._mixer.gain_tts = cfg.gain_tts
        self._apply_duck()

        self._segmenter = SpeechSegmenter(
            on_segment=self._on_segment, samplerate=SR_ASR,
            silence_ms=cfg.silence_ms, max_segment_s=cfg.max_segment_s)

        self._transcriber = Transcriber(model_size=cfg.whisper_model,
                                        compute_type="int8", cpu_threads=8)
        self._translator = Translator(target="pt")
        self._translator.ensure_ready(["en"])
        from tradutor.glossary import Glossary, set_quality_log, theme_path
        set_quality_log(getattr(cfg, "gravar_log", True))
        self._glossary = Glossary(theme_path(cfg.glossario))
        self._speaker = TtsSpeaker(voice=cfg.tts_voice)
        self._capture = LoopbackCapture(device_hint=cfg.capture_device_hint,
                                        samplerate=SR_NATIVE)
        # o argos religa os próprios loggers ao carregar o pacote
        for noisy in ("argostranslate", "argostranslate.utils", "stanza"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        self._components_ready = True

    # --------------------------------------------------------- fluxo de áudio
    def _on_capture_block(self, pcm: np.ndarray, samplerate: int) -> None:
        """Callback da captura (thread da captura): passthrough + ASR feed."""
        if not self._running:
            return
        self._last_block_ts = time.monotonic()   # sinal de vida p/ o watchdog
        self._mixer.feed_passthrough(pcm)
        mono = dsp.to_mono(pcm)
        if samplerate != SR_ASR:
            mono = dsp.resample(mono, samplerate, SR_ASR)
        self._segmenter.feed(mono)

    def _on_capture_error(self, exc: Exception) -> None:
        self._set_status(f"erro de captura: {exc}")

    def _on_segment(self, seg: SpeechSegment) -> None:
        """Callback do segmentador: enfileira para ASR (nunca bloqueia o VAD)."""
        if not self._drop_oldest_put(self._q_asr, seg, "ASR"):
            log.warning("fila ASR cheia, segmento descartado (%.1fs)",
                        seg.t_end - seg.t_start)

    # ----------------------------------------------------------------- filas
    @staticmethod
    def _drop_oldest_put(q: queue.Queue, item, label: str) -> bool:
        """Enfileira sem bloquear; se estiver cheia, joga fora o item MAIS ANTIGO.

        Numa tradução ao vivo o áudio velho não interessa, e um `put`
        bloqueante era a causa raiz do travamento geral: com a rede ruim o TTS
        atrasava, a fila enchia e as threads de ASR/tradução ficavam presas
        para sempre num `put`, sem nem responder ao `Pausar`.
        """
        for _ in range(q.maxsize + 1 if q.maxsize else 1):
            try:
                q.put_nowait(item)
                return True
            except queue.Full:
                try:
                    stale = q.get_nowait()
                except queue.Empty:
                    continue
                log.warning("fila %s cheia, descartando o item mais antigo", label)
                cancel = getattr(stale, "cancel", None)   # Futures de TTS
                if cancel is not None:
                    try:
                        cancel()
                    except Exception:
                        pass
        return False

    def _alive(self, gen: int) -> bool:
        """True enquanto este worker pertencer ao pipeline em execução."""
        return self._running and gen == self._generation

    # ------------------------------------------------------- threads de estágio
    def _asr_worker(self, gen: int) -> None:
        while self._alive(gen):
            try:
                seg = self._q_asr.get(timeout=0.3)
            except queue.Empty:
                continue
            t0 = time.monotonic()
            tr: Optional[Transcript] = None
            try:
                tr = self._transcriber.transcribe(seg)
            except Exception:
                log.exception("falha no ASR")
            if tr is None:
                continue
            with self._lock:
                self._state.detected_lang = tr.lang.upper()
            log.info("ASR %.1fs de áudio em %.1fs [%s] %r",
                     seg.t_end - seg.t_start, time.monotonic() - t0,
                     tr.lang, tr.text[:80])
            self._drop_oldest_put(self._q_mt, tr, "MT")

    _FILLERS = {"uh", "um", "hmm", "mm", "mm-hmm", "uh-huh", "ah", "oh",
                "hm", "huh", "yeah", "okay", "ok"}

    def _mt_worker(self, gen: int) -> None:
        while self._alive(gen):
            try:
                tr = self._q_mt.get(timeout=0.3)
            except queue.Empty:
                continue
            # muletas de fala ("uh", "um"…) não merecem legenda nem voz
            if tr.text.strip(" .,!?").lower() in self._FILLERS:
                continue
            try:
                text_pt = self._glossary.apply(
                    lambda t: self._translator.translate(t, tr.lang), tr.text)
            except Exception:
                log.exception("falha na tradução")
                text_pt = tr.text
            trans = Translation(text_pt=text_pt, source=tr)
            if self._gui is not None and text_pt.strip():
                self._gui.push_subtitle(tr.text, text_pt)
            # fala pt->pt não precisa de TTS (já é audível no original)
            if tr.lang != "pt":
                # síntese em paralelo (2 por vez); a fila guarda Futures em
                # ordem de fala e o tts_worker consome nessa mesma ordem:
                # a frase N+1 sintetiza ENQUANTO a N ainda está tocando.
                pool = self._tts_pool
                if pool is None:      # parada concorrente descartou o pool
                    continue
                try:
                    fut = pool.submit(self._synth_one, trans)
                except RuntimeError:  # pool já encerrado
                    continue
                self._drop_oldest_put(self._q_tts, fut, "TTS")

    def _synth_one(self, trans: Translation):
        """Roda no pool: sintetiza uma frase; devolve (audio, rate) ou None."""
        rate = self._adaptive_rate()
        try:
            audio = self._speaker.synth(trans.text_pt, rate_pct=rate,
                                        source=trans)
        except Exception:
            log.exception("falha no TTS")
            return None
        return None if audio is None else (audio, rate)

    _SYNTH_DEADLINE_S = 45.0

    def _tts_worker(self, gen: int) -> None:
        while self._alive(gen):
            try:
                fut = self._q_tts.get(timeout=0.3)
            except queue.Empty:
                continue
            # espera em fatias curtas: um `result(timeout=45)` seco deixava
            # esta thread surda a um `Pausar` por quase um minuto
            res = None
            t_dead = time.monotonic() + self._SYNTH_DEADLINE_S
            while True:
                if not self._alive(gen):
                    fut.cancel()
                    return
                try:
                    res = fut.result(timeout=0.5)
                    break
                except FutureTimeout:
                    if time.monotonic() >= t_dead:
                        log.warning("síntese passou de %.0fs, frase pulada",
                                    self._SYNTH_DEADLINE_S)
                        fut.cancel()
                        break
                except Exception:
                    log.exception("síntese não concluiu")
                    break
            if res is None:
                continue
            audio, rate = res
            pcm = audio.pcm
            if audio.samplerate != SR_NATIVE:
                pcm = dsp.resample(pcm, audio.samplerate, SR_NATIVE)
            # proteção de atraso máximo: descarta fila antiga e avisa
            if self._mixer.tts_backlog_seconds() > self.config.max_backlog_s:
                log.warning("backlog > %.0fs, pulando para o ao vivo",
                            self.config.max_backlog_s)
                self._mixer.clear_tts()
            self._mixer.enqueue_tts(pcm)
            with self._lock:
                self._state.rate_pct = rate

    def _adaptive_rate(self) -> int:
        """Escada de aceleração conforme o atraso acumulado da fila TTS."""
        backlog = self._mixer.tts_backlog_seconds()
        rate = 0
        for threshold, r in self.config.rate_ladder:
            if backlog >= float(threshold):
                rate = int(r)
        # velocidade base escolhida na GUI soma à escada; teto do edge-tts
        base = int(round((self.config.tts_speed - 1.0) * 100))
        return min(100, base + rate)

    # -------------------------------------------------- ControllerProtocol
    def start_pipeline(self) -> None:
        with self._lock:
            # _starting cobre a janela de carregamento dos modelos: sem ela,
            # um segundo clique em Iniciar criava o pipeline inteiro em dobro
            if self._running or self._starting:
                return
            self._starting = True
        threading.Thread(target=self._start_impl, daemon=True,
                         name="pipeline-start").start()

    def _start_impl(self) -> None:
        try:
            # uma parada anterior pode ainda estar fechando a captura/mixer;
            # esperá-la aqui (fora da GUI) mantém a ordem sem congelar a janela
            td = self._teardown
            if td is not None and td.is_alive():
                td.join(timeout=10.0)
            try:
                self._ensure_components()
            except Exception as exc:
                log.exception("falha ao carregar componentes")
                self._set_status(f"erro ao carregar: {exc}")
                return
            if self._tts_pool is None:   # o stop anterior descartou o pool
                self._tts_pool = ThreadPoolExecutor(
                    max_workers=2, thread_name_prefix="tts-synth")
            with self._lock:
                self._generation += 1
                gen = self._generation
                self._running = True
                self._state.running = True
            self._last_block_ts = time.monotonic()
            self._mixer.start()
            self._capture.start(self._on_capture_block, self._on_capture_error)
            for fn, name in ((self._asr_worker, "asr"), (self._mt_worker, "mt"),
                             (self._tts_worker, "tts"),
                             (self._capture_watchdog, "watchdog")):
                t = threading.Thread(target=fn, args=(gen,), daemon=True,
                                     name=f"stage-{name}")
                t.start()
                self._threads.append(t)
            self._set_status("ouvindo…")
        finally:
            with self._lock:
                self._starting = False

    def stop_pipeline(self) -> None:
        """Para o pipeline e RETORNA NA HORA.

        Fechar a captura e esvaziar o VAD leva alguns segundos; como isto é
        chamado da thread da GUI (botão Pausar / fechar a janela), fazer o
        trabalho aqui congelava a janela até 7 s. O estado muda de imediato e
        o desmonte roda em segundo plano.
        """
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._state.running = False
            self._stopping = True
            self._generation += 1   # aposenta os workers desta rodada
        self._set_status("parando…")
        self._teardown = threading.Thread(target=self._stop_impl, daemon=True,
                                          name="pipeline-stop")
        self._teardown.start()

    def _stop_impl(self) -> None:
        try:
            try:
                with self._capture_lock:   # não deixa o watchdog reabrir agora
                    self._capture.stop()
                self._segmenter.flush()
                self._mixer.stop()
            except Exception:
                log.exception("erro ao parar")
            self._threads.clear()
            # descarta o que ficou pendente para não tocar áudio velho num restart
            for q in (self._q_asr, self._q_mt, self._q_tts):
                while True:
                    try:
                        item = q.get_nowait()
                    except queue.Empty:
                        break
                    cancel = getattr(item, "cancel", None)
                    if cancel is not None:
                        try:
                            cancel()
                        except Exception:
                            pass
            # descarta o pool: as sínteses já enfileiradas são de áudio velho e
            # as threads dele são não-daemon (segurariam a saída do processo)
            pool, self._tts_pool = self._tts_pool, None
            if pool is not None:
                try:
                    pool.shutdown(wait=False, cancel_futures=True)
                except TypeError:      # Python < 3.9
                    pool.shutdown(wait=False)
            self._set_status("pausado")
        finally:
            with self._lock:
                self._stopping = False

    def wait_stopped(self, timeout: float = 3.0) -> None:
        """Aguarda o desmonte terminar (usado no fechamento da janela)."""
        td = self._teardown
        if td is not None and td.is_alive():
            td.join(timeout=timeout)

    # ------------------------------------------------------ watchdog da captura
    _CAPTURE_STALL_S = 20.0

    def _capture_watchdog(self, gen: int) -> None:
        """Reabre a captura quando ela para de entregar áudio.

        O `record()` do soundcard bloqueia para sempre se o loopback morrer
        (troca de dispositivo de saída, app de origem fechando o stream): o
        app seguia mostrando "Ouvindo" sem nada chegar, para sempre. Silêncio
        prolongado também seca o fluxo no WASAPI, por isso o limiar é folgado:
        reabrir custa milissegundos e não corta áudio real.
        """
        while self._alive(gen):
            time.sleep(1.0)
            if not self._alive(gen):
                return
            parada = time.monotonic() - self._last_block_ts
            if parada < self._CAPTURE_STALL_S:
                continue
            log.warning("captura sem áudio há %.0fs, reabrindo", parada)
            self._last_block_ts = time.monotonic()   # evita reabrir em rajada
            try:
                self._restart_capture()
            except Exception:
                log.exception("falha ao reabrir a captura")

    def _restart_capture(self) -> None:
        """Fecha e reabre a captura (watchdog e troca de dispositivo)."""
        from tradutor.audio_io import LoopbackCapture
        with self._capture_lock:
            old = self._capture
            if old is not None:
                old.stop()
            if not self._running:   # pausou no meio da troca: não reabrir
                return
            cap = LoopbackCapture(device_hint=self.config.capture_device_hint,
                                  samplerate=SR_NATIVE)
            self._capture = cap
            self._last_block_ts = time.monotonic()
            cap.start(self._on_capture_block, self._on_capture_error)

    def set_gain_original(self, value: float) -> None:
        self.config.gain_original = value
        if self._mixer:
            self._mixer.gain_original = value
            self._apply_duck()

    def set_gain_tts(self, value: float) -> None:
        self.config.gain_tts = value
        if self._mixer:
            self._mixer.gain_tts = value

    def set_duck_level(self, value: float) -> None:
        # `value` é a fração de REDUÇÃO (0.75 = original cai 75% ao falar)
        self.config.duck_level = value
        if self._mixer:
            self._apply_duck()

    def set_tts_speed(self, value: float) -> None:
        self.config.tts_speed = float(value)

    def set_tts_voice(self, name: str) -> None:
        """Troca a voz do Edge-TTS na hora (a frase em curso termina na antiga)."""
        name = (name or "").strip()
        if not name:
            return
        self.config.tts_voice = name
        if self._speaker is not None:
            self._speaker.voice = name

    def set_glossary(self, name: str) -> None:
        """Troca o tema do glossário na hora (a frase em curso termina no antigo).

        Carregar um tema compila centenas de regexes (~0,1 s), então roda numa
        thread para não travar a interface; a troca do atributo é atômica e o
        estágio de tradução lê `self._glossary` a cada frase.
        """
        name = (name or "").strip()
        if not name:
            return
        self.config.glossario = name
        if self._glossary is None:      # antes do primeiro start: vale quando ele iniciar
            return

        def _load() -> None:
            from tradutor.glossary import Glossary, theme_path
            try:
                self._glossary = Glossary(theme_path(name))
                log.info("glossário trocado para %s", name)
            except Exception:
                log.exception("falha ao carregar o glossário %s", name)

        threading.Thread(target=_load, name="glossario", daemon=True).start()

    def _apply_duck(self) -> None:
        """duck do mixer = ganho restante: gain_original × (1 − redução)."""
        self._mixer.duck_level = max(
            0.0, self.config.gain_original * (1.0 - self.config.duck_level))

    def set_capture_device(self, name: str) -> None:
        self.config.capture_device_hint = name or None
        if self._running:
            # fora da thread da GUI: fechar a captura pode levar 2 s
            threading.Thread(target=self._restart_capture, daemon=True,
                             name="capture-swap").start()

    def set_output_device(self, index: int) -> None:
        if index < 0:
            self.config.output_device = None
            self.config.output_device_name = None
        else:
            self.config.output_device = index
            self.config.output_device_name = next(
                (name for i, name in self.list_output_devices() if i == index),
                None)
        if self._mixer is None:
            return
        if self._running:
            # abrir/fechar streams demora; não pode ser na thread da GUI
            threading.Thread(target=self._swap_mixer_output, daemon=True,
                             name="output-swap").start()
        else:
            # pausado: o próximo start() reutiliza o mixer, então o device
            # precisa ser atualizado aqui
            self._mixer.output_device = self._resolve_output_device()

    def _swap_mixer_output(self) -> None:
        """Troca a saída a quente: abre um mixer novo no dispositivo escolhido
        e só então descarta o antigo (a fala pendente é perdida, mas a próxima
        frase já sai no dispositivo novo)."""
        from tradutor.audio_io import OutputMixer
        new = OutputMixer(output_device=self._resolve_output_device(),
                          samplerate=SR_NATIVE)
        new.agc_enabled = True
        new.gain_original = self.config.gain_original
        new.gain_tts = self.config.gain_tts
        try:
            new.start()
        except Exception as exc:
            log.exception("falha ao abrir a nova saída")
            self._set_status(f"erro: saída indisponível ({exc})")
            return
        old, self._mixer = self._mixer, new
        self._apply_duck()
        if old is not None:
            try:
                old.stop()
            except Exception:
                log.exception("erro ao fechar o mixer antigo")
        self._set_status(f"saída: {self.config.output_device_name or 'padrão do sistema'}")

    def _resolve_output_device(self):
        """Resolve o dispositivo de saída salvo para um índice VÁLIDO nesta máquina.

        Índices do sounddevice mudam entre máquinas (e até entre boots), então
        o nome é a fonte da verdade; um índice sem nome é config legada e não
        é confiável. Sem correspondência => None (padrão do sistema).
        """
        cfg = self.config
        if not cfg.output_device_name:
            if cfg.output_device is not None:
                log.warning("output_device=%r salvo sem nome (config de outra "
                            "máquina?), usando a saída padrão do sistema",
                            cfg.output_device)
            return None
        saved = cfg.output_device_name
        for index, name in self.list_output_devices():
            # prefixo cobre nomes truncados em 31 chars salvos pela lista MME
            # antiga (ex.: "Alto-falantes (2- Dell AC511 US")
            if name == saved or name.startswith(saved):
                if index != cfg.output_device:
                    log.info("saída %r mudou de índice (%r -> %r)",
                             name, cfg.output_device, index)
                    cfg.output_device = index
                if name != saved:
                    cfg.output_device_name = name
                return index
        log.warning("saída salva %r não existe nesta máquina, usando a "
                    "saída padrão do sistema", cfg.output_device_name)
        return None

    def test_output(self) -> None:
        """Toca um bip curto na saída selecionada (diagnóstico da GUI)."""
        threading.Thread(target=self._test_output_impl, daemon=True,
                         name="test-output").start()

    def _test_output_impl(self) -> None:
        t = np.arange(int(0.6 * SR_NATIVE), dtype=np.float32) / SR_NATIVE
        tone = (0.35 * np.sin(2 * np.pi * 660.0 * t)).astype(np.float32)
        n = int(0.01 * SR_NATIVE)
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        tone[:n] *= ramp
        tone[-n:] *= ramp[::-1]
        if self._running and self._mixer:
            # caminho real da voz: se o bip sair no lugar certo, a voz também sai
            self._mixer.enqueue_tts(tone)
            self._set_status("bip de teste enviado à saída atual")
            return
        try:
            import sounddevice as sd
            stereo = np.repeat(tone.reshape(-1, 1), 2, axis=1)
            sd.play(stereo, samplerate=SR_NATIVE,
                    device=self._resolve_output_device(), blocking=True)
            self._set_status("bip de teste tocado")
        except Exception as exc:
            log.exception("falha no bip de teste")
            self._set_status(f"erro no teste de saída: {exc}")

    def skip_to_live(self) -> None:
        if self._mixer:
            self._mixer.clear_tts()

    def get_ui_state(self) -> UiState:
        with self._lock:
            st = UiState(**vars(self._state))
            if self._starting:
                # o botão fica em "Pausar" durante o carregamento; sem isso a
                # GUI o devolvia para "Iniciar" e convidava um clique duplicado
                st.running = True
        if self._mixer and self._running:
            st.backlog_seconds = self._mixer.tts_backlog_seconds()
        return st

    def list_capture_devices(self) -> list:
        from tradutor.audio_io import list_loopback_devices
        try:
            return list_loopback_devices()
        except Exception:
            return []

    def list_output_devices(self) -> list:
        from tradutor.audio_io import list_output_devices
        try:
            return list_output_devices()
        except Exception:
            return []

    def _set_status(self, msg: str) -> None:
        with self._lock:
            self._state.status = msg
        log.info("status: %s", msg)


_SINGLETON_MUTEX = None   # mantém o handle vivo pela vida do processo


def _another_instance_running() -> bool:
    """True se já existe um Tradutor Simultâneo aberto (mutex nomeado)."""
    global _SINGLETON_MUTEX
    if os.name != "nt":
        return False
    try:
        import ctypes
        _SINGLETON_MUTEX = ctypes.windll.kernel32.CreateMutexW(
            None, False, "TradutorSimultaneo-instancia-unica")
        return ctypes.windll.kernel32.GetLastError() == 183  # ERROR_ALREADY_EXISTS
    except Exception:
        return False


def _disable_console_quickedit() -> None:
    """Desliga o QuickEdit do console do Windows.

    Com ele ligado (padrão do Windows 11), um clique dentro da janela preta
    põe o console em modo de seleção e BLOQUEIA o processo no próximo print.
    Como o app loga a cada frase, isso congelava o tradutor inteiro (janela,
    voz e legenda) até alguém apertar Esc na janela do console.
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        handle = k32.GetStdHandle(-10)          # STD_INPUT_HANDLE
        mode = ctypes.c_uint32()
        if not k32.GetConsoleMode(handle, ctypes.byref(mode)):
            return                              # sem console (pythonw)
        ENABLE_QUICK_EDIT, ENABLE_EXTENDED_FLAGS = 0x0040, 0x0080
        k32.SetConsoleMode(
            handle, (mode.value & ~ENABLE_QUICK_EDIT) | ENABLE_EXTENDED_FLAGS)
    except Exception:
        log.debug("não foi possível desligar o QuickEdit", exc_info=True)


def _boost_priority() -> None:
    """Prioridade ABOVE_NORMAL no Windows.

    Em produção, picos de CPU de outros processos (antivírus etc.) deixaram o
    ASR com proc de ~30 s (rtf > 4) e o app mudo por meio minuto. Com
    prioridade acima do normal o tradutor segue em tempo real mesmo com
    antivírus/afins rodando.
    """
    try:
        import ctypes
        ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass(k32.GetCurrentProcess(),
                             ABOVE_NORMAL_PRIORITY_CLASS)
    except Exception:
        log.debug("não foi possível elevar a prioridade", exc_info=True)


def _log_handlers() -> list:
    """Console + arquivo rotativo na raiz do projeto (o console some ao fechar)."""
    handlers: list = [logging.StreamHandler()]
    try:
        from logging.handlers import RotatingFileHandler
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        handlers.append(RotatingFileHandler(
            os.path.join(root, "tradutor.log"), maxBytes=1_000_000,
            backupCount=2, encoding="utf-8"))
    except Exception:
        pass
    return handlers


def main() -> None:
    _disable_console_quickedit()
    _boost_priority()
    if _another_instance_running():
        # duas instâncias disputariam a captura e falariam em saídas
        # diferentes, melhor avisar e sair
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showwarning(
                "Tradutor Simultâneo",
                "O Tradutor Simultâneo já está aberto.\n\n"
                "Procure a janela existente (ou o ícone na barra de tarefas) "
                "antes de abrir outro.")
            root.destroy()
        except Exception:
            print("Tradutor Simultâneo já está aberto.")
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=_log_handlers())
    # bibliotecas tagarelas em INFO
    for noisy in ("argostranslate", "argostranslate.utils", "stanza",
                  "httpx", "urllib3", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # aviso benigno do soundcard na abertura do recorder (spam)
    import warnings
    try:
        from soundcard import SoundcardRuntimeWarning
        warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning)
    except Exception:
        pass
    config = AppConfig.load()
    pipeline = Pipeline(config)

    from tradutor.gui import App
    app = App(pipeline, config)
    pipeline.attach_gui(app)
    app.run()
    # A janela já fechou e a config já foi salva. Sair pela via normal faria o
    # Python esperar a síntese em andamento (até ~30 s com a rede ruim): o app
    # sumia da tela mas o processo continuava vivo segurando o áudio e o mutex
    # de instância única, e reabrir dava "já está aberto".
    logging.shutdown()
    os._exit(0)


if __name__ == "__main__":
    main()
