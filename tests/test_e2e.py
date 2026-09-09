"""Teste ponta a ponta: injeta fala em inglês no CABLE Input e verifica se o
pipeline completo (captura -> VAD -> ASR -> tradução -> TTS -> mixer) produz
legenda em português e áudio de voz.

Executar da raiz do projeto:  .venv\\Scripts\\python.exe tests\\test_e2e.py
Requer: VB-CABLE instalado e internet (edge-tts).
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
for noisy in ("argostranslate", "stanza", "httpx", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

import numpy as np
import sounddevice as sd

from tradutor import dsp
from tradutor.config import AppConfig
from tradutor.main import Pipeline

TEXT_EN = ("The Federal Reserve decided to keep interest rates unchanged. "
           "Stock markets in New York rallied strongly after the announcement.")


class FakeGui:
    """Coleta as legendas que o pipeline empurraria para o overlay."""

    def __init__(self):
        self.subtitles = []

    def push_subtitle(self, original, translated):
        self.subtitles.append((original, translated))
        print(f"\n=== LEGENDA ===\n  EN: {original}\n  PT: {translated}\n")


def find_cable_output_index():
    """Índice sounddevice do dispositivo de SAÍDA 'CABLE Input' (WASAPI de preferência)."""
    apis = sd.query_hostapis()
    best = None
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_output_channels"] > 0 and "CABLE Input" in dev["name"]:
            api_name = apis[dev["hostapi"]]["name"]
            if "WASAPI" in api_name:
                return i
            if best is None:
                best = i
    return best


def main():
    fails = []

    cable_out = find_cable_output_index()
    if cable_out is None:
        print("FALHA: dispositivo 'CABLE Input' não encontrado no sounddevice.")
        sys.exit(1)
    print(f"CABLE Input para injeção: índice {cable_out}")

    config = AppConfig.load()
    pipeline = Pipeline(config)
    gui = FakeGui()
    pipeline.attach_gui(gui)

    print(">> start_pipeline(): aguardando carga dos modelos…")
    t0 = time.monotonic()
    pipeline.start_pipeline()
    # `running` já fica True durante o carregamento (é o que mantém o botão
    # da GUI em "Pausar"); a captura só está aberta quando o status vira
    # "ouvindo…". Injetar antes disso perdia o começo da frase.
    def _ouvindo(st):
        return st.running and st.status.startswith("ouvindo")

    while time.monotonic() - t0 < 240:
        st = pipeline.get_ui_state()
        if _ouvindo(st):
            break
        if st.status.startswith("erro"):
            print(f"FALHA ao iniciar: {st.status}")
            sys.exit(1)
        time.sleep(1)
    st = pipeline.get_ui_state()
    if not _ouvindo(st):
        print(f"FALHA: pipeline não iniciou em 240s (status: {st.status})")
        sys.exit(1)
    print(f">> pipeline rodando em {time.monotonic() - t0:.0f}s. "
          f"Gerando fala em inglês…")

    # fala em inglês via edge-tts (simula o locutor gringo)
    from tradutor.translate_tts import TtsSpeaker
    speaker_en = TtsSpeaker(voice="en-US-AriaNeural")
    audio = speaker_en.synth(TEXT_EN)
    if audio is None:
        print("FALHA: não gerou o áudio em inglês (internet?)")
        sys.exit(1)
    pcm48 = dsp.resample(audio.pcm, audio.samplerate, 48000)
    print(f">> Injetando {len(pcm48)/48000:.1f}s de inglês no CABLE…")
    sd.play(pcm48, 48000, device=cable_out)
    sd.wait()
    print(">> Injeção concluída. Aguardando legenda (até 60s)…")

    t1 = time.monotonic()
    first_subtitle_at = None
    while time.monotonic() - t1 < 60:
        if gui.subtitles and first_subtitle_at is None:
            first_subtitle_at = time.monotonic() - t1
        st = pipeline.get_ui_state()
        if gui.subtitles and st.backlog_seconds == 0 and \
                time.monotonic() - t1 > 20:
            break  # legenda recebida e voz já reproduzida
        time.sleep(1)

    st = pipeline.get_ui_state()
    print(f"\n>> Resultado: {len(gui.subtitles)} legenda(s); "
          f"idioma detectado: {st.detected_lang}; backlog: {st.backlog_seconds:.1f}s")

    if not gui.subtitles:
        fails.append("nenhuma legenda produzida")
    else:
        all_pt = " ".join(t for _, t in gui.subtitles).lower()
        if first_subtitle_at is not None:
            print(f">> Primeira legenda {first_subtitle_at:.1f}s após o fim da injeção")
        if "federal reserve" not in all_pt:
            fails.append(f"glossário: 'Federal Reserve' ausente na tradução: {all_pt!r}")
        if "juro" not in all_pt and "taxa" not in all_pt:
            fails.append(f"tradução suspeita (sem 'juros/taxas'): {all_pt!r}")
    if st.detected_lang.lower() != "en":
        fails.append(f"idioma detectado {st.detected_lang!r}, esperado EN")

    pipeline.stop_pipeline()
    time.sleep(1)

    if fails:
        print("\nFALHAS:")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print("\nE2E OK: captura, ASR, tradução com glossário, TTS e mixer funcionando.")
    sys.exit(0)


if __name__ == "__main__":
    main()
