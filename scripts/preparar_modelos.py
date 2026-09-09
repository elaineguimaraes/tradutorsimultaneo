"""Pré-baixa e converte os modelos de IA para o primeiro uso não travar.

Executado pelo instalar.bat com o Python do .venv, a partir da raiz. Faz 4
etapas: (1) baixa o modelo Whisper, (2) baixa e converte o Opus-MT en->pt-BR
para CTranslate2 (só na primeira vez; demora alguns minutos), (3) faz uma
tradução de teste para confirmar qual backend ficou ativo, (4) testa a voz
(precisa de internet).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np


def main() -> int:
    ok = True

    print("  - Whisper (transcrição)…", flush=True)
    try:
        from tradutor.config import AppConfig
        from faster_whisper import WhisperModel
        model = WhisperModel(AppConfig.load().whisper_model, device="cpu",
                             compute_type="int8")
        list(model.transcribe(np.zeros(16000, dtype=np.float32))[0])
        print("    OK")
    except Exception as exc:
        print(f"    FALHOU: {exc}")
        ok = False

    print("  - Tradutor Opus-MT en->pt-BR (download ~1 GB + conversão para "
          "CTranslate2, alguns minutos, só na primeira vez)…", flush=True)
    try:
        from tradutor.translate_tts import Translator
        tr = Translator(target="pt")
        if not tr.prepare_local_model("en"):
            print("    AVISO: não foi possível preparar o Opus-MT local, "
                  "o app usará o Argos Translate como reserva (qualidade "
                  "menor, pt-PT)")
            ok = False
        else:
            print("    OK (backend ct2 ativo)")
    except Exception as exc:
        print(f"    FALHOU: {exc}")
        ok = False

    print("  - Tradução de teste…", flush=True)
    try:
        from tradutor.translate_tts import Translator
        # Instância nova de propósito: reproduz o que o app faz no boot e
        # confirma qual backend é escolhido a partir do que ficou em disco.
        tr = Translator(target="pt")
        tr.ensure_ready(["en"])
        out = tr.translate("Interest rates were left unchanged.", "en")
        print(f"    backend={tr.backend}  resultado={out!r}")
    except Exception as exc:
        print(f"    FALHOU: {exc}")
        ok = False

    print("  - Voz pt-BR (edge-tts, requer internet)…", flush=True)
    try:
        from tradutor.translate_tts import TtsSpeaker
        audio = TtsSpeaker().synth("Instalação concluída.")
        print("    OK" if audio is not None else
              "    sem internet agora, o app usa a voz normalmente quando houver conexão")
    except Exception as exc:
        print(f"    FALHOU: {exc}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
