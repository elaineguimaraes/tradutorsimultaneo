"""Teste standalone de translate_tts.py (tradução + TTS).

Executar:  .venv\\Scripts\\python.exe tests\\test_translate_tts.py

Parte A (tradução) exige os pacotes do backend; a primeira execução baixa o
pacote en->pt e pode demorar alguns minutos.
Parte B (TTS) exige internet: sem rede, o teste REPORTA e PULA (não falha).
Gera scratch/tts_sample.wav para conferência manual.
"""

from __future__ import annotations

import logging
import os
import sys
import wave

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

# O console do Windows é cp1252: sem isto, imprimir japonês estoura.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from tradutor.translate_tts import Translator, TtsSpeaker  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
# argostranslate/stanza logam cada tokenização em INFO: ruído demais.
for _n in ("argostranslate", "stanza", "urllib3", "filelock"):
    logging.getLogger(_n).setLevel(logging.WARNING)

FRASE_EN = ("The Federal Reserve raised interest rates by 25 basis points, "
            "signaling further tightening ahead.")
FRASE_PT = "O Federal Reserve subiu os juros em 25 pontos-base."

falhas: list[str] = []
pulados: list[str] = []


def check(cond: bool, msg: str) -> bool:
    """Registra o resultado de uma verificação e devolve o próprio booleano."""
    print(f"  [{'OK ' if cond else 'FALHA'}] {msg}")
    if not cond:
        falhas.append(msg)
    return cond


def salvar_wav(caminho: str, pcm: np.ndarray, sr: int) -> None:
    """Grava float32 mono como WAV PCM 16 bits."""
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    dados = np.clip(pcm, -1.0, 1.0)
    with wave.open(caminho, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((dados * 32767).astype("<i2").tobytes())


# ==========================================================================
# (a) Tradução
# ==========================================================================

def teste_traducao() -> None:
    print("\n=== (a) Translator ===")
    tr = Translator()
    print(f"  backend selecionado: {tr.backend}")
    if tr.backend == "none":
        pulados.append("tradução: nenhum backend disponível no ambiente")
        print("  [PULADO] nenhum backend de tradução instalado")
        return

    print("  ensure_ready(['en']): pode demorar no primeiro uso…")
    tr.ensure_ready(["en"])

    import time
    t0 = time.monotonic()
    saida = tr.translate(FRASE_EN, "en")
    frio = time.monotonic() - t0
    t0 = time.monotonic()
    tr.translate("The central bank will meet again in September to review policy.", "en")
    quente = time.monotonic() - t0
    print(f"\n  EN : {FRASE_EN}\n  PT : {saida}")
    print(f"  latência: 1ª chamada {frio*1000:.0f} ms (carrega modelo) / "
          f"2ª chamada {quente*1000:.0f} ms\n")

    if "en" not in tr._ready_langs:
        pulados.append("tradução en->pt: pacote não pôde ser baixado (offline?)")
        print("  [PULADO] pacote en->pt indisponível; translate() devolveu o original")
        check(saida.strip() != "", "translate() devolveu texto não vazio mesmo sem pacote")
    else:
        check(saida.strip() != "", "tradução não vazia")
        check(saida.strip().lower() != FRASE_EN.strip().lower(),
              "tradução difere do texto em inglês")
        check("juros" in saida.lower() or "taxa" in saida.lower() or "pontos" in saida.lower(),
              "tradução contém vocabulário esperado (juros/taxa/pontos)")

    # texto já em pt -> devolve o próprio texto
    mesmo = tr.translate(FRASE_PT, "pt")
    check(mesmo.strip() == FRASE_PT.strip(),
          f"src_lang='pt' devolve o próprio texto (obtido: {mesmo!r})")

    # idioma sem pacote -> devolve o original, sem exceção.
    # auto_install=False para o teste não disparar download de ja->en em 2º plano.
    tr2 = Translator(auto_install=False)
    exotico = "これはテストです。"
    try:
        saida_ja = tr2.translate(exotico, "ja")
        ok_ja = True
    except Exception as exc:  # noqa: BLE001
        saida_ja, ok_ja = "", False
        print(f"  exceção inesperada: {exc!r}")
    check(ok_ja, "idioma sem pacote não lança exceção")
    ja_instalado = tr2.backend == "argos" and tr2._argos_has_path("ja")
    if ok_ja and not ja_instalado:
        check(saida_ja == exotico,
              f"idioma sem pacote devolve o original (obtido: {saida_ja!r})")
    elif ok_ja:
        print(f"  (pacote ja->pt já instalado; tradução: {saida_ja!r})")

    # texto vazio -> ""
    check(tr.translate("", "en") == "", "texto vazio devolve ''")
    check(tr.translate("   ", "en") == "", "texto só com espaços devolve ''")


# ==========================================================================
# (b) TTS
# ==========================================================================

def teste_tts() -> None:
    print("\n=== (b) TtsSpeaker ===")
    try:
        import edge_tts  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        pulados.append(f"TTS: edge-tts não instalado ({exc})")
        print("  [PULADO] edge-tts não instalado")
        return

    tts = TtsSpeaker()
    a0 = tts.synth(FRASE_PT, rate_pct=0)
    if a0 is None:
        pulados.append("TTS: sem resposta do edge-tts (sem internet?)")
        print("  [PULADO] síntese falhou, provavelmente sem internet")
        tts.close()
        return

    print(f"  decodificador de mp3: {tts.decoder}")
    check(isinstance(a0.pcm, np.ndarray), "pcm é numpy.ndarray")
    check(a0.pcm.dtype == np.float32, f"pcm é float32 (obtido: {a0.pcm.dtype})")
    check(a0.pcm.ndim == 1, f"pcm é mono/1-D (obtido: {a0.pcm.ndim}-D)")
    check(a0.pcm.size > 0, "pcm não é vazio")
    check(a0.samplerate > 0, f"samplerate > 0 (obtido: {a0.samplerate})")
    check(float(np.max(np.abs(a0.pcm))) <= 1.0, "amostras dentro de [-1, 1]")
    check(float(np.max(np.abs(a0.pcm))) > 0.01, "áudio tem energia (não é silêncio)")

    d0 = a0.pcm.size / a0.samplerate
    print(f"  rate 0  -> {d0:.2f}s @ {a0.samplerate} Hz")

    a25 = tts.synth(FRASE_PT, rate_pct=25)
    if a25 is None:
        pulados.append("TTS: síntese com rate=25 falhou (rede instável)")
        print("  [PULADO] síntese rate=25 falhou")
    else:
        d25 = a25.pcm.size / a25.samplerate
        print(f"  rate 25 -> {d25:.2f}s @ {a25.samplerate} Hz")
        check(a25.pcm.size > 0 and a25.samplerate > 0, "áudio rate=25 válido")
        check(d25 < d0, f"duração com rate 25 ({d25:.2f}s) < rate 0 ({d0:.2f}s)")

    destino = os.path.join(_ROOT, "scratch", "tts_sample.wav")
    salvar_wav(destino, a0.pcm, a0.samplerate)
    print(f"  amostra salva em {destino}")
    check(os.path.getsize(destino) > 1000, "wav de amostra gravado")

    # texto vazio -> None, sem estourar
    check(tts.synth("   ") is None, "texto vazio devolve None")
    tts.close()


def main() -> int:
    teste_traducao()
    teste_tts()
    print("\n=== RESUMO ===")
    for p in pulados:
        print(f"  PULADO: {p}")
    if falhas:
        for f in falhas:
            print(f"  FALHA : {f}")
        print(f"\n{len(falhas)} verificação(ões) falharam.")
        return 1
    print("Todas as verificações executadas passaram.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
