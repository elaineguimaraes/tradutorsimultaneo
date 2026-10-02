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

from tradutor.translate_tts import (JOB_END as TtsJob_END, Translator, TtsSpeaker,  # noqa: E402
                                    _chunk_long, _dedupe_echo, _join_pieces,
                                    _src_has_adjacent_dup, _src_repeats_content_word,
                                    _fix_number_echo)

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

def teste_eco_e_divisao() -> None:
    """Lógica pura (sem modelo): eco do MT sem origem e divisão de frase longa."""
    print("\n=== (0) eco do MT e divisão de frases longas ===")
    # ecos que NÃO existem no inglês: devem ser removidos
    ecos = [
        ("o próximo Nike Nike", "the next Nike", "o próximo Nike"),
        ("DGN Play Play", "DGN play", "DGN Play"),
        ("280.61, RTY, RTY", "280.61, RTY", "280.61, RTY"),
        ("Venha, venha.", "Come on!", "Venha."),
        ("Certo, certo!", "Alright!", "Certo!"),
        ("repicando repicando", "ripping", "repicando"),
        ("Para realmente ficar na camisa da Piper, você tem que ficar na camisa da Piper.",
         "To actually stay in Piper's shirt, you have", "Para realmente ficar na camisa da Piper."),
        ("os QQs estão escancarando os QQs estão escancarados",
         "the QQs are wide open", "os QQs estão escancarando"),
        ("Comigo dia após dia, dia fora, dia após dia.",
         "With me every day", "Comigo dia após dia."),
        ("Tesla subiu 3%, Meta aumentou 2% de rum, e Meta subiu 2% de rum.",
         "Tesla was up 3%, Meta gained 2%", "Tesla subiu 3%, Meta aumentou 2% de rum."),
    ]
    for pt, en, esperado in ecos:
        got, mudou = _dedupe_echo(pt, en)
        check(got == esperado and mudou, f"eco removido: {pt!r} -> {got!r}")
    got, _ = _dedupe_echo("90% ano ano anos", "90% year over year")
    check("ano ano" not in got, f"'ano ano' colapsado: {got!r}")
    # repetição legítima (o inglês também repete): intocada
    legit = [
        ("muito, muito", "really, really"),
        ("Muitos de nós antes de você chegar hoje, muitos de nós estávamos conversando.",
         "A lot of us before you got here today, a lot of us were talking."),
        ("5-SIM para a conta Funded Features, 5-SIM para a conta Funded Options",
         "5-SIM to Funded Features account, 5-SIM to Funded Options account"),
    ]
    for pt, en in legit:
        got, mudou = _dedupe_echo(pt, en)
        check(got == pt and not mudou, f"repetição legítima preservada: {got!r}")

    # inglês com repetição de conteúdo: a poda de frase não pode agir
    # (o paralelismo em inglês vira trecho repetido legítimo em português)
    paralelos = [
        ("Preço acima da média e volume acima da média.",
         "Price above average and volume above average."),
        ("Se o preço for acima de 100, compre, se o preço cair abaixo de 90, venda.",
         "If price goes above 100 buy it, if price falls below 90 sell it."),
        ("A ação de preço é fundamental e a ação de preço diz tudo.",
         "Price action is key and price action tells you everything."),
    ]
    for pt, en in paralelos:
        check(_src_repeats_content_word(en), f"guarda: inglês repete conteúdo: {en!r}")
        got, mudou = _dedupe_echo(pt, en)
        check(got == pt and not mudou, f"paralelismo preservado: {got!r}")
    check(not _src_repeats_content_word("To actually stay in Piper's shirt, you have"),
          "guarda não dispara sem repetição de conteúdo")

    # gagueira de stopword no inglês não desliga o colapso de palavra duplicada
    check(not _src_has_adjacent_dup("the the market is up to to 3"),
          "stopword duplicada não conta como repetição")
    check(_src_has_adjacent_dup("really really strong"), "conteúdo duplicado conta")
    got, _ = _dedupe_echo("o próximo Nike Nike", "the the next Nike")
    check(got == "o próximo Nike", f"colapso com gagueira de stopword: {got!r}")

    # eco com conector, cauda repetida e número concatenado
    novos = [
        ("apenas assine e assine.", "just sign up", "apenas assine."),
        ("só transparência e transparência.", "just transparency", "só transparência."),
        ("você pode verificar aqui mesmo aqui.", "check out right here",
         "você pode verificar aqui mesmo."),
        ("400.400.", "400.", "400."),
        ("preço em 400400 hoje", "price at 400 today", "preço em 400 hoje"),
        ("preço em 400,400 hoje", "price at 400 today", "preço em 400 hoje"),
    ]
    for pt, en, esperado in novos:
        got, mudou = _dedupe_echo(pt, en)
        check(got == esperado and mudou, f"eco removido: {pt!r} -> {got!r}")
    # decimais: o MT localiza "1.1" -> "1,1"; os tokens XPROTECTEDnX não são números
    decimais = [
        ("subiu 1,1 por cento", "XPROTECTED0X and XPROTECTED1X said CPI rose 1.1 percent."),
        ("1,1 bilhão de dólares", "Revenue in Q1 was 1.1 billion dollars."),
        ("de 4 para 4,4 por cento", "The yield went from 4 to 4.4 percent."),
        ("de 10 para 10,10 hoje", "It moved from 10 to 10.10 today."),
        ("de 5 para 5,5", "from 5 to 5.5"),
        ("de 55 para 5", "from 5 to 5"),
    ]
    for pt, en in decimais:
        check(_fix_number_echo(pt, en) == pt, f"decimal preservado: {pt!r}")
        got, mudou = _dedupe_echo(pt, en)
        check(got == pt and not mudou, f"decimal intacto no dedupe: {got!r}")
    check(_fix_number_echo("400.400.", "400.") == "400.", "400.400. -> 400.")

    # tail rule não come idiomas "W e W" / "W a W" no fim da frase
    for pt, en in (("continua subindo e subindo.", "keeps going up and up."),
                   ("lado a lado.", "side by side."),
                   ("pouco a pouco.", "little by little."),
                   ("avançou lado a lado.", "it moved sideways"),
                   ("continua subindo e subindo.", "it keeps rising")):
        got, mudou = _dedupe_echo(pt, en)
        check(got == pt and not mudou, f"idioma de fim de frase preservado: {got!r}")
    # "No, no": ênfase legítima; gagueira de artigo não desliga o colapso
    got, mudou = _dedupe_echo("Não, não, está errado.", "No, no, that's wrong.")
    check(got == "Não, não, está errado." and not mudou, f"'No, no' preservado: {got!r}")
    check(_src_has_adjacent_dup("No, no, that's wrong") and
          not _src_has_adjacent_dup("the the market"), "stutter só de artigos")

    intactos = [
        ("mais e mais", "more and more"),
        ("dia após dia", "day after day"),
        ("passo a passo", "step by step"),
        ("pouco a pouco", "little by little"),
        ("cara a cara", "face to face"),
        ("subiu de novo e de novo", "up again and again"),
        ("custa US$ 1.125 hoje", "it costs $1,125 today"),
        ("subiu 3.125 hoje", "up 3.125 today"),
        ("preço em 400 e 4.000", "price at 400 and 4,000"),
        ("para o mercado para", "for the market for"),
    ]
    for pt, en in intactos:
        got, mudou = _dedupe_echo(pt, en)
        check(got == pt and not mudou, f"legítimo preservado: {got!r}")

    # reagrupamento: capitalização e vírgula pendurada
    j = _join_pieces(["Algo subiu.", "E ninguém sabe."],
                     ["something went up", "and nobody knows"], [True, False])
    check(j == "Algo subiu, e ninguém sabe.", f"'E' de uma letra desce: {j!r}")
    j = _join_pieces(["Algo subiu.", "EUA sobem."], ["it went up", "and USA rises"],
                     [True, False])
    check("EUA" in j, f"sigla preservada: {j!r}")
    j = _join_pieces(["Algo subiu.", "Acho que sim."], ["it went up", "I think so"],
                     [True, False])
    check(j == "Algo subiu, acho que sim.", f"origem 'I think' libera minúscula: {j!r}")
    j = _join_pieces(["Algo subiu.", "Nike subiu."], ["it went up", "Nike went up"],
                     [True, False])
    check("Nike" in j, f"nome próprio preservado: {j!r}")
    j = _join_pieces(["Algo subiu.", ""], ["it went up", "so"], [True, False])
    check(j == "Algo subiu", f"sem vírgula pendurada: {j!r}")

    # divisão suave de frase longa sem pontuação
    run_on = ("So their are downside gap fills so just from a technical "
              "perspective if there is profit taking out of the gate that it's "
              "just something obviously you should be aware")
    pecas = _chunk_long(run_on)
    print(f"  pedaços: {pecas}")
    check(len(pecas) >= 2, "run-on de 30 palavras foi dividido")
    check(all(4 <= len(p.split()) <= 22 for p in pecas), "pedaços entre 4 e 22 palavras")
    check(" ".join(pecas) == run_on, "divisão não perde nem altera palavras")
    check(_chunk_long("one two three four five") == ["one two three four five"],
          "frase curta fica inteira")
    sem_fronteira = " ".join(f"w{i}" for i in range(30))
    check(_chunk_long(sem_fronteira) == [sem_fronteira], "sem fronteira: fica inteira")
    com_virgula = ("alpha beta gamma delta epsilon zeta eta, theta iota kappa lambda "
                   "mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega")
    p2 = _chunk_long(com_virgula)
    check(len(p2) == 2 and not p2[0].endswith(","), f"vírgula preferida: {p2}")
    tok = ("they said XPROTECTED0X is strong and XPROTECTED1X is weak but "
           "XPROTECTED2X was flat so everyone sat still waiting for the next "
           "headline to hit the tape today")
    check(all("XPROTECTED" not in w or w.startswith("XPROTECTED") and w.endswith("X")
              for p in _chunk_long(tok) for w in p.split()), "tokens protegidos intactos")
    # reagrupamento: continuação termina em vírgula, minúscula se a origem era
    juntado = _join_pieces(["O mercado subiu.", "Mas caiu depois."],
                           ["the market rose", "but fell later"], [True, False])
    check(juntado == "O mercado subiu, mas caiu depois.", f"reagrupado: {juntado!r}")


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

    if "en" in tr._ready_langs:
        for en, deve in (
                ("Price above average and volume above average.", "volume acima"),
                ("If price goes above 100 buy it, if price falls below 90 sell it.", "90"),
                ("Price action is key and price action tells you everything.", "tudo")):
            pt = tr.translate(en, "en")
            print(f"  EN : {en}\n  PT : {pt}")
            check(deve in pt.lower(), f"eco não poda paralelismo real ({deve!r} em {pt!r})")
        # concordância dos termos de prop firm (MT real + glossário)
        from tradutor.glossary import Glossary, theme_path
        gl = Glossary(theme_path("trading"))
        for en, deve in (("I passed my evals", "minhas avaliações"),
                         ("two funded accounts", "duas contas financiadas"),
                         ("an eval", "uma avaliação"),
                         ("My e-vow got reset", "minha avaliação"),
                         ("Your funded account is safe", "sua conta financiada")):
            pt = gl.apply(lambda x: tr.translate(x, "en"), en)
            print(f"  EN : {en}\n  PT : {pt}")
            check(deve in pt.lower(), f"concordância ({deve!r} em {pt!r})")
        # decimais reais (a localização do MT "1.1" -> "1,1" não é eco)
        for en, deve in (("Revenue in Q1 was 1.1 billion dollars.", "1,1"),
                         ("The yield went from 4 to 4.4 percent.", "4,4"),
                         ("It moved from 10 to 10.10 today.", "10,10")):
            pt = tr.translate(en, "en")
            print(f"  EN : {en}\n  PT : {pt}")
            check(deve in pt, f"decimal preservado ({deve!r} em {pt!r})")
        pt = tr.translate("Fed and Powell said CPI rose 1.1 percent.", "en")
        check("1,1" in pt, f"decimal com nomes preservado: {pt!r}")

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

    # streaming real: os trechos somam o mesmo áudio e o 1º chega antes do fim
    import time
    t0 = time.monotonic()
    job = tts.synth_stream(FRASE_PT, rate_pct=0)
    total_ms = (time.monotonic() - t0) * 1000
    trechos = []
    while True:
        it = job.get(timeout=1.0)
        if it is None or it is TtsJob_END:
            break
        trechos.append(it[0])
    if not trechos:
        pulados.append("TTS: streaming real falhou (rede instável)")
        print("  [PULADO] streaming real falhou")
    else:
        y = np.concatenate(trechos)
        primeiro_ms = (job.t_first - t0) * 1000 if job.t_first else -1
        print(f"  streaming: {len(trechos)} trechos, 1º em {primeiro_ms:.0f} ms, "
              f"síntese total {total_ms:.0f} ms, áudio {y.size / job.samplerate:.2f}s")
        check(len(trechos) >= 2, "o áudio chega em vários trechos")
        check(0 < primeiro_ms < total_ms, "1º trecho chega antes do fim da síntese")
        check(abs(y.size - a0.pcm.size) <= 2400,
              f"streaming e síntese inteira têm a mesma duração ({y.size} x {a0.pcm.size})")

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
