"""Testes standalone do módulo `tradutor.sentence_buffer`.

Executar:  .venv\\Scripts\\python.exe tests\\test_sentence_buffer.py

Não depende de pytest nem de áudio: lógica pura de texto, com relógio falso.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from tradutor.sentence_buffer import SentenceBuffer  # noqa: E402

try:  # acentos no console do Windows
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass


class FakeClock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def test_segura_cauda_e_junta() -> None:
    b = SentenceBuffer()
    out = b.push("Growth was fine and we lost by a", "en", True)
    assert out == [], out
    assert b.pending() == "Growth was fine and we lost by a"
    out = b.push("point in a half.", "en", True)
    assert out == [("Growth was fine and we lost by a point in a half.", "en")], out
    assert b.pending() == ""


def test_forced_cut_emite_completo_e_segura_resto() -> None:
    b = SentenceBuffer()
    out = b.push("Stocks rose. Oil fell. Then the Fed", "en", True)
    assert out == [("Stocks rose. Oil fell.", "en")], out
    assert b.pending() == "Then the Fed"
    out = b.push("spoke today.", "en", False)
    assert out == [("Then the Fed spoke today.", "en")], out


def test_pausa_real_emite_tudo() -> None:
    b = SentenceBuffer()
    assert b.push("Stocks rose. Oil fell and then", "en", False) == \
        [("Stocks rose. Oil fell and then", "en")]
    assert b.pending() == ""


def test_reticencias_finais_removidas_na_juncao() -> None:
    b = SentenceBuffer()
    assert b.push("We are going right...", "en", True) == []
    out = b.push("up to the highs.", "en", True)
    assert out == [("We are going right up to the highs.", "en")], out
    b = SentenceBuffer()
    b.push("It was big…", "en", True)
    assert b.push("Today we rallied.", "en", True) == \
        [("It was big Today we rallied.", "en")]


def test_minuscula_so_para_palavra_funcional() -> None:
    b = SentenceBuffer()
    b.push("Inflation came in at 3.6% on a", "en", True)
    out = b.push("Month over month basis.", "en", True)
    assert out == [("Inflation came in at 3.6% on a month over month basis.",
                    "en")], out

    b = SentenceBuffer()
    b.push("The market is strong and", "en", True)
    out = b.push("Nvidia is up.", "en", True)
    assert out == [("The market is strong and Nvidia is up.", "en")], out

    b = SentenceBuffer()
    b.push("The market is strong and", "en", True)
    out = b.push("I think so.", "en", True)
    assert out == [("The market is strong and I think so.", "en")], out


def test_abreviacoes_nao_dividem() -> None:
    for txt in ("Then Mr. Rick Santelli said",
                "that the U.S. economy is fine and the p.m. session",
                "Dr. Smith vs. the market, etc. and Inc. shares"):
        b = SentenceBuffer()
        assert b.push(txt, "en", True) == [], txt
        assert b.pending() == txt
    b = SentenceBuffer()
    out = b.push("The U.S. economy went up. Then", "en", True)
    assert out == [("The U.S. economy went up.", "en")], out
    assert b.pending() == "Then"


def test_reticencias_no_meio_nao_dividem() -> None:
    b = SentenceBuffer()
    out = b.push("Well... I think it goes up", "en", True)
    assert out == [] and b.pending() == "Well... I think it goes up"
    out = b.push("Well… I guess", "en", True)
    assert out == [], out


def test_decimal_nao_divide() -> None:
    b = SentenceBuffer()
    assert b.push("CPI came in at 3.6% and", "en", True) == []


def test_teto_de_palavras_pendentes() -> None:
    b = SentenceBuffer(max_pending_words=5)
    out = b.push("one two three four five six", "en", True)
    assert out == [("one two three four five six", "en")], out
    assert b.pending() == ""
    assert b.push("one two three four five", "en", True) == []


def test_flush_due_com_relogio_falso() -> None:
    clk = FakeClock()
    b = SentenceBuffer(max_hold_s=10.0, clock=clk)
    b.push("We lost by a", "en", True)
    clk.t += 9.0
    assert b.flush_due() == []
    # continuação sem fronteira mantém o relógio antigo
    assert b.push("small amount of", "en", True) == []
    clk.t += 1.5
    assert b.flush_due() == [("We lost by a small amount of", "en")]
    assert b.flush_due() == []
    assert b.pending() == ""


def test_troca_de_idioma_libera_a_cauda_antiga() -> None:
    b = SentenceBuffer()
    b.push("We lost by a", "en", True)
    out = b.push("Hola amigos.", "es", False)
    assert out == [("We lost by a", "en"), ("Hola amigos.", "es")], out


def test_reset_descarta() -> None:
    b = SentenceBuffer()
    b.push("We lost by a", "en", True)
    b.reset()
    assert b.pending() == "" and b.flush() == [] and b.flush_due() == []
    assert b.push("point.", "en", True) == [("point.", "en")]


def test_flush_emite_pendente_e_nunca_vazio() -> None:
    b = SentenceBuffer()
    b.push("hello there", "en", True)
    assert b.flush() == [("hello there", "en")]
    assert b.push("   ", "en", False) == []
    assert b.push("  ", "en", True) == []


def test_lacuna_grande_libera_a_cauda_antiga() -> None:
    b = SentenceBuffer()
    assert b.push("We lost by a", "en", True, 10.0, 17.0) == []
    out = b.push("Then the Fed spoke.", "en", False, 19.0, 21.0)
    assert out == [("We lost by a", "en"), ("Then the Fed spoke.", "en")], out
    # contíguo (lacuna ~0) continua juntando
    b.push("We lost by a", "en", True, 10.0, 17.0)
    out = b.push("point in a half.", "en", True, 17.05, 20.0)
    assert out == [("We lost by a point in a half.", "en")], out


def test_minuscula_nao_corrompe_sigla_nem_artigo() -> None:
    def junta(tail: str, new: str) -> str:
        b = SentenceBuffer()
        b.push(tail, "en", True)
        return b.push(new, "en", False)[0][0]

    assert junta("We are talking about", "A.I. stocks.") ==         "We are talking about A.I. stocks."
    assert junta("We are buying the", "IT stocks.") == "We are buying the IT stocks."
    assert junta("This is such an", "A plus setup.") == "This is such an A plus setup."
    # cauda terminada em abreviação ("U.S."): não minusculiza
    assert junta("Data from the U.S.", "The economy is fine.") ==         "Data from the U.S. The economy is fine."
    # palavra funcional Title-case normal continua minusculizada
    assert junta("Up on a", "Month over month basis.") ==         "Up on a month over month basis."


def test_teto_padrao_e_20() -> None:
    b = SentenceBuffer()
    assert b.max_pending_words == 20
    assert b.push(" ".join(["w"] * 20), "en", True) == []
    assert len(b.push(" ".join(["w"] * 21), "en", True)) == 1


def test_teto_8_palavras_emite_9_e_segura_8() -> None:
    b = SentenceBuffer(max_pending_words=8)
    out = b.push(" ".join(["w"] * 9), "en", True)
    assert out == [(" ".join(["w"] * 9), "en")], out
    assert b.pending() == ""
    b = SentenceBuffer(max_pending_words=8)
    assert b.push(" ".join(["w"] * 8), "en", True) == []
    assert b.pending() == " ".join(["w"] * 8)


def test_config_max_palavras_espera() -> None:
    import json
    import tempfile
    from tradutor.config import AppConfig
    assert AppConfig().max_palavras_espera == 8
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "config.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"max_palavras_espera": 20}, f)
        assert AppConfig.load(p).max_palavras_espera == 20
        assert AppConfig.load(os.path.join(d, "nao_existe.json")).max_palavras_espera == 8


def test_espacos_normalizados() -> None:
    b = SentenceBuffer()
    assert b.push("  Up   we   go.  ", "en", False) == [("Up we go.", "en")]


TESTS = [
    test_segura_cauda_e_junta,
    test_forced_cut_emite_completo_e_segura_resto,
    test_pausa_real_emite_tudo,
    test_reticencias_finais_removidas_na_juncao,
    test_minuscula_so_para_palavra_funcional,
    test_abreviacoes_nao_dividem,
    test_reticencias_no_meio_nao_dividem,
    test_decimal_nao_divide,
    test_teto_de_palavras_pendentes,
    test_flush_due_com_relogio_falso,
    test_troca_de_idioma_libera_a_cauda_antiga,
    test_reset_descarta,
    test_flush_emite_pendente_e_nunca_vazio,
    test_lacuna_grande_libera_a_cauda_antiga,
    test_minuscula_nao_corrompe_sigla_nem_artigo,
    test_teto_padrao_e_20,
    test_teto_8_palavras_emite_9_e_segura_8,
    test_config_max_palavras_espera,
    test_espacos_normalizados,
]


def main() -> int:
    ok = failed = 0
    for fn in TESTS:
        name = fn.__name__
        t0 = time.monotonic()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
        else:
            ok += 1
            print(f"[ OK ] {name} ({time.monotonic() - t0:.2f}s)")
    print(f"\n{ok} passaram, {failed} falharam")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
