"""Testes standalone do módulo `tradutor.glossary` (blindagem/tradução fixa).

Executar:  .venv\\Scripts\\python.exe tests\\test_glossary.py

Não depende de pytest. Usa o `glossario.json` real da raiz do projeto (para
pegar regressões introduzidas por edição do arquivo) e uma `translate_fn`
identidade, que devolve o texto já mascarado, sem passar por um MT de
verdade, então os asserts checam o texto MASCARADO ou o `unmask`+`fix` final,
conforme os casos observados no `traducoes.log`.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from tradutor.glossary import Glossary, _fix_unk  # noqa: E402

try:  # acentos no console do Windows
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

_GLOSSARIO_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "glossario.json")


def _identity(text: str) -> str:
    """`translate_fn` que não traduz nada, deixa ver o texto mascarado."""
    return text


class Skip(Exception):
    pass


def _new_glossary() -> Glossary:
    return Glossary(_GLOSSARIO_PATH)


def _pt(gl: Glossary, en: str) -> str:
    """Aplica mask -> identity -> unmask -> fix, como `apply()` faria."""
    return gl.apply(_identity, en)


def _masked(gl: Glossary, en: str) -> str:
    masked, _found = gl.mask(en)
    return masked


# --------------------------------------------------------------------------
def test_long_short_lookaround() -> None:
    gl = _new_glossary()
    # 1. "my long time discomfort" não pode mascarar "long"
    masked = _masked(gl, "my long time discomfort")
    assert "XPROTECTED" not in masked, masked
    pt = _pt(gl, "I'm long here")
    assert "comprado" in pt, pt
    # 2. "how long the market stays" não pode mascarar nada de long/short
    masked2 = _masked(gl, "how long the market stays")
    assert "XPROTECTED" not in masked2, masked2


def test_long_short_protegido_vs_duracao() -> None:
    # B3: "short squeeze" é protegido mesmo depois de "that" (frase inteira
    # tem sentido financeiro completo, não "that" + duração de "short").
    gl = _new_glossary()
    masked, found = gl.mask("that short squeeze")
    assert "short squeeze" in found, found
    out = gl.unmask(masked, found)
    assert "short squeeze" in out, out
    # "it took that long" - duração, não posição - não mascara nada
    masked2, found2 = gl.mask("it took that long")
    assert found2 == [], found2
    # "this long red candle" - "long" descrevendo o candle, não posição
    masked3, found3 = gl.mask("this long red candle")
    assert found3 == [], found3
    # sentido financeiro completo em fim de frase/pontuação continua válido
    pt = _pt(gl, "I'm long.")
    assert "comprado" in pt.lower(), pt
    pt2 = _pt(gl, "who is short?")
    assert "vendido" in pt2.lower(), pt2


def test_hilo_determinante() -> None:
    gl = _new_glossary()
    pt = _pt(gl, "the low here at 196")
    assert "fundo" in pt, pt
    # "setup" é protegido (fica em inglês) - "high" não pode ser mascarado
    masked, found = gl.mask("a high probability setup")
    assert "topo" not in found, found
    assert " high " in masked, masked
    masked2, found2 = gl.mask("the high side")
    assert "topo" not in found2, found2
    assert " high " in masked2, masked2


def test_down_up_lookahead() -> None:
    gl = _new_glossary()
    pt = _pt(gl, "I'm down to 2 micros")
    assert "no negativo" not in pt, pt
    pt2 = _pt(gl, "I'm down 200 bucks")
    assert "no negativo" in pt2, pt2


def test_free_trade() -> None:
    gl = _new_glossary()
    pt = _pt(gl, "free trade")
    assert "trade sem risco" in pt.lower(), pt
    pt2 = _pt(gl, "free trade agreement")
    assert "acordo de livre comércio" in pt2.lower(), pt2


def test_trade_verbo_conjugado() -> None:
    gl = _new_glossary()
    pt = _pt(gl, "to trade SPY")
    assert "operar" in pt.lower(), pt
    assert "SPY" in pt, pt
    assert "trocar" not in pt.lower(), pt


def test_stops_break_even() -> None:
    gl = _new_glossary()
    pt = _pt(gl, "the stops are all at break even")
    assert "stops" in pt, pt
    assert "break even" in pt, pt


def test_fix_anos_fantasma() -> None:
    gl = _new_glossary()
    # source tem "years" de verdade -> mantém "25 anos"
    out = gl.fix("Conheço Kevin há 25 anos",
                  source="I've known Kevin for 25 years")
    assert "25 anos" in out, out
    # source não tem "year" -> número era preço/idade fantasma, remove "anos".
    # Forma escolhida: "no N" (consistente com a regra já existente
    # `\baos (\d+) anos\b -> no \1`, que roda antes da nova regra "aos N solto
    # -> em N" e já consome o "aos ... anos" inteiro).
    out2 = gl.fix("Gostei aos 50 anos", source="I liked it at 50")
    assert out2 == "Gostei no 50", out2
    # a nova regra "aos N solto (sem anos) -> em N" cobre o caso em que o
    # "anos" já foi removido por outro motivo (ou nunca existiu na frase)
    out3 = gl.fix("Gostei aos 50", source="I liked it at 50")
    assert out3 == "Gostei em 50", out3


def test_fix_unk_token() -> None:
    gl = _new_glossary()
    out = gl.fix("⁇ s vezes o mercado")
    assert out == "Às vezes o mercado", out
    out2 = gl.fix("⁇ timo")
    assert out2 == "Ótimo", out2
    assert "⁇" not in out and "⁇" not in out2


def test_fix_unk_direct() -> None:
    # _fix_unk isolado (sem passar por fix(), que capitaliza/mexe em mais)
    assert "⁇" not in _fix_unk("um teste ⁇ qualquer")


def test_unmask_residuos() -> None:
    gl = _new_glossary()
    out = gl.unmask("Esta é a hora XPROTECTEDED0X", ["10 da manhã"])
    assert out == "Esta é a hora 10 da manhã", out
    out2 = gl.unmask("com o XPROTECTED0XX2", ["S&P 500"])
    assert out2 == "com o S&P 500", out2
    out3 = gl.unmask("garante que é apenas ED0X.", [])
    assert "ED0X" not in out3, out3
    out4 = gl.unmask("MFF é 900x", [])
    assert out4 == "MFF é 900", out4


def test_unmask_nao_apaga_alavancagem() -> None:
    # B2: prefixo "ED"/"D" é OBRIGATÓRIO no resíduo removido - sem isso a
    # regex apagava alavancagem legítima ("3X ETF", "5X leverage").
    gl = _new_glossary()
    out = gl.unmask("um 3X ETF", [])
    assert out == "um 3X ETF", out
    out2 = gl.unmask("5X leverage", [])
    assert out2 == "5X leverage", out2
    out3 = gl.unmask("é apenas ED0X.", [])
    assert "ED0X" not in out3, out3


def test_concordancia_generos() -> None:
    gl = _new_glossary()
    assert gl.fix("uma quebra falso") == "uma quebra falsa"
    assert gl.fix("Cancele todos os ordens") == "Cancele todas as ordens"
    out = gl.fix("um execução acima de 200")
    assert out == "uma execução acima de 200", out
    out2 = gl.fix("manter esta posição comprador")
    assert "posição compradora" in out2, out2


def test_concordancia_plural_adjetivo() -> None:
    # N1: adjetivo tem que pluralizar junto com o substantivo restaurado.
    gl = _new_glossary()
    assert gl.fix("falso quebras") == "falsas quebras"
    assert gl.fix("as quedas pequeno") == "as quedas pequenas"
    assert gl.fix("as ordens executado") == "as ordens executadas"


def test_sexta_feira_e_quando() -> None:
    gl = _new_glossary()
    text = "Sexta-feira é quando temos"
    assert gl.fix(text) == text


def test_gerundio_pt_pt() -> None:
    gl = _new_glossary()
    out = gl.fix("o mercado está a subir")
    assert "está subindo" in out, out


def test_biggest_trade_mascarado() -> None:
    gl = _new_glossary()
    masked = _masked(gl, "the biggest trade ever")
    assert "XPROTECTED" in masked, masked
    pt = _pt(gl, "the biggest trade ever")
    assert "comércio" not in pt.lower(), pt


def test_ripped_nao_vira_mandado_ver() -> None:
    gl = _new_glossary()
    pt = _pt(gl, "he's really ripped")
    assert "mandado ver" not in pt, pt


def test_move_play_nao_se_confundem() -> None:
    gl = _new_glossary()
    pt = _pt(gl, "the move here")
    assert "o movimento" in pt.lower(), pt
    pt2 = _pt(gl, "this play")
    assert "essa jogada" in pt2.lower(), pt2
    assert "jogada" in pt2.lower(), pt2
    assert "movimento" not in pt2.lower(), pt2


def test_concordancia_nao_mutila_hifenizado() -> None:
    # R3: "o novo quebra-cabeça" não pode virar "o nova quebra-cabeça".
    gl = _new_glossary()
    assert gl.fix("o novo quebra-cabeça") == "o novo quebra-cabeça"


def test_fix_re_precompilado() -> None:
    # B1: performance - _fix_re precisa existir e ter uma entrada compilada
    # por chave de `corrigir` (evita recompilar regex a cada chamada de fix).
    gl = _new_glossary()
    assert hasattr(gl, "_fix_re")
    assert len(gl._fix_re) == len(gl._fix)
    import re as _re
    assert all(hasattr(regex, "search") for regex, _ in gl._fix_re[:5])


def test_glossario_json_valido() -> None:
    import json
    with open(_GLOSSARIO_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "proteger" in data and "traduzir" in data and "corrigir" in data


def test_glossary_carrega_sem_excecao() -> None:
    gl = _new_glossary()
    assert len(gl._fix) > 0
    assert len(gl._protect_re) > 0


def test_corrigir_sem_entradas_mortas() -> None:
    """N3: nenhuma chave curta pode vir antes de uma chave mais longa que a
    contém como palavra inteira (a curta dispara primeiro e mata a longa)."""
    import json
    import re as _re
    with open(_GLOSSARIO_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    keys = list(data["corrigir"].keys())

    def contains_whole(big: str, small: str) -> bool:
        pattern = r"(?<![\w])" + _re.escape(small) + r"(?![\w])"
        return _re.search(pattern, big, _re.IGNORECASE) is not None

    dead = []
    for i, long_k in enumerate(keys):
        for j in range(i):
            short_k = keys[j]
            if short_k == long_k:
                continue
            if len(short_k) < len(long_k) and contains_whole(long_k, short_k):
                dead.append((short_k, long_k))
    assert dead == [], dead


TESTS = [
    test_long_short_lookaround,
    test_long_short_protegido_vs_duracao,
    test_hilo_determinante,
    test_down_up_lookahead,
    test_free_trade,
    test_trade_verbo_conjugado,
    test_stops_break_even,
    test_fix_anos_fantasma,
    test_fix_unk_token,
    test_fix_unk_direct,
    test_unmask_residuos,
    test_unmask_nao_apaga_alavancagem,
    test_concordancia_generos,
    test_concordancia_plural_adjetivo,
    test_concordancia_nao_mutila_hifenizado,
    test_fix_re_precompilado,
    test_sexta_feira_e_quando,
    test_gerundio_pt_pt,
    test_biggest_trade_mascarado,
    test_ripped_nao_vira_mandado_ver,
    test_move_play_nao_se_confundem,
    test_glossario_json_valido,
    test_glossary_carrega_sem_excecao,
    test_corrigir_sem_entradas_mortas,
]


def main() -> int:
    import time
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
