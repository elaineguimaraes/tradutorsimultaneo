"""Re-junção de frases cortadas antes da tradução.

Quem fala ao vivo raramente pausa o bastante para o VAD fechar o segmento
por silêncio: a maioria sai cortada por `max_segment_s`, no meio da frase
("We're expecting up three tenths our last look" | "was up 3 tenths."). O
Opus-MT traduz frase a frase, sem contexto, e inventa um final para a
meia-frase. Este módulo segura a cauda sem pontuação final e a cola ao
início do segmento seguinte, de modo que o MT receba frases inteiras.

Lógica pura (sem imports do app): o relógio é injetável para teste.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Callable, List, Optional, Tuple

Item = Tuple[str, str]   # (texto, idioma)

# Whisper capitaliza todo começo de trecho ("Month over month basis"); o MT
# lê maiúscula como nome próprio. Só palavras funcionais comuns são
# minusculizadas na junção (nunca "I", "I'm"... nem nomes como "Nvidia").
_CONTINUATION = frozenset("""
the a an and but so or to of in on at for with from by as that this these
those it its is are was were be been we you they he she there which what when
where who how if because then than up down about into over under month year
week day more less not no than their our your his her them out off per
""".split())

_ABBREV = frozenset({
    "mr", "mrs", "ms", "dr", "st", "jr", "sr", "vs", "etc", "inc", "corp",
    "ltd", "co", "e.g", "i.e", "a.m", "p.m",
})

_TRAIL_ELLIPSIS_RE = re.compile(r"\s*(?:\.{2,}|…)+\s*$")
_WS_RE = re.compile(r"\s+")
# fim de frase: . ! ? (nunca o último ponto de reticências), com aspas/
# parênteses de fechamento opcionais, seguido de espaço ou fim do texto
_BOUNDARY_RE = re.compile(r"(?<!\.)[.!?][\"')\]]*(?=\s|$)")
_INITIALISM_RE = re.compile(r"^(?:[A-Za-z]\.){2,}$")


def _norm(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _is_abbrev(joined: str, dot_idx: int) -> bool:
    """True se o ponto em `dot_idx` fecha uma abreviação ou sigla pontuada."""
    start = dot_idx
    while start > 0 and not joined[start - 1].isspace():
        start -= 1
    token = joined[start:dot_idx + 1].lstrip("\"'([")
    if _INITIALISM_RE.match(token):
        return True
    return token[:-1].lower() in _ABBREV


def _split_last_boundary(text: str) -> Tuple[str, str]:
    """Divide em (completo, resto) na ÚLTIMA fronteira de frase real."""
    cut = 0
    for m in _BOUNDARY_RE.finditer(text):
        if text[m.start()] == "." and _is_abbrev(text, m.start()):
            continue
        cut = m.end()
    return text[:cut], text[cut:]


def _lower_first_if_function_word(text: str, tail: str = "") -> str:
    """Minusculiza a 1ª palavra se for funcional e escrita em Title-case.

    Nunca mexe em palavra de uma letra ("A plus setup", "I"), em sigla
    ("A.I.", "IT stocks": a palavra não é Title-case ou vem seguida de ponto)
    nem quando a cauda termina em ponto de abreviação ("U.S.", "a.m.").
    """
    if tail.rstrip().endswith("."):
        return text
    m = re.match(r"\W*([A-Za-z']+)", text)
    if not m:
        return text
    w = m.group(1)
    if len(w) < 2 or not (w[0].isupper() and w[1:].islower()):
        return text
    if text[m.end(1):m.end(1) + 1] == ".":
        return text
    if w.lower() in _CONTINUATION:
        i = m.start(1)
        return text[:i] + text[i].lower() + text[i + 1:]
    return text


class SentenceBuffer:
    """Segura a cauda incompleta de um segmento e a une ao seguinte."""

    def __init__(self, max_pending_words: int = 20, max_hold_s: float = 10.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.max_pending_words = int(max_pending_words)
        self.max_hold_s = float(max_hold_s)
        self._clock = clock
        self._lock = threading.Lock()   # reset() vem de outra thread
        self._tail = ""
        self._lang = ""
        self._since = 0.0
        self._tail_end: Optional[float] = None   # t_end do último trecho da cauda

    # ------------------------------------------------------------------ API
    # lacuna (s) entre o fim da cauda e o início do próximo trecho acima da
    # qual não há continuação (o ASR descartou algo no meio, pausa de verdade)
    _MAX_GAP_S = 1.0

    def push(self, text: str, lang: str, forced_cut: bool,
             t_start: Optional[float] = None,
             t_end: Optional[float] = None) -> List[Item]:
        """Recebe um transcript; devolve os itens prontos para traduzir."""
        out: List[Item] = []
        with self._lock:
            gap = (t_start is not None and self._tail_end is not None
                   and t_start - self._tail_end > self._MAX_GAP_S)
            if self._tail and (lang != self._lang or gap):
                out.append((self._tail, self._lang))
                self._tail = ""

            new = _norm(text)
            if self._tail:
                head = _TRAIL_ELLIPSIS_RE.sub("", self._tail)
                joined = _norm(head + " " + _lower_first_if_function_word(new, head))
            else:
                joined = new
            had_tail = bool(self._tail)
            self._tail = ""
            if not joined:
                return out

            if not forced_cut:
                # pausa real do falante: fim de frase mesmo sem pontuação
                out.append((joined, lang))
                return out

            complete, rest = _split_last_boundary(joined)
            complete, rest = complete.strip(), rest.strip()
            if complete:
                out.append((complete, lang))
            if rest:
                if len(rest.split()) > self.max_pending_words:
                    out.append((rest, lang))
                else:
                    if not had_tail or complete:
                        self._since = self._clock()
                    self._tail = rest
                    self._lang = lang
                    self._tail_end = t_end
        return out

    def flush_due(self) -> List[Item]:
        """Libera a cauda se ela esperou além de `max_hold_s`."""
        with self._lock:
            if self._tail and self._clock() - self._since > self.max_hold_s:
                return self._take()
        return []

    def flush(self) -> List[Item]:
        """Libera o que estiver pendente."""
        with self._lock:
            return self._take()

    def reset(self) -> None:
        """Descarta a cauda sem emitir (parada/reinício)."""
        with self._lock:
            self._tail = ""

    def pending(self) -> str:
        with self._lock:
            return self._tail

    def _take(self) -> List[Item]:
        if not self._tail:
            return []
        item = (self._tail, self._lang)
        self._tail = ""
        return [item]
