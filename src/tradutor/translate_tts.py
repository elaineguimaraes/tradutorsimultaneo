"""Tradução para pt-BR (Opus-MT via CTranslate2 / Argos Translate) e voz via edge-tts.

Implementa `TranslatorProtocol` e `TtsSpeakerProtocol` de `contracts.py`.

Duas classes públicas:
    Translator: tradução SÍNCRONA, offline, nunca lança exceção.
    TtsSpeaker: síntese SÍNCRONA por fora, asyncio numa thread dedicada por dentro.

Escolha de backend de tradução em runtime:
    1) Opus-MT tc-big en->pt-BR local em `models/opus-mt-en-pt-ct2/` via
       CTranslate2 (gerado por `scripts/preparar_modelos.py`; produz pt-BR
       nativo com o marcador `>>pob<<`, é o caminho oficial de instalação);
    2) argostranslate: reserva automática quando o modelo local não existe;
    3) ctranslate2 + huggingface_hub convertendo o modelo na hora, último
       recurso, usado se o argos não importar/instalar.
O backend ativo é logado em `tradutor.mt` na inicialização.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import queue
import re
import threading
import time
from typing import Optional

import numpy as np

try:  # uso normal como pacote
    from .contracts import SpeechSegment, Transcript, Translation, TtsAudio
    from .segmenter import collapse_repetitions
except ImportError:  # execução direta / testes standalone
    from contracts import SpeechSegment, Transcript, Translation, TtsAudio  # type: ignore
    from segmenter import collapse_repetitions  # type: ignore

log_mt = logging.getLogger("tradutor.mt")
log_tts = logging.getLogger("tradutor.tts")

# Raiz do projeto (…/src/tradutor/translate_tts.py -> …/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")

# Limites de texto do TTS (proteção contra segmentos absurdos do ASR).
TTS_SOFT_LIMIT = 400
TTS_HARD_LIMIT = 2000

_WS_RE = re.compile(r"[ \t]{2,}")
_SENT_RE = re.compile(r"(?<=[.!?;:])\s+")


_DOUBLED_WORD_RE = re.compile(
    r"(?i)^(\W*)(\w+)[,;]?\s+\2(\W*)$")


# Conectores onde uma frase longa sem pontuação pode ser partida (2ª opção,
# depois da vírgula). Sem eles o Marian recebe 30 palavras de uma vez e cai
# em loop de paráfrase.
_SOFT_CONNECTORS = frozenset(
    {"so", "but", "because", "if", "which", "when", "while", "and"})
_SOFT_MIN_WORDS = 4   # nunca gera pedaço menor que isto


def _soft_split(words: "list[str]", max_words: int) -> "list[list[str]]":
    """Parte a lista de palavras perto do meio, vírgula antes de conector."""
    n = len(words)
    if n <= max_words:
        return [words]
    for tier in (0, 1):
        best = None
        for k in range(_SOFT_MIN_WORDS, n - _SOFT_MIN_WORDS + 1):
            if tier == 0:
                ok = words[k - 1].endswith(",")
            else:
                ok = words[k].lower().strip(",.;:!?") in _SOFT_CONNECTORS
            if ok and (best is None or abs(k - n / 2) < abs(best - n / 2)):
                best = k
        if best is not None:
            return (_soft_split(words[:best], max_words)
                    + _soft_split(words[best:], max_words))
    return [words]   # sem fronteira que renda 2 pedaços de >= 4 palavras


def _chunk_long(sent: str, max_words: int = 22) -> "list[str]":
    """Frase longa (ASR sem pontuação) faz o Marian ecoar; divide em pedaços.

    Devolve os pedaços sem a vírgula final (o MT a trocaria por ponto); quem
    reagrupa a saída sabe que todos, menos o último, são continuação.
    """
    pieces = _soft_split(sent.split(), max_words)
    if len(pieces) == 1:
        return [sent]
    return [" ".join(p).rstrip(",;") for p in pieces]


_I_START_RE = re.compile(r"I(?:'(?:m|ve|ll|d))?\b")


def _join_pieces(outs: "list[str]", srcs: "list[str]", soft: "list[bool]") -> str:
    """Reagrupa a saída: pedaço de continuação termina em vírgula, não ponto."""
    res = []
    prev_cont = False
    for out, src, cont in zip(outs, srcs, soft):
        out = out.strip()
        if not out:
            continue
        # origem em minúscula no meio da frase = não é nome próprio: o MT
        # capitalizou por achar que abria frase
        # (EUA/SPY ficam: 2ª letra maiúscula; "E"/"É" de uma letra descem)
        if (prev_cont and (src[:1].islower() or _I_START_RE.match(src))
                and out[0].isupper() and (len(out) == 1 or not out[1].isupper())):
            out = out[0].lower() + out[1:]
        if cont:
            out = out.rstrip(" .,;!?") + ","
        prev_cont = cont
        res.append(out)
    # último pedaço traduziu para vazio: não deixa a vírgula pendurada
    if res and outs and not outs[-1].strip():
        res[-1] = res[-1].rstrip(",")
    return " ".join(res)


_ADJ_DUP_RE = re.compile(r"(?i)\b(\w+)[,;]?\s+\1\b")
_WORD_RE = re.compile(r"\w+")
_PT_TRAIL_CONN = frozenset(
    "e ou mas que de do da dos das em no na a o os as um uma com para por se "
    "como então".split())


_EN_STOP = frozenset(
    "the a an and or but so to of in on at for with from by as that this "
    "these those it its is are was were be been we you they he she i there "
    "which what when where who how if because then than not no up down "
    "out off do did does have has had".split())


# gagueira do ASR que não conta como repetição: só artigos/preposições
# ("No, no" é ênfase legítima e vira "Não, não")
_EN_STUTTER = frozenset("the a an to of in on at".split())


def _src_has_adjacent_dup(en: str) -> bool:
    """Repetição de palavra no inglês; gagueira de artigo/preposição não conta."""
    return any(m.group(1).lower() not in _EN_STUTTER
               for m in _ADJ_DUP_RE.finditer(en))


def _src_repeats_content_word(en: str) -> bool:
    """O inglês repete alguma palavra de conteúdo (>= 3 letras, não stopword)?

    Frase com ênfase/paralelismo ("above average ... above average", "if price
    goes... if price falls") vira, em português, um trecho de 3+ palavras que
    se repete legitimamente; nesse caso a poda de frase não pode agir.
    """
    seen = set()
    for w in (x.lower() for x in _WORD_RE.findall(en)):
        if len(w) < 3 or w in _EN_STOP:
            continue
        if w in seen:
            return True
        seen.add(w)
    return False


def _src_has_repeated_3gram(en: str) -> bool:
    w = [x.lower() for x in _WORD_RE.findall(en)]
    grams = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    return len(grams) != len(set(grams))


def _first_repeated_span(pt: str) -> "int | None":
    """Posição (char) do início da 2ª ocorrência do maior trecho repetido (>= 3 palavras)."""
    ms = list(_WORD_RE.finditer(pt))
    keys = [m.group().lower() for m in ms]
    for n in range(len(keys) // 2, 2, -1):
        for a in range(len(keys) - 2 * n + 1):
            span = keys[a:a + n]
            if not any(len(w) >= 3 for w in span):
                continue
            for b in range(a + n, len(keys) - n + 1):
                if keys[b:b + n] == span:
                    return ms[b].start()
    return None


def _dedupe_echo(pt: str, en: str) -> "tuple[str, bool]":
    """Remove eco do MT que NÃO existe no inglês de origem (por pedaço).

    O Marian repete palavra ("Nike Nike") e trecho ("...da Piper, você tem
    que ficar na camisa da Piper") sem que o falante tenha dito isso. Se o
    inglês repete (ênfase real: "really, really"), a tradução é preservada.
    """
    out = pt
    if not _src_has_repeated_3gram(en) and not _src_repeats_content_word(en):
        pos = _first_repeated_span(out)
        if pos is not None:
            cut = out[:pos].rstrip(" ,;:-")
            if "," in cut:
                head, _, tail = cut.rpartition(",")
                if head.strip() and len(tail.split()) <= 3:
                    cut = head
            words = cut.split()
            while len(words) > 3 and words[-1].lower().strip(",;") in _PT_TRAIL_CONN:
                words.pop()
            cut = " ".join(words).rstrip(" ,;:-")
            if cut:
                end = out.rstrip()[-1:]
                out = cut + (end if end in ".!?" else "")
    if not _src_has_adjacent_dup(en):
        prev = None
        while prev != out:
            prev = out
            out = _ADJ_DUP_RE.sub(r"\1", out)
    if not _src_repeats_content_word(en):
        # "assine e assine" (EN "just sign up"): palavra repetida com conector.
        # O inglês de "mais e mais"/"dia após dia"/"passo a passo" repete a
        # palavra (ou a liga por and/by/after/to), por isso o guarda.
        if not _EN_LINKED_DUP_RE.search(en):
            out = _CONN_DUP_RE.sub(_keep_first, out)
        # "aqui mesmo aqui" no fim da frase: 2ª ocorrência é eco
        if not _EN_LINKED_DUP_RE.search(en):
            out = _TAIL_DUP_RE.sub(_keep_before_tail, out)
    out = _fix_number_echo(out, en)
    return out, out != pt


_CONN_DUP_RE = re.compile(r"(?i)\b(\w{3,})\s+(?:e|ou)\s+\1\b")
_TAIL_DUP_RE = re.compile(r"(?i)\b(\w{3,})((?:[ ,]+\w+){1,2})[ ,]+\1\b(?=\W*$)")
_EN_LINKED_DUP_RE = re.compile(
    r"(?i)\b(\w+)\s+(?:and|or|by|to|after|on|over|upon|in|for)\s+\1\b")
_PT_STOP = frozenset(
    "que com para por uma uns umas dos das nos nas não sim mais como mas "
    "seu sua isso esse essa este esta ele ela são foi ser ter tem".split())


_PT_LINK = frozenset({"e", "ou", "a", "por", "após", "apos", "para", "de"})


def _keep_first(m: "re.Match") -> str:
    return m.group(1)


def _keep_before_tail(m: "re.Match") -> str:
    """Descarta a repetição final, exceto se a palavra for função em português."""
    if m.group(1).lower() in _PT_STOP:
        return m.group(0)
    # "subindo e subindo", "lado a lado": o miolo é só um conector, forma da regra (a)
    if m.group(2).replace(",", " ").strip().lower() in _PT_LINK:
        return m.group(0)
    return m.group(1) + m.group(2)


_NUM_RE = re.compile(r"\d[\d.,]*\d|\d")


def _fix_number_echo(pt: str, en: str) -> str:
    """"400" -> "400.400": número do MT que é o do inglês concatenado consigo."""
    # os tokens XPROTECTEDnX do inglês mascarado não são números
    nums = set(_NUM_RE.findall(re.sub(r"(?i)XPROTECTED\d+X", " ", en)))
    if not nums:
        return pt

    def _sub(m: "re.Match") -> str:
        p = m.group(0)
        # o MT localiza "1.1" -> "1,1": a forma com o outro separador é a mesma
        if p in nums or p.replace(",", ".") in nums or p.replace(".", ",") in nums:
            return p
        for n in nums:
            if len(n) >= 2 and len(p) > len(n) and any(p == n + sep + n for sep in ("", ".", ",")):
                return n
        return p

    return _NUM_RE.sub(_sub, pt)


def _dedupe_adjacent_sentences(text: str) -> "tuple[str, bool]":
    """Remove frases idênticas consecutivas na saída do MT ("Pergunta. Pergunta.")."""
    sents = [s for s in _SENT_RE.split(text) if s.strip()]
    if len(sents) < 2:
        return text, False
    out = [sents[0]]
    cut = False
    for s in sents[1:]:
        a = re.sub(r"\W+", "", s).lower()
        b = re.sub(r"\W+", "", out[-1]).lower()
        # igual OU uma contida na outra ("Não entendo. Eu não entendo.")
        if a == b or (min(len(a), len(b)) >= 8 and (a in b or b in a)):
            cut = True
            if len(a) > len(b):
                out[-1] = s   # fica a versão mais completa
            continue
        out.append(s)
    return " ".join(out), cut

# Muletas de fala que o ASR transcreve ("uh", "um"…). Removidas ANTES da
# tradução: o Opus-MT entra em loop de decodificação com elas ("uh, uhm,
# uhu…" até estourar o limite) e o Argos as deixava passar cruas.
_FILLER_RE = re.compile(
    r"(?i)(?:^|[,.]?\s+)(?:uh+|um+|uh-huh|mm-hmm|hmm+|erm+)(?=[,.!?\s]|$)")


# Muletas de discurso e conectores pendurados no fim do segmento: gatilho
# clássico de eco do MT. Removidos SEMPRE (mesmo com pontuação final,
# "…sizable range I mean." não perde sentido sem eles).
_TRAIL_CONN_RE = re.compile(
    r"(?i)(?:[\s,]+(?:i mean|you know|so|and|but|because|or|of|to|in|at|on|with|for|the|a|an|whether|if))+[\s,.!?]*$")


# Preposições/pronomes que só são "pendurados" quando o segmento foi CORTADO
# no meio ("…just running into", "…look at its"). Com pontuação final a frase
# está completa ("The market is up.") e cortar mutilaria o sentido, por isso
# este grupo só age em texto SEM pontuação no fim.
_TRAIL_CUT_RE = re.compile(
    r"(?i)[\s,]+(?:into|onto|from|by|about|like|than|over|under|through"
    r"|toward|towards|up|down|out|off|as|that|it's|its|it|this|which)$")


def _strip_trailing_junk(text: str) -> str:
    """Limpa a cauda pendurada do segmento até estabilizar.

    "…just running into a" perde o "a" e depois o "into": um passo só
    deixaria a preposição no fim, que é justamente o gatilho do eco.
    """
    for _ in range(4):
        before = text
        text = _TRAIL_CONN_RE.sub("", text) or before
        if not text.rstrip().endswith((".", "!", "?")):
            text = _TRAIL_CUT_RE.sub("", text) or text
        if text == before:
            break
    return text


_STOP_SEGMENTS = {"the", "and", "a", "an", "of", "to", "in", "at",
                  "on", "or", "but", "so", "uh", "um"}


def _strip_fillers(text: str) -> str:
    text = _FILLER_RE.sub("", text)
    return _WS_RE.sub(" ", text).strip(" ,")


# ==========================================================================
# Utilidades de texto
# ==========================================================================

def _polish(text: str) -> str:
    """Pós-processamento leve: colapsa espaços e garante maiúscula inicial."""
    out = _WS_RE.sub(" ", text.replace("\r", " ").replace("\n", " ")).strip()
    if not out:
        return ""
    # Espaço antes de pontuação é ruído comum de MT.
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    if out[0].islower():
        out = out[0].upper() + out[1:]
    return out


# ==========================================================================
# Translator
# ==========================================================================

class Translator:
    """Traduz para pt-BR usando Opus-MT/CT2 (ou Argos Translate como reserva).

    Nunca propaga exceção: em qualquer falha devolve o texto original, para a
    GUI mostrar o cru em vez de quebrar o pipeline.
    """

    def __init__(self, target: str = "pt", auto_install: bool = True) -> None:
        self.target = target
        self.auto_install = auto_install     # baixa pacote de idioma novo em 2º plano
        self.backend: str = "none"          # "argos" | "ct2" | "none"
        self._ready_langs: set[str] = set()  # idiomas com caminho de tradução ok
        self._failed_langs: set[str] = set()  # idiomas sem pacote (não retentar)
        self._pending: set[str] = set()      # instalações em andamento
        self._lock = threading.Lock()
        self._argos = None                   # módulos (package, translate)
        self._ct2 = None                     # dict com translator/sp/…
        self._select_backend()

    # ---------------------------------------------------------------- setup

    def _select_backend(self) -> None:
        """Decide o backend disponível neste interpretador (uma vez)."""
        # Opus-MT tc-big local (models/) tem prioridade sobre o argos: gera
        # pt-BR nativo (>>pob<<) e traduz bem melhor. Só entra se o modelo já
        # foi baixado/convertido; se `models/` não existe (instalador não
        # rodou ou falhou), cai para o argos.
        try:
            import ctranslate2  # noqa: F401
            import sentencepiece  # noqa: F401
            local = os.path.join(MODELS_DIR, "opus-mt-en-pt-ct2", "model.bin")
            if os.path.exists(local):
                self.backend = "ct2"
                log_mt.info("backend de tradução ativo: ctranslate2 "
                            "(Opus-MT tc-big en->pt-BR local)")
                return
        except Exception:
            pass
        try:
            from argostranslate import package as argos_package
            from argostranslate import translate as argos_translate
            self._argos = (argos_package, argos_translate)
            self.backend = "argos"
            log_mt.info("backend de tradução ativo: argostranslate")
            return
        except Exception:
            log_mt.warning("argostranslate indisponível; tentando ctranslate2 + Opus-MT",
                           exc_info=True)
        try:
            import ctranslate2  # noqa: F401
            import sentencepiece  # noqa: F401
            self.backend = "ct2"
            log_mt.info("backend de tradução ativo: ctranslate2 (Opus-MT)")
        except Exception:
            self.backend = "none"
            log_mt.error("nenhum backend de tradução disponível, textos seguirão sem tradução",
                         exc_info=True)

    def ensure_ready(self, langs: list[str]) -> None:
        """Baixa/instala o necessário para traduzir `langs` -> pt (pode demorar)."""
        with self._lock:
            for lang in langs:
                lang = (lang or "").split("-")[0].lower()
                if not lang or lang == self.target or lang in self._ready_langs:
                    continue
                t0 = time.monotonic()
                try:
                    if self.backend == "argos":
                        ok = self._argos_ensure(lang)
                    elif self.backend == "ct2":
                        ok = self._ct2_ensure(lang)
                    else:
                        ok = False
                except Exception:
                    log_mt.exception("falha ao preparar tradução %s->%s", lang, self.target)
                    ok = False
                dt = time.monotonic() - t0
                if ok:
                    self._ready_langs.add(lang)
                    self._failed_langs.discard(lang)
                    log_mt.info("pacote %s->%s pronto em %.1fs", lang, self.target, dt)
                else:
                    self._failed_langs.add(lang)
                    log_mt.warning("sem pacote %s->%s (%.1fs), texto seguirá original",
                                   lang, self.target, dt)

    def prepare_local_model(self, lang: str = "en") -> bool:
        """Garante o Opus-MT `lang`->pt convertido em `models/` e ativa o backend ct2.

        Usado pelo instalador (`scripts/preparar_modelos.py`). Baixa o modelo do
        Hugging Face e converte para CTranslate2 int8 uma única vez (alguns
        minutos); nas execuções seguintes só confirma que o cache existe.
        Devolve True se o backend ct2 ficou ativo.
        """
        if lang not in self._HF_REPOS:
            return False
        if not self._ct2_fetch(lang):
            return False
        self._ct2 = None
        self._select_backend()
        return self.backend == "ct2"

    # ------------------------------------------------------------ argos ---

    def _argos_ensure(self, lang: str) -> bool:
        """Instala lang->pt; se não existir par direto, instala lang->en + en->pt."""
        pkg, _tr = self._argos
        if self._argos_has_path(lang):
            return True
        try:
            pkg.update_package_index()
        except Exception:
            log_mt.warning("update_package_index falhou (offline?)", exc_info=True)
        available = pkg.get_available_packages()

        def _find(frm: str, to: str):
            for p in available:
                if p.from_code == frm and p.to_code == to:
                    return p
            return None

        direct = _find(lang, self.target)
        chain = [direct] if direct else [_find(lang, "en"), _find("en", self.target)]
        if any(c is None for c in chain):
            return False
        for p in chain:
            if self._argos_installed(p.from_code, p.to_code):
                continue
            log_mt.info("baixando pacote argos %s->%s…", p.from_code, p.to_code)
            try:
                p.install()          # argostranslate >= 1.9
            except AttributeError:
                pkg.install_from_path(p.download())
        return self._argos_has_path(lang)

    def _argos_installed(self, frm: str, to: str) -> bool:
        pkg, _tr = self._argos
        try:
            return any(p.from_code == frm and p.to_code == to
                       for p in pkg.get_installed_packages())
        except Exception:
            return False

    def _argos_translation(self, lang: str):
        """Objeto de tradução lang->pt (inclui pivô automático) ou None."""
        _pkg, tr = self._argos
        langs = {l.code: l for l in tr.get_installed_languages()}
        src, dst = langs.get(lang), langs.get(self.target)
        if src is None or dst is None:
            return None
        try:
            return src.get_translation(dst)
        except Exception:
            return None

    def _argos_has_path(self, lang: str) -> bool:
        try:
            return self._argos_translation(lang) is not None
        except Exception:
            return False

    # -------------------------------------------------------------- ct2 ---

    # Repositórios do Hub JÁ convertidos para CTranslate2 (opcional: preencha se
    # conhecer um, evita a conversão local). Vazio => converte na 1ª execução.
    _CT2_REPOS: dict[str, list[str]] = {}
    # Modelos transformers Opus-MT convertidos localmente com ct2 (cache em models/).
    _HF_REPOS = {"en": "Helsinki-NLP/opus-mt-tc-big-en-pt"}

    def _ct2_ensure(self, lang: str) -> bool:
        """Baixa (ou converte) o modelo Opus-MT lang->pt e carrega no CT2."""
        if self._ct2 is not None and self._ct2.get("lang") == lang:
            return True
        if lang not in self._HF_REPOS:
            return False
        model_dir = self._ct2_fetch(lang)
        if not model_dir:
            return False
        import ctranslate2
        import sentencepiece as spm
        sp_src = os.path.join(model_dir, "source.spm")
        sp_tgt = os.path.join(model_dir, "target.spm")
        if not os.path.exists(sp_src):
            log_mt.error("source.spm ausente em %s", model_dir)
            return False
        self._ct2 = {
            "lang": lang,
            # intra_threads=4: divide o CPU com o Whisper (8 threads) sem
            # disputar todos os núcleos nos picos
            "translator": ctranslate2.Translator(model_dir, device="cpu",
                                                 compute_type="int8",
                                                 inter_threads=1,
                                                 intra_threads=4),
            "sp_src": spm.SentencePieceProcessor(model_file=sp_src),
            "sp_tgt": (spm.SentencePieceProcessor(model_file=sp_tgt)
                       if os.path.exists(sp_tgt) else None),
        }
        return True

    def _ct2_fetch(self, lang: str) -> Optional[str]:
        """Retorna o diretório local do modelo CT2 (cacheado em models/)."""
        cache = os.path.join(MODELS_DIR, f"opus-mt-{lang}-{self.target}-ct2")
        if os.path.exists(os.path.join(cache, "model.bin")):
            return cache
        os.makedirs(MODELS_DIR, exist_ok=True)
        try:
            from huggingface_hub import snapshot_download
        except Exception:
            log_mt.exception("huggingface_hub indisponível")
            return None
        for repo in self._CT2_REPOS.get(lang, []):
            try:
                path = snapshot_download(repo_id=repo, local_dir=cache)
                if os.path.exists(os.path.join(path, "model.bin")):
                    log_mt.info("modelo CT2 obtido de %s", repo)
                    return path
            except Exception:
                log_mt.info("repo CT2 %s indisponível", repo)
        # Último recurso: converter o modelo transformers (lento, uma única vez).
        try:
            from ctranslate2.converters import TransformersConverter
            log_mt.info("convertendo %s para CT2 (primeira execução, demora)…",
                        self._HF_REPOS[lang])
            TransformersConverter(self._HF_REPOS[lang]).convert(
                cache, quantization="int8", force=True)
            # Os .spm vêm do repositório original.
            from huggingface_hub import hf_hub_download
            for fname in ("source.spm", "target.spm", "vocab.json"):
                try:
                    src = hf_hub_download(self._HF_REPOS[lang], fname)
                    import shutil
                    shutil.copy(src, os.path.join(cache, fname))
                except Exception:
                    pass
            return cache
        except Exception:
            log_mt.exception("conversão para CT2 falhou")
            return None

    # O opus-mt-tc-big-en-pt é multi-alvo: o 1º token da origem escolhe a
    # variante. ">>pob<<" = português do Brasil (">>por<<" seria pt-PT).
    _CT2_TARGET_TOKEN = ">>pob<<"

    def _ct2_translate(self, text: str) -> str:
        c = self._ct2
        # O Marian é treinado FRASE a FRASE: segmento do ASR com 2-3 frases
        # fragmentadas é o gatilho dos loops de paráfrase ("ação ação ação…").
        # Traduzir cada frase separadamente (em lote, mesma chamada) reduz
        # muito os ecos e ainda melhora a qualidade.
        sents = [s for s in _SENT_RE.split(text) if s.strip()] or [text]
        srcs, soft = [], []
        for s in sents:
            parts = _chunk_long(s)
            srcs.extend(parts)
            soft.extend([True] * (len(parts) - 1) + [False])
        sents = srcs
        batch = [[self._CT2_TARGET_TOKEN] + c["sp_src"].encode(s, out_type=str)
                 for s in sents]
        # repetition_penalty segura os loops do Marian sem mutilar os tokens
        # XPROTECTEDnX do glossário (no_repeat_ngram_size os corrompe, pois
        # tokens diferentes compartilham subpalavras, não usar).
        # max_decoding_length ~1.6x a origem: eco que sobreviver é truncado
        # (pt-BR fica ~1.1-1.2x o inglês em subpalavras; 1.6x tem folga).
        res = c["translator"].translate_batch(
            batch, beam_size=2, max_batch_size=8,
            repetition_penalty=1.2,
            max_decoding_length=max(24, int(1.6 * max(len(b) for b in batch)) + 8))
        sp = c["sp_tgt"] or c["sp_src"]
        outs = []
        for r, src in zip(res, srcs):
            hyp = r.hypotheses[0]
            try:
                o = sp.decode(hyp)
            except Exception:
                o = "".join(hyp).replace("▁", " ").strip()
            # eco que não existe no inglês deste pedaço (comparado por pedaço)
            o, ecoou = _dedupe_echo(o, src)
            if ecoou:
                log_mt.info("eco da tradução removido | %r", o[:100])
            outs.append(o)
        return _join_pieces(outs, srcs, soft)

    # ---------------------------------------------------------- tradução ---

    def _install_async(self, lang: str) -> None:
        """Prepara `lang` numa thread de fundo (nunca bloqueia o pipeline)."""
        if lang in self._pending:
            return
        self._pending.add(lang)

        def _worker() -> None:
            try:
                self.ensure_ready([lang])
            finally:
                self._pending.discard(lang)

        log_mt.info("idioma %s ainda sem pacote; baixando em 2º plano", lang)
        threading.Thread(target=_worker, name=f"mt-install-{lang}",
                         daemon=True).start()

    def translate(self, text: str, src_lang: str) -> str:
        """Traduz para pt-BR. Devolve o texto original se não houver como traduzir.

        Nunca bloqueia baixando pacote: idioma novo dispara instalação em 2º
        plano (se `auto_install`) e este texto segue sem tradução.
        """
        if not text or not text.strip():
            return ""
        text = _strip_fillers(text) or text
        # eco do ASR na entrada ("resistance resistance resistance")
        # faz o Marian alucinar; conector solto no fim idem
        text = collapse_repetitions(text, min_repeats=2)[0]
        text = _strip_trailing_junk(text)
        if text.lower().strip(" .,!?") in _STOP_SEGMENTS:
            return ""   # "the" solto vira alucinação do MT
        lang = (src_lang or "").split("-")[0].lower()
        if lang == self.target or self.backend == "none" or lang in self._failed_langs:
            return text
        t0 = time.monotonic()
        try:
            if lang not in self._ready_langs:
                # Já instalado de execuções anteriores? (checagem local, barata)
                if self.backend == "argos" and self._argos_has_path(lang):
                    self._ready_langs.add(lang)
                else:
                    if self.auto_install:
                        self._install_async(lang)
                    return text
            if self.backend == "argos":
                tr = self._argos_translation(lang)
                if tr is None:
                    self._ready_langs.discard(lang)
                    self._failed_langs.add(lang)
                    return text
                out = tr.translate(text)
                out, ecoou = _dedupe_echo(out, text)
                if ecoou:
                    log_mt.info("eco da tradução removido | %r", out[:100])
            else:
                out = self._ct2_translate(text)
            # rede de segurança contra loop do PRÓPRIO tradutor (o Marian
            # repete n-gramas com entrada ruidosa; o Argos idem)
            # na saída do MT a régua é mais dura que no ASR: palavra 3x já
            # é loop ("durável durável durável"), nunca ênfase legítima
            out, cut = collapse_repetitions(out, min_repeats=2)
            out, cut2 = _dedupe_adjacent_sentences(out)
            out = _DOUBLED_WORD_RE.sub(r"\1\2\3", out)
            if cut or cut2:
                log_mt.info("loop de repetição da tradução colapsado | %r",
                            out[:100])
            out = _polish(out) or text
            log_mt.info("traduzido %s->%s em %.0f ms (%d->%d chars)",
                        lang, self.target, (time.monotonic() - t0) * 1000,
                        len(text), len(out))
            return out
        except Exception:
            log_mt.exception("erro traduzindo (%s->%s); devolvendo original",
                             lang, self.target)
            return text


# ==========================================================================
# TtsSpeaker
# ==========================================================================

def _placeholder_translation(text: str) -> Translation:
    """Translation sintética para preencher TtsAudio.source quando não fornecida."""
    seg = SpeechSegment(pcm=np.zeros(0, dtype=np.float32), t_start=0.0, t_end=0.0)
    return Translation(text_pt=text,
                       source=Transcript(text=text, lang="pt", lang_prob=0.0, segment=seg))


def _decode_audio(data: bytes) -> tuple[np.ndarray, int, str]:
    """mp3 -> (float32 mono em [-1,1], samplerate real, nome do decodificador).

    Tenta `soundfile` (libsndfile >= 1.2 lê mp3); se falhar, usa `miniaudio`
    convertendo int16 -> float32 e fazendo downmix quando vier estéreo.
    """
    try:
        import soundfile as sf
        pcm, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        if pcm.size:
            return np.ascontiguousarray(pcm.mean(axis=1), dtype=np.float32), int(sr), "soundfile"
        raise RuntimeError("soundfile devolveu 0 amostras")
    except Exception as exc:
        log_tts.debug("soundfile não decodificou o mp3 (%s); tentando miniaudio", exc)
    import miniaudio
    dec = miniaudio.decode(data)
    arr = np.asarray(dec.samples, dtype=np.int16).astype(np.float32) / 32768.0
    ch = int(getattr(dec, "nchannels", 1) or 1)
    if ch > 1:
        arr = arr[: (len(arr) // ch) * ch].reshape(-1, ch).mean(axis=1)
    return np.ascontiguousarray(arr, dtype=np.float32), int(dec.sample_rate), "miniaudio"


#: marcador de fim devolvido por `TtsJob.get()`
JOB_END = object()


class TtsJob:
    """Uma frase em síntese: fila thread-safe de trechos de PCM em ordem.

    O produtor (thread do pool, via `TtsSpeaker.synth_stream`) chama
    `put_chunk()` e por fim `finish()`; o consumidor (thread de TTS do
    pipeline) lê com `get()` e toca os trechos conforme chegam, enquanto o
    próximo job já sintetiza em paralelo. `cancel()` abandona o job (parada,
    fila cheia): o produtor para de empurrar e o consumidor acorda.
    """

    def __init__(self, source: Optional[Translation] = None) -> None:
        self.source = source
        self.rate = 0                       # aceleração (%) escolhida p/ esta frase
        self.cancelled = threading.Event()
        self.failed = False                 # terminou por falha no meio do stream
        self.samples = 0                    # amostras já produzidas
        self.samplerate = 0
        self.t_first: Optional[float] = None   # monotonic do 1º trecho
        self._q: "queue.Queue" = queue.Queue()
        self._lock = threading.Lock()
        self._closed = False
        self._attempt = 0

    def new_attempt(self) -> int:
        """Abre uma tentativa de síntese; as anteriores passam a ser obsoletas.

        Sem isto, a tentativa 1 (abandonada por timeout) podia empurrar o seu
        1º trecho depois de a tentativa 2 já ter começado: a frase tocava
        repetida. `put_chunk(attempt=id)` descarta o que vier de id velho.
        """
        with self._lock:
            self._attempt += 1
            return self._attempt

    def is_current(self, attempt: int) -> bool:
        return attempt == self._attempt

    def put_chunk(self, pcm: np.ndarray, samplerate: int, last: bool = False,
                  attempt: Optional[int] = None) -> None:
        arr = np.ascontiguousarray(np.asarray(pcm, dtype=np.float32).reshape(-1))
        with self._lock:
            if self._closed or self.cancelled.is_set():
                return
            if attempt is not None and attempt != self._attempt:
                return   # tentativa obsoleta
            if arr.size and self.t_first is None:
                self.t_first = time.monotonic()
            self.samples += arr.size
            self.samplerate = int(samplerate)
            self._q.put((arr, int(samplerate), bool(last)))

    def fail(self) -> None:
        self.failed = True

    def finish(self) -> None:
        """Marca o fim do stream (idempotente)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._q.put(JOB_END)

    def cancel(self) -> None:
        """Abandona o job (também serve de `cancel()` p/ `_drop_oldest_put`)."""
        self.cancelled.set()
        self.finish()

    def get(self, timeout: float = 0.2):
        """Próximo `(pcm, samplerate, last)`, `JOB_END` ou None (timeout)."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def log_done(self, nchars: int, rate: str, t0: float) -> None:
        now = time.monotonic()
        primeiro = (self.t_first - t0) * 1000 if self.t_first else 0.0
        log_tts.info("síntese %d chars rate=%s: 1º trecho em %.0f ms, total "
                     "%.0f ms (áudio %.2fs @%d Hz)", nchars, rate, primeiro,
                     (now - t0) * 1000, self.samples / max(1, self.samplerate),
                     self.samplerate)


class TtsSpeaker:
    """Voz pt-BR via edge-tts (nuvem, gratuito), com API síncrona.

    O edge-tts é assíncrono: o __init__ sobe uma thread daemon com um event
    loop próprio e `synth()` despacha para lá com `run_coroutine_threadsafe`.
    """

    # Disjuntor da nuvem: com o DNS caindo, cada frase custava ~29 s (2
    # tentativas de 8 s + espera + voz offline) e o pipeline inteiro atolava.
    # Depois de _EDGE_FAIL_LIMIT falhas seguidas vamos DIRETO para a voz
    # offline por _EDGE_COOLDOWN_S, e só então testamos a nuvem de novo.
    _EDGE_FAIL_LIMIT = 3
    _EDGE_COOLDOWN_S = 60.0
    # streaming: 1ª decodificação com ~0.2 s de mp3 (48 kbps = 6000 B/s), as
    # seguintes a cada ~0.4 s; segura 1 quadro mp3 (1152 amostras @24 kHz)
    _STREAM_FIRST_BYTES = 1200
    _STREAM_STEP_BYTES = 2400
    _MP3_FRAME = 1152

    def __init__(self, voice: str = "pt-BR-FranciscaNeural") -> None:
        self.voice = voice
        self._edge_fails = 0
        self._edge_blocked_until = 0.0
        self._decoder = "?"          # "soundfile" | "miniaudio" (definido no 1º uso)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="tts-loop",
                                        daemon=True)
        self._thread.start()
        self._sapi = None            # engine pyttsx3 (voz offline reserva), lazy
        self._sapi_falhou = False
        log_tts.info("TtsSpeaker pronto (voz=%s)", voice)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def close(self) -> None:
        """Encerra o event loop da thread de TTS."""
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=2.0)
        except Exception:
            pass

    # ----------------------------------------------------------- síntese ---

    def _prepare(self, text: str) -> str:
        """Normaliza e protege contra textos absurdamente longos."""
        t = _WS_RE.sub(" ", (text or "").replace("\n", " ")).strip()
        if len(t) > TTS_HARD_LIMIT:
            corte = t[:TTS_HARD_LIMIT]
            frases = _SENT_RE.split(corte)
            if len(frases) > 1:
                corte = " ".join(frases[:-1])
            log_tts.warning("texto de %d chars truncado para %d", len(t), len(corte))
            t = corte
        elif len(t) > TTS_SOFT_LIMIT:
            # Longo mas aceitável: uma única chamada, só registramos.
            log_tts.info("texto longo (%d chars, %d frases) numa única síntese",
                         len(t), len(_SENT_RE.split(t)))
        return t

    @staticmethod
    def _decode_prefix(data: bytes) -> "Optional[tuple[np.ndarray, int]]":
        """Decodifica um mp3 TRUNCADO (soundfile lê o prefixo sem reclamar).

        Medido com mp3 real do edge-tts: o prefixo decodifica igual ao arquivo
        inteiro, exceto a última fração de quadro (a amostra final de um
        prefixo pode vir parcial), por isso quem chama segura 1 quadro.
        Devolve None se ainda não deu para decodificar nada.
        """
        try:
            import soundfile as sf
            pcm, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        except Exception:
            return None
        if not pcm.size:
            return None
        return np.ascontiguousarray(pcm.mean(axis=1), dtype=np.float32), int(sr)

    async def _stream_job(self, text: str, rate: str, job: "TtsJob",
                          attempt: Optional[int] = None) -> None:
        """Faz o stream do edge-tts e empurra PCM no `job` à medida que chega.

        A cada ~0.4 s de mp3 recebido decodifica o buffer acumulado e entrega
        só as amostras novas, segurando o último quadro (1152 amostras) até
        haver mais dados ou o fim: quadro parcial nunca vira estalo. A
        decodificação repetida é O(n²), mas custa ~2,5 ms mesmo para 6,5 s de
        fala (medido), irrelevante perto da rede.
        """
        import edge_tts
        comm = edge_tts.Communicate(text, self.voice, rate=rate)
        buf = bytearray()
        emitted = 0
        next_at = self._STREAM_FIRST_BYTES
        async for chunk in comm.stream():
            if job.cancelled.is_set() or (
                    attempt is not None and not job.is_current(attempt)):
                return
            if chunk.get("type") != "audio":
                continue
            buf.extend(chunk["data"])
            if len(buf) < next_at:
                continue
            next_at = len(buf) + self._STREAM_STEP_BYTES
            dec = self._decode_prefix(bytes(buf))
            if dec is None:
                continue
            pcm, sr = dec
            end = pcm.size - self._MP3_FRAME
            if end > emitted:
                job.put_chunk(pcm[emitted:end], sr, attempt=attempt)
                emitted = end
        if job.cancelled.is_set():
            return
        if not buf:
            raise RuntimeError("edge-tts devolveu áudio vazio")
        pcm, sr, dec = _decode_audio(bytes(buf))   # fim: decodificação completa
        if pcm.size == 0 or sr <= 0:
            raise RuntimeError("decodificação resultou em áudio vazio")
        if self._decoder != dec:
            self._decoder = dec
            log_tts.info("decodificador de mp3 ativo: %s", dec)
        job.put_chunk(pcm[min(emitted, pcm.size):], sr, last=True, attempt=attempt)

    def _note_edge_failure(self) -> None:
        """Disjuntor: contabiliza uma síntese que falhou de vez na nuvem."""
        self._edge_fails += 1
        if self._edge_fails >= self._EDGE_FAIL_LIMIT:
            self._edge_blocked_until = time.monotonic() + self._EDGE_COOLDOWN_S
            self._edge_fails = 0
            log_tts.warning("edge-tts falhou %d vezes seguidas, só voz offline "
                            "pelos próximos %.0fs", self._EDGE_FAIL_LIMIT,
                            self._EDGE_COOLDOWN_S)

    def synth_stream(self, text: str, rate_pct: int = 0,
                     source: Optional[Translation] = None,
                     job: "Optional[TtsJob]" = None) -> "TtsJob":
        """Sintetiza empurrando PCM em `job` (criado aqui se não vier) e o encerra.

        Bloqueia até o fim da síntese (roda numa thread de pool); quem toca
        consome `job.get()` em paralelo. Falha ANTES do 1º trecho: tenta de
        novo, depois cai na voz offline (SAPI, um trecho só). Falha DEPOIS de
        já haver áudio no job: avisa e encerra (não repete do início).
        """
        job = job if job is not None else TtsJob(source)
        t0 = time.monotonic()
        try:
            self._synth_into(job, text, rate_pct, source, t0)
        except Exception:
            log_tts.exception("erro inesperado na síntese")
            job.fail()
        finally:
            job.finish()
        return job

    def _synth_into(self, job: "TtsJob", text: str, rate_pct: int,
                    source: Optional[Translation], t0: float) -> None:
        t = self._prepare(text)
        if not t:
            return
        rate = f"{'+' if rate_pct >= 0 else '-'}{abs(int(rate_pct))}%"
        if time.monotonic() < self._edge_blocked_until:
            self._stream_sapi(job, t, rate_pct, source)   # nuvem em quarentena
            return
        delays = (0.8,)
        for tentativa in range(2):
            fut = None
            if job.cancelled.is_set():   # descartado: não gasta uma conexão
                return
            aid = job.new_attempt()
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self._stream_job(t, rate, job, aid), self._loop)
                fut.result(timeout=8)
                if job.cancelled.is_set():
                    return
                if job.samples == 0:
                    raise RuntimeError("edge-tts devolveu áudio vazio")
                job.log_done(len(t), rate, t0)
                self._edge_fails = 0
                return
            except Exception as exc:
                # sem isto a corotina abandonada seguia rodando no loop e as
                # conexões pendentes iam se acumulando sessão afora
                if fut is not None:
                    fut.cancel()
                job.new_attempt()   # a tentativa que falhou não empurra mais nada
                if job.cancelled.is_set():
                    return
                if job.samples > 0:
                    # já tocou parte da frase: não repete do início
                    log_tts.warning("síntese interrompida no meio (%s); frase "
                                    "cortada em %.2fs de áudio", exc,
                                    job.samples / max(1, job.samplerate))
                    job.fail()
                    self._note_edge_failure()
                    return
                if tentativa < len(delays):
                    log_tts.warning("falha na síntese (tentativa %d): %s, nova em %.1fs",
                                    tentativa + 1, exc, delays[tentativa])
                    time.sleep(delays[tentativa])
                else:
                    log_tts.warning("edge-tts indisponível (%s), usando voz "
                                    "offline do Windows", exc)
        self._note_edge_failure()
        self._stream_sapi(job, t, rate_pct, source)

    def _stream_sapi(self, job: "TtsJob", text: str, rate_pct: int,
                     source: Optional[Translation]) -> None:
        """Voz offline como um único trecho (o SAPI não faz streaming)."""
        if job.cancelled.is_set():
            return
        audio = self._synth_sapi(text, rate_pct, source)
        if audio is not None:
            job.put_chunk(audio.pcm, audio.samplerate, last=True)

    def synth(self, text: str, rate_pct: int = 0,
              source: Optional[Translation] = None) -> Optional[TtsAudio]:
        """Sintetiza `text` acelerado em `rate_pct`%. None se tudo falhar.

        Versão não-streaming, em cima do mesmo caminho: junta os trechos.
        """
        job = self.synth_stream(text, rate_pct, source)
        parts, sr = [], 0
        while True:
            item = job.get(timeout=0.05)
            if item is JOB_END:
                break
            if item is None:
                continue
            parts.append(item[0])
            sr = item[1]
        pcm = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        if pcm.size == 0 or sr <= 0:
            return None
        return TtsAudio(pcm=pcm, samplerate=sr,
                        source=source or _placeholder_translation(text))

    def _synth_sapi(self, text: str, rate_pct: int,
                    source: Optional[Translation]) -> Optional[TtsAudio]:
        """Reserva offline: voz SAPI do Windows (ex.: Microsoft Maria pt-BR).

        Usada só quando o edge-tts falha (sem internet/DNS). Qualidade menor,
        mas a tradução nunca fica muda.

        Roda por PowerShell num PROCESSO SEPARADO com timeout: pyttsx3 no
        mesmo processo podia travar para sempre (runAndWait em thread de pool)
        e emudecer o app inteiro. Num subprocesso, um travamento é morto pelo
        timeout e só custa UMA frase.

        Sem lock: cada chamada usa arquivos próprios (uuid), e serializar aqui
        reduzia à metade a vazão justamente quando a nuvem está fora e a voz
        offline é o único caminho.
        """
        if self._sapi_falhou:
            return None
        import subprocess
        import tempfile
        import uuid

        base = os.path.join(tempfile.gettempdir(),
                            f"tradutor_sapi_{uuid.uuid4().hex[:8]}")
        txt_path, wav_path = base + ".txt", base + ".wav"
        try:
            with open(txt_path, "w", encoding="utf-8-sig") as f:
                f.write(text)
            sapi_rate = max(-10, min(10, int(rate_pct / 12)))
            ps = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$v = $s.GetInstalledVoices() | ForEach-Object "
                "{ $_.VoiceInfo } | Where-Object "
                "{ $_.Culture.Name -eq 'pt-BR' -or $_.Name -match 'Maria' } "
                "| Select-Object -First 1; "
                "if ($v) { $s.SelectVoice($v.Name) }; "
                f"$s.Rate = {sapi_rate}; "
                f"$s.SetOutputToWaveFile('{wav_path}'); "
                f"$s.Speak([IO.File]::ReadAllText('{txt_path}', "
                "[Text.Encoding]::UTF8)); $s.Dispose()"
            )
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           timeout=12, creationflags=flags,
                           capture_output=True)
            import soundfile as sf
            pcm, sr = sf.read(wav_path, dtype="float32")
            if pcm.ndim > 1:
                pcm = pcm.mean(axis=1).astype("float32")
            if pcm.size == 0:
                raise RuntimeError("wav vazio")
            log_tts.info("síntese OFFLINE (SAPI) de %d chars (áudio %.2fs @%d Hz)",
                         len(text), pcm.size / sr, sr)
            return TtsAudio(pcm=pcm, samplerate=sr,
                            source=source or _placeholder_translation(text))
        except subprocess.TimeoutExpired:
            log_tts.warning("voz offline excedeu 12s, frase pulada (sem voz)")
            return None
        except Exception:
            log_tts.exception("voz offline também falhou, seguindo sem voz")
            return None
        finally:
            for p in (txt_path, wav_path):
                try:
                    os.remove(p)
                except OSError:
                    pass

    @property
    def decoder(self) -> str:
        """Nome do decodificador de mp3 em uso ('?' antes da primeira síntese)."""
        return self._decoder
