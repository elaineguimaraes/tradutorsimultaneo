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


def _chunk_long(sent: str, max_words: int = 22) -> "list[str]":
    """Sentença longa (ASR sem pontuação) faz o Marian ecoar; divide."""
    if len(sent.split()) <= max_words:
        return [sent]
    parts = re.split(r"(?<=,)\s+", sent)
    chunks, cur, cnt = [], [], 0
    for part in parts:
        n = len(part.split())
        if cur and cnt + n > max_words:
            chunks.append(" ".join(cur))
            cur, cnt = [], 0
        cur.append(part)
        cnt += n
    if cur:
        chunks.append(" ".join(cur))
    final = []
    for c in chunks:
        cw = c.split()
        while len(cw) > max_words + 6:   # bloco sem vírgula: corte seco
            final.append(" ".join(cw[:max_words]))
            cw = cw[max_words:]
        if cw:
            final.append(" ".join(cw))
    return final or [sent]


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
        sents = [c for s in sents for c in _chunk_long(s)]
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
        for r in res:
            hyp = r.hypotheses[0]
            try:
                outs.append(sp.decode(hyp))
            except Exception:
                outs.append("".join(hyp).replace("▁", " ").strip())
        return " ".join(o for o in outs if o.strip())

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

    async def _stream_mp3(self, text: str, rate: str) -> bytes:
        import edge_tts
        comm = edge_tts.Communicate(text, self.voice, rate=rate)
        buf = bytearray()
        async for chunk in comm.stream():
            if chunk.get("type") == "audio":
                buf.extend(chunk["data"])
        return bytes(buf)

    def synth(self, text: str, rate_pct: int = 0,
              source: Optional[Translation] = None) -> Optional[TtsAudio]:
        """Sintetiza `text` acelerado em `rate_pct`%. None se a rede falhar."""
        t = self._prepare(text)
        if not t:
            return None
        rate = f"{'+' if rate_pct >= 0 else '-'}{abs(int(rate_pct))}%"
        if time.monotonic() < self._edge_blocked_until:
            return self._synth_sapi(t, rate_pct, source)   # nuvem em quarentena
        delays = (0.8,)
        for tentativa in range(2):
            t0 = time.monotonic()
            fut = None
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self._stream_mp3(t, rate), self._loop)
                mp3 = fut.result(timeout=8)
                if not mp3:
                    raise RuntimeError("edge-tts devolveu áudio vazio")
                pcm, sr, dec = _decode_audio(mp3)
                if pcm.size == 0 or sr <= 0:
                    raise RuntimeError("decodificação resultou em áudio vazio")
                if self._decoder != dec:
                    self._decoder = dec
                    log_tts.info("decodificador de mp3 ativo: %s", dec)
                dur = pcm.size / sr
                log_tts.info("síntese %d chars rate=%s em %.0f ms (áudio %.2fs @%d Hz)",
                             len(t), rate, (time.monotonic() - t0) * 1000, dur, sr)
                self._edge_fails = 0
                return TtsAudio(pcm=pcm, samplerate=sr,
                                source=source or _placeholder_translation(t))
            except Exception as exc:
                # sem isto a corotina abandonada seguia rodando no loop e as
                # conexões pendentes iam se acumulando sessão afora
                if fut is not None:
                    fut.cancel()
                if tentativa < len(delays):
                    log_tts.warning("falha na síntese (tentativa %d): %s, nova em %.1fs",
                                    tentativa + 1, exc, delays[tentativa])
                    time.sleep(delays[tentativa])
                else:
                    log_tts.warning("edge-tts indisponível (%s), usando voz "
                                    "offline do Windows", exc)
        self._edge_fails += 1
        if self._edge_fails >= self._EDGE_FAIL_LIMIT:
            self._edge_blocked_until = time.monotonic() + self._EDGE_COOLDOWN_S
            self._edge_fails = 0
            log_tts.warning("edge-tts falhou %d vezes seguidas, só voz offline "
                            "pelos próximos %.0fs", self._EDGE_FAIL_LIMIT,
                            self._EDGE_COOLDOWN_S)
        return self._synth_sapi(t, rate_pct, source)

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
