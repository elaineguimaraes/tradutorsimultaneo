"""Testes standalone da voz em streaming (TtsJob, TtsSpeaker.synth_stream, mixer).

Executar:  .venv\\Scripts\\python.exe tests\\test_tts_stream.py

Não depende de pytest, de rede nem de dispositivo de áudio: o edge-tts é
substituído por um falso que serve um mp3 gerado na hora (soundfile escreve
mp3), e o mixer por um gravador.
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
import threading
import time
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from tradutor import dsp                                          # noqa: E402
from tradutor import translate_tts as T                           # noqa: E402
from tradutor.config import AppConfig                             # noqa: E402
from tradutor.contracts import TtsAudio                           # noqa: E402
from tradutor.main import Pipeline                                # noqa: E402

try:  # acentos no console do Windows
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass


# --------------------------------------------------------------------------
# Apoio
# --------------------------------------------------------------------------

def _mp3_sintetico(seg: float = 3.0, sr: int = 24000) -> bytes:
    import soundfile as sf
    t = np.arange(int(seg * sr)) / sr
    x = (0.3 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t)))
    buf = io.BytesIO()
    sf.write(buf, x.astype("float32"), sr, format="MP3")
    return buf.getvalue()


def _decode_inteiro(mp3: bytes) -> np.ndarray:
    pcm, _sr, _dec = T._decode_audio(mp3)
    return pcm


class FakeEdge:
    """Substitui o módulo edge_tts: serve `mp3` em pedaços de 720 bytes.

    `falhar_apos` = nº de pedaços entregues antes de lançar (None = não falha).
    """
    mp3 = b""
    falhar_apos = None
    criados = 0

    class Communicate:
        def __init__(self, text, voice, rate="+0%"):
            FakeEdge.criados += 1

        async def stream(self):
            data = FakeEdge.mp3
            n = 0
            for i in range(0, len(data), 720):
                if FakeEdge.falhar_apos is not None and n >= FakeEdge.falhar_apos:
                    raise ConnectionError("rede caiu")
                n += 1
                yield {"type": "audio", "data": data[i:i + 720]}
                await asyncio.sleep(0)


class com_edge_falso:
    def __init__(self, mp3: bytes = b"", falhar_apos=None) -> None:
        self.mp3, self.falhar_apos = mp3, falhar_apos

    def __enter__(self):
        mod = types.ModuleType("edge_tts")
        mod.Communicate = FakeEdge.Communicate
        self._old = sys.modules.get("edge_tts")
        sys.modules["edge_tts"] = mod
        FakeEdge.mp3, FakeEdge.falhar_apos, FakeEdge.criados = (
            self.mp3, self.falhar_apos, 0)

    def __exit__(self, *exc):
        if self._old is None:
            sys.modules.pop("edge_tts", None)
        else:
            sys.modules["edge_tts"] = self._old


def _drena(job: T.TtsJob):
    itens = []
    while True:
        it = job.get(timeout=1.0)
        if it is T.JOB_END:
            return itens
        assert it is not None, "job não terminou"
        itens.append(it)


class MixerFalso:
    def __init__(self) -> None:
        self.chunks: list = []
        self.cleared = 0
        self.backlog = 0.0

    def enqueue_tts(self, pcm, *, fade_in=True, fade_out=True):
        self.chunks.append((np.array(pcm), fade_in, fade_out))

    def tts_backlog_seconds(self):
        return self.backlog

    def clear_tts(self):
        self.cleared += 1


def _pipeline(mixer: MixerFalso) -> Pipeline:
    p = Pipeline(AppConfig())
    p._mixer = mixer
    p._running = True
    p._generation = 1
    return p


# --------------------------------------------------------------------------
# Testes
# --------------------------------------------------------------------------

def test_job_ordem_e_fim() -> None:
    job = T.TtsJob()
    for i in range(5):
        job.put_chunk(np.full(10, i, dtype=np.float32), 24000, last=(i == 4))
    job.finish()
    job.finish()                       # idempotente
    job.put_chunk(np.ones(3, np.float32), 24000)   # depois do fim: ignorado
    itens = _drena(job)
    assert [int(c[0][0]) for c in itens] == [0, 1, 2, 3, 4], itens
    assert [c[2] for c in itens] == [False] * 4 + [True]
    assert job.samples == 50


def test_job_cancel_acorda_o_consumidor() -> None:
    job = T.TtsJob()
    out = []
    th = threading.Thread(target=lambda: out.append(job.get(timeout=5.0)))
    th.start()
    time.sleep(0.05)
    job.cancel()
    th.join(2.0)
    assert out == [T.JOB_END] and job.cancelled.is_set()
    job.put_chunk(np.ones(3, np.float32), 24000)   # cancelado: não entra
    assert job.samples == 0


def test_resample_em_blocos_igual_ao_inteiro() -> None:
    rng = np.random.default_rng(1)
    for sr in (24000, 22050):
        x = (0.3 * rng.standard_normal(sr * 2)).astype(np.float32)
        inteiro = dsp.resample(x, sr, 48000)
        for tamanhos in ([2880, 5760, 1000, 333, 17], [4800], [1, 700]):
            r = dsp.StreamResampler(sr, 48000)
            partes, i, k = [], 0, 0
            while i < len(x):
                n = tamanhos[k % len(tamanhos)]
                partes.append(r.process(x[i:i + n]))
                i += n
                k += 1
            partes.append(r.flush())
            y = np.concatenate(partes)
            assert len(y) == len(inteiro), (sr, tamanhos, len(y), len(inteiro))
            assert float(np.abs(y - inteiro).max()) < 1e-6, (sr, tamanhos)
    # sem estalo nas juntas: a derivada do tom em blocos é a do inteiro
    t = np.arange(24000) / 24000
    tom = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    r = dsp.StreamResampler(24000, 48000)
    y = np.concatenate([r.process(tom[i:i + 1152]) for i in range(0, len(tom), 1152)]
                       + [r.flush()])
    ref = dsp.resample(tom, 24000, 48000)
    assert np.abs(np.diff(y)).max() <= np.abs(np.diff(ref)).max() + 1e-6
    # o resample por trecho INGÊNUO teria transitório: o teste vale a pena
    ingenuo = np.concatenate([dsp.resample(tom[i:i + 1152], 24000, 48000)
                              for i in range(0, len(tom), 1152)])
    assert np.abs(ingenuo - ref[:len(ingenuo)]).max() > 1e-3


def test_stream_igual_a_decodificacao_inteira() -> None:
    mp3 = _mp3_sintetico()
    ref = _decode_inteiro(mp3)
    spk = T.TtsSpeaker()
    try:
        with com_edge_falso(mp3):
            job = spk.synth_stream("Olá, mundo.", 0)
            itens = _drena(job)
    finally:
        spk.close()
    assert len(itens) >= 3, f"esperava vários trechos, veio {len(itens)}"
    assert [c[2] for c in itens[:-1]] == [False] * (len(itens) - 1)
    assert itens[-1][2] is True
    y = np.concatenate([c[0] for c in itens])
    assert y.shape == ref.shape, (y.shape, ref.shape)
    assert float(np.abs(y - ref).max()) == 0.0, "trechos devem somar o áudio inteiro"
    assert job.t_first is not None and not job.failed


def test_synth_nao_streaming_continua_valendo() -> None:
    mp3 = _mp3_sintetico(2.0)
    spk = T.TtsSpeaker()
    try:
        with com_edge_falso(mp3):
            audio = spk.synth("Olá, mundo.", 0)
    finally:
        spk.close()
    assert isinstance(audio, TtsAudio) and audio.samplerate == 24000
    assert np.array_equal(audio.pcm, _decode_inteiro(mp3))


def test_falha_no_meio_encerra_sem_repetir() -> None:
    mp3 = _mp3_sintetico()
    spk = T.TtsSpeaker()
    chamado = []
    spk._synth_sapi = lambda *a, **k: chamado.append(1)
    try:
        with com_edge_falso(mp3, falhar_apos=3):
            job = spk.synth_stream("Olá, mundo.", 0)
            itens = _drena(job)
            criados = FakeEdge.criados
    finally:
        spk.close()
    assert itens, "o que já foi decodificado deve ter sido entregue"
    assert job.failed and job.samples > 0
    assert criados == 1, "não pode repetir do início depois de já ter áudio"
    assert not chamado, "SAPI só entra quando falha ANTES do 1º trecho"
    assert all(not c[2] for c in itens)


def test_falha_antes_do_1o_trecho_cai_no_sapi_e_disjuntor() -> None:
    spk = T.TtsSpeaker()
    voz = np.full(2205, 0.1, dtype=np.float32)
    spk._synth_sapi = lambda text, rate_pct, source: TtsAudio(
        pcm=voz, samplerate=22050, source=source or T._placeholder_translation(text))
    sleep_orig = T.time.sleep
    T.time.sleep = lambda s: None     # a espera entre tentativas não importa aqui
    try:
        with com_edge_falso(b"", falhar_apos=0):
            for i in range(3):
                itens = _drena(spk.synth_stream(f"frase {i}", 0))
                assert len(itens) == 1 and itens[0][2] is True
                assert itens[0][1] == 22050
            assert FakeEdge.criados == 6, "2 tentativas por frase"
            assert time_blocked(spk), "3 falhas seguidas abrem o disjuntor"
            antes = FakeEdge.criados
            itens = _drena(spk.synth_stream("frase 4", 0))
            assert len(itens) == 1
            assert FakeEdge.criados == antes, "em quarentena vai direto ao SAPI"
    finally:
        T.time.sleep = sleep_orig
        spk.close()


def time_blocked(spk) -> bool:
    return time.monotonic() < spk._edge_blocked_until


def test_sapi_sem_audio_termina_vazio() -> None:
    spk = T.TtsSpeaker()
    spk._synth_sapi = lambda *a, **k: None
    spk._edge_blocked_until = time.monotonic() + 60
    try:
        assert _drena(spk.synth_stream("oi", 0)) == []
        assert spk.synth("oi", 0) is None
    finally:
        spk.close()


def test_play_job_enfileira_trechos_com_fades() -> None:
    mixer = MixerFalso()
    p = _pipeline(mixer)
    job = T.TtsJob()
    job.rate = 25
    rng = np.random.default_rng(3)
    x = (0.2 * rng.standard_normal(24000)).astype(np.float32)
    for i in range(0, len(x), 4800):
        job.put_chunk(x[i:i + 4800], 24000, last=(i + 4800 >= len(x)))
    job.finish()
    assert p._play_job(job, 1) is True
    y = np.concatenate([c[0] for c in mixer.chunks])
    assert np.abs(y - dsp.resample(x, 24000, 48000)).max() < 1e-6
    fades = [(c[1], c[2]) for c in mixer.chunks]
    assert fades[0][0] is True and all(not f[0] for f in fades[1:]), fades
    assert fades[-1][1] is True and all(not f[1] for f in fades[:-1]), fades
    assert p._state.rate_pct == 25 and mixer.cleared == 0


def test_play_job_backlog_limpa_uma_vez() -> None:
    mixer = MixerFalso()
    mixer.backlog = 99.0
    p = _pipeline(mixer)
    job = T.TtsJob()
    for i in range(3):
        job.put_chunk(np.zeros(4800, np.float32), 24000, last=(i == 2))
    job.finish()
    p._play_job(job, 1)
    assert mixer.cleared == 1, mixer.cleared


def test_play_job_falha_no_meio_fecha_com_fade_de_saida() -> None:
    mixer = MixerFalso()
    p = _pipeline(mixer)
    job = T.TtsJob()
    job.put_chunk(np.zeros(4800, np.float32), 24000)
    job.put_chunk(np.zeros(4800, np.float32), 24000)
    job.fail()
    job.finish()          # sem trecho "last": o worker fecha o resto
    assert p._play_job(job, 1) is True
    assert mixer.chunks[-1][2] is True
    assert sum(len(c[0]) for c in mixer.chunks) == 2 * 4800 * 2


def test_stop_cancela_o_job_e_nao_toca_mais() -> None:
    mixer = MixerFalso()
    p = _pipeline(mixer)
    job = T.TtsJob()
    job.put_chunk(np.zeros(4800, np.float32), 24000)
    res = []
    th = threading.Thread(target=lambda: res.append(p._play_job(job, 1)))
    th.start()
    time.sleep(0.4)
    n_antes = len(mixer.chunks)
    p._running = False     # parada do pipeline (geração aposentada)
    th.join(3.0)
    assert res == [False] and job.cancelled.is_set()
    job.put_chunk(np.zeros(4800, np.float32), 24000)
    assert len(mixer.chunks) == n_antes, "nada toca depois da parada"
    # job abandonado na fila: o cancel() de _drop_oldest_put/_stop_impl acorda
    j2 = T.TtsJob()
    j2.cancel()
    assert p._alive(1) is False


def test_job_cancelado_pelo_producer_para_o_stream() -> None:
    mp3 = _mp3_sintetico()
    spk = T.TtsSpeaker()
    try:
        with com_edge_falso(mp3):
            job = T.TtsJob()
            job.cancel()
            spk.synth_stream("Olá.", 0, job=job)
    finally:
        spk.close()
    assert job.samples == 0


def test_ao_vivo_cancela_o_job_em_andamento() -> None:
    mixer = MixerFalso()
    p = _pipeline(mixer)
    job = T.TtsJob()
    job.put_chunk(np.zeros(9600, np.float32), 24000)
    res = []
    th = threading.Thread(target=lambda: res.append(p._play_job(job, 1)))
    th.start()
    time.sleep(0.4)
    assert p._playing_job is job
    p.skip_to_live()                      # "ao vivo" durante a frase
    th.join(3.0)
    assert res == [True] and job.cancelled.is_set() and mixer.cleared == 1
    n = len(mixer.chunks)
    job.put_chunk(np.ones(9600, np.float32), 24000, last=True)   # resto da frase velha
    time.sleep(0.2)
    assert len(mixer.chunks) == n, "o resto da frase cancelada não pode tocar"
    assert p._playing_job is None


def test_job_cancelado_nao_gasta_conexao() -> None:
    mp3 = _mp3_sintetico()
    spk = T.TtsSpeaker()
    p = _pipeline(MixerFalso())
    p._speaker = spk
    try:
        with com_edge_falso(mp3):
            job = T.TtsJob(T._placeholder_translation("oi"))
            job.cancel()
            p._synth_job(job)             # job descartado antes de começar
            assert FakeEdge.criados == 0, "nem abriu o stream"
            # cancelado entre a 1ª tentativa (que falha) e a 2ª: não tenta de novo
            FakeEdge.falhar_apos = 0
            job2 = T.TtsJob(T._placeholder_translation("oi"))
            orig = T.time.sleep
            T.time.sleep = lambda s: job2.cancel()
            try:
                spk.synth_stream("oi", 0, job=job2)
            finally:
                T.time.sleep = orig
            assert FakeEdge.criados == 1, FakeEdge.criados
    finally:
        spk.close()


def test_fim_abrupto_fecha_com_fade_de_saida_real() -> None:
    mixer = MixerFalso()
    p = _pipeline(mixer)
    job = T.TtsJob()
    job.put_chunk(np.full(4800, 0.5, np.float32), 24000)
    job.put_chunk(np.full(4800, 0.5, np.float32), 24000)
    job.fail()
    job.finish()                          # acaba sem trecho "last"
    p._play_job(job, 1)
    saida = mixer.chunks[-1]
    assert saida[2] is True and len(saida[0]) >= p._PLAY_TAIL_SAMPLES, len(saida[0])
    assert [c[1] for c in mixer.chunks][0] is True
    assert sum(len(c[0]) for c in mixer.chunks) == 2 * 4800 * 2
    # um trecho só, curto: tudo sai de uma vez com os dois fades
    mixer2 = MixerFalso()
    p2 = _pipeline(mixer2)
    j2 = T.TtsJob()
    j2.put_chunk(np.full(100, 0.5, np.float32), 24000)
    j2.finish()
    p2._play_job(j2, 1)
    assert len(mixer2.chunks) == 1 and mixer2.chunks[0][1:] == (True, True)


def test_tentativa_obsoleta_nao_empurra_trecho() -> None:
    job = T.TtsJob()
    a1 = job.new_attempt()
    a2 = job.new_attempt()                # a 2ª começou: a 1ª é obsoleta
    job.put_chunk(np.ones(10, np.float32), 24000, attempt=a1)
    assert job.samples == 0
    job.put_chunk(np.ones(10, np.float32), 24000, attempt=a2)
    assert job.samples == 10
    # o stream obsoleto para sozinho (não decodifica nem empurra)
    mp3 = _mp3_sintetico()
    spk = T.TtsSpeaker()
    try:
        with com_edge_falso(mp3):
            j = T.TtsJob()
            old = j.new_attempt()
            j.new_attempt()
            fut = asyncio.run_coroutine_threadsafe(
                spk._stream_job("oi", "+0%", j, old), spk._loop)
            fut.result(timeout=5)
            assert j.samples == 0
    finally:
        spk.close()


def test_geracao_antiga_nao_submete_no_pool_novo() -> None:
    mixer = MixerFalso()
    p = _pipeline(mixer)
    submetidos = []

    class PoolFalso:
        def submit(self, fn, *a):
            submetidos.append(a)

    class Gl:
        def apply(self, fn, text):
            return "oi"

    class Tr:
        def translate(self, t, lang):
            return "oi"

    p._tts_pool, p._glossary, p._translator = PoolFalso(), Gl(), Tr()
    trans = T._placeholder_translation("hello").source
    trans.lang = "en"
    p._translate_and_queue("hello", trans, 0)       # geração 0: aposentada
    assert not submetidos and p._q_tts.empty()
    p._translate_and_queue("hello", trans, 1)       # geração atual
    assert len(submetidos) == 1 and p._q_tts.qsize() == 1


def test_enqueue_tts_fades_do_mixer() -> None:
    """O mixer real: fade só onde pedido (sem dispositivo de áudio)."""
    from tradutor.audio_io import OutputMixer
    mx = OutputMixer.__new__(OutputMixer)
    mx.samplerate = 48000
    mx._lock = threading.Lock()
    mx._tts = []
    mx._tts_remaining = 0
    ones = np.ones(4800, dtype=np.float32)
    mx.enqueue_tts(ones, fade_in=True, fade_out=False)
    mx.enqueue_tts(ones, fade_in=False, fade_out=True)
    a, b = mx._tts
    assert a[0] == 0.0 and a[-1] == 1.0
    assert b[0] == 1.0 and b[-1] == 0.0
    mx.enqueue_tts(ones)       # padrão antigo: os dois fades
    c = mx._tts[-1]
    assert c[0] == 0.0 and c[-1] == 0.0


TESTS = [
    test_job_ordem_e_fim,
    test_job_cancel_acorda_o_consumidor,
    test_resample_em_blocos_igual_ao_inteiro,
    test_stream_igual_a_decodificacao_inteira,
    test_synth_nao_streaming_continua_valendo,
    test_falha_no_meio_encerra_sem_repetir,
    test_falha_antes_do_1o_trecho_cai_no_sapi_e_disjuntor,
    test_sapi_sem_audio_termina_vazio,
    test_play_job_enfileira_trechos_com_fades,
    test_play_job_backlog_limpa_uma_vez,
    test_play_job_falha_no_meio_fecha_com_fade_de_saida,
    test_stop_cancela_o_job_e_nao_toca_mais,
    test_job_cancelado_pelo_producer_para_o_stream,
    test_ao_vivo_cancela_o_job_em_andamento,
    test_job_cancelado_nao_gasta_conexao,
    test_fim_abrupto_fecha_com_fade_de_saida_real,
    test_tentativa_obsoleta_nao_empurra_trecho,
    test_geracao_antiga_nao_submete_no_pool_novo,
    test_enqueue_tts_fades_do_mixer,
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
