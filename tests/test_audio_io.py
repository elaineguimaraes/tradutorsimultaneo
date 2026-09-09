"""Teste standalone do módulo de áudio.

Uso:
    python tests\\test_audio_io.py            # roda tudo (faz barulho!)
    python tests\\test_audio_io.py --offline  # só a lógica, sem abrir dispositivos

Cobre: listagem de dispositivos, tom de 440 Hz pelo caminho do TTS
(enqueue_tts), passthrough com ruído a ganho baixo, ducking simultâneo
(verificando que a rampa não dá degrau => não estala), underrun e, se houver
loopback disponível, 2 s de captura com relatório de RMS.
"""

from __future__ import annotations

import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tradutor.audio_io import (  # noqa: E402
    BLOCKSIZE,
    SR_NATIVE,
    LoopbackCapture,
    OutputMixer,
    default_loopback_name,
    find_cable_device,
    list_loopback_devices,
    list_output_devices,
)

SR = SR_NATIVE
OFFLINE = "--offline" in sys.argv or os.environ.get("TRADUTOR_TEST_OFFLINE") == "1"
RESULTS: list[tuple[str, str, str]] = []   # (nome, status, detalhe)


def record(nome: str, ok: bool, detalhe: str = "", skip: bool = False) -> None:
    status = "SKIP" if skip else ("PASS" if ok else "FALHOU")
    RESULTS.append((nome, status, detalhe))
    print(f"  [{status}] {nome}" + (f": {detalhe}" if detalhe else ""))


def tone(freq: float, seconds: float, amp: float = 0.35) -> np.ndarray:
    """Tom senoidal mono float32."""
    t = np.arange(int(seconds * SR), dtype=np.float32) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def noise(seconds: float, amp: float = 0.25) -> np.ndarray:
    """Ruído branco mono float32."""
    rng = np.random.default_rng(1234)
    return (amp * rng.standard_normal(int(seconds * SR))).astype(np.float32)


def prime(mx: OutputMixer, valor: float = 1.0, ms: int = 120) -> None:
    """Enche o colchão de passthrough (o mixer só toca depois de ~60 ms)."""
    n = int(ms / 1000 * SR)
    mx.feed_passthrough(np.full((n, 2), valor, np.float32))


# --------------------------------------------------------------------------
# 1. Dispositivos
# --------------------------------------------------------------------------

def test_devices() -> None:
    print("\n== 1. Dispositivos ==")
    loop = list_loopback_devices()
    print(f"  loopback ({len(loop)}):")
    for name in loop:
        print(f"    - {name}")
    outs = list_output_devices()
    print(f"  saída ({len(outs)}):")
    for idx, name in outs:
        print(f"    [{idx}] {name}")
    print(f"  speaker padrão: {default_loopback_name()!r}")
    print(f"  VB-CABLE: {find_cable_device()!r}")
    record("listagem de dispositivos", True,
           f"{len(loop)} loopback / {len(outs)} saída")


# --------------------------------------------------------------------------
# 2. Lógica do mixer (offline, determinística, não precisa de dispositivo)
# --------------------------------------------------------------------------

def test_mixer_logic() -> None:
    print("\n== 2. Lógica do mixer (offline) ==")
    mx = OutputMixer(samplerate=SR, channels=2,
                     gain_original=0.30, gain_tts=1.0, duck_level=0.15)

    # 2a. underrun com fila vazia -> zeros, sem exceção
    blk = mx.render_block(BLOCKSIZE)
    record("underrun devolve zeros", blk.shape == (BLOCKSIZE, 2) and not blk.any(),
           f"shape={blk.shape}")

    # 2b. normalização de canais
    ok = True
    for pcm in (np.zeros(480, np.float32), np.zeros((480, 1), np.float32),
                np.zeros((480, 2), np.float32), np.zeros((480, 6), np.float32)):
        mx.feed_passthrough(pcm)
        ok = ok and mx.render_block(480).shape == (480, 2)
    record("feed_passthrough aceita (n,), (n,1), (n,2), (n,6)", ok)

    # 2c. teto do ring (~200 ms): 1 s alimentado não pode virar 1 s de atraso
    for _ in range(50):
        mx.feed_passthrough(np.ones((BLOCKSIZE, 2), np.float32) * 0.1)
    held = mx._pass_remaining / SR
    record("ring de passthrough limitado a ~200 ms", held <= 0.21,
           f"{held * 1000:.0f} ms retidos")

    # 2d. rampa de ducking: passthrough DC + TTS silencioso => a saída É o envelope
    mx2 = OutputMixer(samplerate=SR, channels=2,
                      gain_original=0.30, gain_tts=1.0, duck_level=0.15)
    env: list[np.ndarray] = []
    prime(mx2)                                # colchão inicial
    for i in range(40):                       # 40 blocos = 800 ms
        mx2.feed_passthrough(np.ones((BLOCKSIZE, 2), np.float32))
        if i == 5:                            # TTS mudo: aciona o duck sem somar sinal
            mx2.enqueue_tts(np.zeros(int(0.20 * SR), np.float32))
        env.append(mx2.render_block(BLOCKSIZE)[:, 0])
    curve = np.concatenate(env)
    jump = float(np.abs(np.diff(curve)).max())
    limite = 1.0 / (0.05 * SR) * 1.5          # rampa de 50 ms + folga
    record("ducking sem degrau (rampa suave)", jump <= limite,
           f"maior salto por amostra = {jump:.2e} (limite {limite:.2e})")
    record("ducking atinge duck_level", abs(curve.min() - 0.15) < 0.01,
           f"mínimo {curve.min():.3f}")
    record("passthrough volta a gain_original", abs(curve[-1] - 0.30) < 0.01,
           f"final {curve[-1]:.3f}")
    desceu = np.where(curve < 0.30 - 1e-4)[0]
    dur = int(np.argmin(curve[desceu[0]:])) if desceu.size else 0
    record("transição dura ~7 ms (não é degrau)", dur > 200, f"{dur} amostras")

    # 2e. clipping
    mx3 = OutputMixer(samplerate=SR, channels=2, gain_original=1.0,
                      gain_tts=1.5, duck_level=1.0)
    prime(mx3)
    mx3.enqueue_tts(np.ones(BLOCKSIZE, np.float32))
    out = mx3.render_block(BLOCKSIZE)
    record("saída clipada em [-1, 1]", float(out.max()) <= 1.0 and float(out.min()) >= -1.0,
           f"max={out.max():.3f}")

    # 2f. colchão inicial: só toca depois de acumular ~60 ms
    mx5 = OutputMixer(samplerate=SR, channels=2, gain_original=1.0)
    mx5.feed_passthrough(np.ones((BLOCKSIZE, 2), np.float32))   # 20 ms < colchão
    mudo = not mx5.render_block(BLOCKSIZE).any()
    prime(mx5)                                                   # +120 ms
    soa = mx5.render_block(BLOCKSIZE).any()
    record("colchão de ~60 ms antes de tocar o passthrough", mudo and bool(soa))

    # 2g. backlog / is_tts_active / clear_tts
    mx4 = OutputMixer(samplerate=SR)
    mx4.enqueue_tts(tone(440, 1.0))
    b0 = mx4.tts_backlog_seconds()
    mx4.render_block(SR // 2)
    b1 = mx4.tts_backlog_seconds()
    ok = abs(b0 - 1.0) < 0.01 and abs(b1 - 0.5) < 0.01 and mx4.is_tts_active()
    record("tts_backlog_seconds decresce", ok, f"{b0:.2f}s -> {b1:.2f}s")
    mx4.clear_tts()
    resto = mx4.tts_backlog_seconds()
    record("clear_tts esvazia (só o fade de 10 ms)", resto <= 0.011,
           f"resto {resto * 1000:.1f} ms")
    mx4.render_block(SR // 10)
    record("is_tts_active falso após consumir tudo", not mx4.is_tts_active())


# --------------------------------------------------------------------------
# 3. Reprodução real
# --------------------------------------------------------------------------

def _pump_passthrough(mx: OutputMixer, pcm_mono: np.ndarray, stop_at: float,
                      alvo_s: float = 0.10, on_tick=None) -> None:
    """Alimenta o passthrough mantendo ~`alvo_s` em buffer (como faria a captura)."""
    pos = 0
    while time.monotonic() < stop_at:
        while mx.passthrough_seconds() < alvo_s:
            if pos + BLOCKSIZE > pcm_mono.size:
                pos = 0                        # repete o material
            mx.feed_passthrough(pcm_mono[pos:pos + BLOCKSIZE])
            pos += BLOCKSIZE
        if on_tick is not None:
            on_tick()
        time.sleep(0.005)


def test_playback() -> None:
    print("\n== 3. Reprodução real (você deve ouvir) ==")
    if OFFLINE:
        record("reprodução", False, "modo --offline", skip=True)
        return
    outs = list_output_devices()
    if not outs:
        record("reprodução", False, "nenhum dispositivo de saída", skip=True)
        return
    try:
        mx = OutputMixer(samplerate=SR, channels=2, gain_original=0.15,
                         gain_tts=0.8, duck_level=0.05)
        mx.start()
    except Exception as exc:
        record("abrir OutputStream", False, repr(exc))
        return
    record("abrir OutputStream", True)

    try:
        # 3a. TTS: 1 s de 440 Hz pelo caminho enqueue_tts
        print("  tocando 1 s de 440 Hz via enqueue_tts…")
        mx.enqueue_tts(tone(440, 1.0, amp=0.4))
        t0 = time.monotonic()
        while mx.is_tts_active() and time.monotonic() - t0 < 3.0:
            time.sleep(0.05)
        dur = time.monotonic() - t0
        record("tom de 440 Hz reproduzido via enqueue_tts", 0.8 <= dur <= 2.0,
               f"drenou em {dur:.2f}s")
        time.sleep(0.3)

        # 3b. passthrough com ruído a ganho baixo
        print("  tocando 1,5 s de ruído via feed_passthrough (ganho 0.15)…")
        base = mx.underruns
        _pump_passthrough(mx, noise(2.0, 0.25), time.monotonic() + 1.5)
        un_b = mx.underruns - base
        record("passthrough de ruído sem underrun", un_b <= 2,
               f"{un_b} underrun(s) em 1,5 s; buffer {mx.passthrough_seconds()*1000:.0f} ms")
        time.sleep(0.3)

        # 3c. ducking: ruído contínuo + tom no meio
        print("  ducking: ruído contínuo + tom de 440 Hz a partir de 0,7 s…")
        base = mx.underruns
        inicio = time.monotonic()
        estado = {"disparado": False}

        def gatilho() -> None:
            if not estado["disparado"] and time.monotonic() - inicio > 0.7:
                mx.enqueue_tts(tone(440, 1.0, amp=0.4))
                estado["disparado"] = True

        _pump_passthrough(mx, noise(2.0, 0.25), inicio + 2.6, on_tick=gatilho)
        un_c = mx.underruns - base
        record("ducking simultâneo sem exceção/underrun",
               estado["disparado"] and not mx.is_tts_active() and un_c <= 2,
               f"{un_c} underrun(s) em 2,6 s")

        # 3d. clear_tts durante reprodução
        mx.enqueue_tts(tone(660, 2.0, amp=0.4))
        time.sleep(0.3)
        mx.clear_tts()
        time.sleep(0.2)
        record("clear_tts corta a fala em andamento", not mx.is_tts_active())
    finally:
        mx.stop()
        record("mixer parado", True)


# --------------------------------------------------------------------------
# 4. Captura de loopback
# --------------------------------------------------------------------------

def test_capture() -> None:
    print("\n== 4. Captura de loopback (2 s) ==")
    if OFFLINE:
        record("captura", False, "modo --offline", skip=True)
        return
    if not list_loopback_devices():
        record("captura", False, "nenhum dispositivo de loopback", skip=True)
        return

    blocos: list[np.ndarray] = []
    erros: list[Exception] = []
    cap = LoopbackCapture(device_hint=None, samplerate=SR, blocksize=BLOCKSIZE)
    cap.start(on_block=lambda pcm, sr: blocos.append(pcm),
              on_error=erros.append)
    time.sleep(2.0)
    cap.stop()

    if erros:
        record("captura sem erros", False, f"{len(erros)} erro(s): {erros[0]!r}")
    else:
        record("captura sem erros", True, f"dispositivo: {cap.device_name!r}")

    if not blocos:
        record("blocos recebidos", False, "nenhum bloco em 2 s")
        return
    dados = np.concatenate(blocos, axis=0)
    segundos = dados.shape[0] / SR
    rms = float(np.sqrt(np.mean(np.square(dados, dtype=np.float64))))
    dbfs = 20 * np.log10(rms) if rms > 0 else -np.inf
    print(f"  {len(blocos)} blocos, {dados.shape[1]} canais, {segundos:.2f}s")
    print(f"  RMS = {rms:.6f} ({dbfs:.1f} dBFS){'  [silêncio, nada tocando]' if rms < 1e-5 else ''}")
    record("~2 s capturados a 48 kHz", 1.5 <= segundos <= 2.6, f"{segundos:.2f}s")
    record("dtype float32 e faixa [-1,1]",
           dados.dtype == np.float32 and float(np.abs(dados).max()) <= 1.0,
           f"pico {float(np.abs(dados).max()):.3f}")


# --------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    print("=" * 68)
    print("Teste do módulo de áudio: tradutor.audio_io" +
          ("  [OFFLINE]" if OFFLINE else ""))
    print("=" * 68)
    for fn in (test_devices, test_mixer_logic, test_playback, test_capture):
        try:
            fn()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            record(fn.__name__, False, f"exceção: {exc!r}")

    print("\n" + "=" * 68)
    falhas = [r for r in RESULTS if r[1] == "FALHOU"]
    skips = [r for r in RESULTS if r[1] == "SKIP"]
    print(f"RESUMO: {len(RESULTS) - len(falhas) - len(skips)} ok, "
          f"{len(falhas)} falha(s), {len(skips)} pulado(s)")
    for nome, _s, det in falhas:
        print(f"  FALHOU: {nome}: {det}")
    print("=" * 68)
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
