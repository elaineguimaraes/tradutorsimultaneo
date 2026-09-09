"""Revisão visual manual da GUI com um controller falso.

Uso:
    py tests\\test_gui.py

Abre o painel de controle + overlay de legenda com dados falsos. Um gerador em
thread separada empurra legendas a cada 2 s, alternando textos curtos e longos.
Todas as chamadas do controller são impressas no console.

Variável de ambiente `TRADUTOR_GUI_AUTOCLOSE=1` fecha a janela sozinha após 6 s
(usado na validação automatizada).
"""

from __future__ import annotations

import logging
import os
import random
import sys
import threading
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tradutor.config import AppConfig            # noqa: E402
from tradutor.contracts import UiState           # noqa: E402
from tradutor.gui import App                     # noqa: E402


# --------------------------------------------------------------------------
# Frases de exemplo (curtas e longas, alternadas)
# --------------------------------------------------------------------------

FRASES_CURTAS = [
    ("The Fed holds rates.", "O Fed mantém as taxas."),
    ("Markets are up.", "Os mercados estão em alta."),
    ("Oil slipped again.", "O petróleo caiu de novo."),
    ("Watch the dollar.", "Fique de olho no dólar."),
]

FRASES_LONGAS = [
    ("The committee decided to keep the target range for the federal funds rate "
     "unchanged, and signaled that further tightening would depend on incoming "
     "inflation and labor market data over the next several meetings.",
     "O comitê decidiu manter inalterada a faixa-alvo da taxa dos fundos federais "
     "e sinalizou que um aperto adicional dependerá dos dados de inflação e do "
     "mercado de trabalho que chegarem nas próximas reuniões."),
    ("Equity futures pared earlier gains after the retail sales print came in "
     "hotter than expected, pushing two-year yields to the highest level since "
     "the beginning of the quarter and reviving the stronger-for-longer debate.",
     "Os futuros de ações reduziram os ganhos iniciais depois que o dado de vendas "
     "no varejo veio mais forte que o esperado, levando os juros de dois anos ao "
     "maior nível desde o início do trimestre e reacendendo o debate sobre juros "
     "altos por mais tempo."),
]


class FakeController:
    """Controller de mentira: imprime as chamadas e devolve valores dummy."""

    def __init__(self) -> None:
        self._running = False
        self._backlog = 0.0
        self._rate = 0
        self._lang = "en"
        self._app: App | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- ligação com a GUI -------------------------------------------

    def attach(self, app: App) -> None:
        """Guarda a GUI e liga o gerador de legendas de exemplo."""
        self._app = app
        self._thread = threading.Thread(target=self._feeder, daemon=True)
        self._thread.start()

    def _feeder(self) -> None:
        i = 0
        while not self._stop.wait(2.0):
            if self._app is None:
                continue
            fonte = FRASES_CURTAS if i % 2 == 0 else FRASES_LONGAS
            original, traduzido = fonte[(i // 2) % len(fonte)]
            print(f"[fake] push_subtitle #{i}: {traduzido[:48]}…")
            self._app.push_subtitle(original, traduzido)
            # Simula variação de atraso/aceleração para a linha de status.
            self._backlog = round(random.uniform(0.4, 9.5), 1)
            self._rate = 0 if self._backlog < 4 else (10 if self._backlog < 8 else 25)
            i += 1

    # ---- ControllerProtocol ------------------------------------------

    def start_pipeline(self) -> None:
        print("[fake] start_pipeline()")
        self._running = True

    def stop_pipeline(self) -> None:
        print("[fake] stop_pipeline()")
        self._running = False

    def set_gain_original(self, value: float) -> None:
        print(f"[fake] set_gain_original({value:.2f})")

    def set_gain_tts(self, value: float) -> None:
        print(f"[fake] set_gain_tts({value:.2f})")

    def set_duck_level(self, value: float) -> None:
        print(f"[fake] set_duck_level({value:.2f})")

    def set_capture_device(self, name: str) -> None:
        print(f"[fake] set_capture_device({name!r})")

    def set_output_device(self, index: int) -> None:
        print(f"[fake] set_output_device({index})")

    def skip_to_live(self) -> None:
        print("[fake] skip_to_live(): fila de voz descartada")
        self._backlog = 0.0
        self._rate = 0

    def get_ui_state(self) -> UiState:
        return UiState(
            running=self._running,
            detected_lang=self._lang,
            backlog_seconds=self._backlog if self._running else 0.0,
            rate_pct=self._rate if self._running else 0,
            status="ouvindo…" if self._running else "",
        )

    def list_capture_devices(self) -> list[str]:
        return ["CABLE Output (VB-Audio Virtual Cable)",
                "Alto-falantes (Realtek(R) Audio)",
                "Fones de ouvido (Bluetooth)"]

    def list_output_devices(self) -> list[tuple[int, str]]:
        return [(3, "Alto-falantes (Realtek(R) Audio)"),
                (5, "CABLE Input (VB-Audio Virtual Cable)"),
                (7, "Monitor DELL (NVIDIA High Definition Audio)")]


def main() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = AppConfig()
    # Não escrever config.json de verdade durante a revisão visual.
    config.save = lambda path="config.json": print(f"[fake] config.save({path!r}) -> ignorado")

    controller = FakeController()
    app = App(controller, config)
    controller.attach(app)

    if os.environ.get("TRADUTOR_GUI_AUTOCLOSE") == "1":
        print("[teste] TRADUTOR_GUI_AUTOCLOSE=1: a janela fecha sozinha em 6 s.")
    else:
        print("[teste] GUI aberta. Feche a janela 'Tradutor Simultâneo' para sair.")

    inicio = time.time()
    app.run()
    print(f"[teste] mainloop encerrado após {time.time() - inicio:.1f}s sem exceções.")


if __name__ == "__main__":
    main()
