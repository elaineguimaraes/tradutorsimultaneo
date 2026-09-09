"""Interface gráfica do Tradutor Simultâneo (Tkinter, roda na MAIN thread).

Implementa `GuiProtocol` de `contracts.py`:

* **Painel de controle**: janela normal (~420x530) com Iniciar/Pausar, "Ao vivo",
  três sliders de volume, combo de tema do glossário, combos de dispositivo,
  linha de status ao vivo e controles da legenda (tamanho de fonte,
  click-through, texto original).
* **Overlay de legenda**: `Toplevel` sem borda, sempre no topo, semitransparente,
  arrastável, com auto-hide por fade após 6 s sem legenda nova.

A GUI só conhece `contracts`, `config`, `glossary` (lista de temas) e
`tkinter`; nada de áudio/ASR aqui.
"""

from __future__ import annotations

import ctypes
import logging
import os
import queue
import tkinter as tk
from ctypes import wintypes
from tkinter import ttk
from typing import Optional

from .config import AppConfig
from .contracts import ControllerProtocol, UiState
from .glossary import list_themes

log = logging.getLogger("tradutor.gui")

# --------------------------------------------------------------------------
# Constantes de aparência / comportamento
# --------------------------------------------------------------------------

OVERLAY_BG = "#111111"
OVERLAY_ALPHA = 0.88
OVERLAY_WIDTH_RATIO = 0.70
OVERLAY_BOTTOM_MARGIN = 200          # px acima da borda inferior da tela

COLOR_ORIGINAL = "#cfcfcf"           # cinza-claro (texto original)
COLOR_PREVIOUS = "#8a8a8a"           # cinza médio (tradução anterior)
COLOR_CURRENT = "#ffffff"            # branco (tradução atual)

COLOR_START = "#1f7a34"              # verde: pronto para iniciar
COLOR_STOP = "#a52222"               # vermelho: rodando (clique pausa)
COLOR_STATUS_OK = "#0b6b2f"
COLOR_STATUS_IDLE = "#555555"
COLOR_STATUS_ERROR = "#c62828"

FONT_MIN = 8
FONT_MAX = 40

POLL_SUBTITLE_MS = 100               # consumo da fila de legendas
POLL_STATE_MS = 500                  # polling de get_ui_state()
HIDE_DELAY_MS = 6000                 # inatividade até começar o fade
FADE_STEPS = 10
FADE_INTERVAL_MS = 45

CAPTURE_DEFAULT_LABEL = "(Saída padrão do sistema)"
OUTPUT_DEFAULT_LABEL = "(Dispositivo padrão do sistema)"

_PLACEHOLDER = "Legenda aparecerá aqui. Arraste esta janela para posicioná-la."


# --------------------------------------------------------------------------
# Click-through (Windows): WS_EX_LAYERED | WS_EX_TRANSPARENT
# --------------------------------------------------------------------------

_GWL_EXSTYLE = -20
_WS_EX_LAYERED = 0x00080000
_WS_EX_TRANSPARENT = 0x00000020
_GA_ROOT = 2
_SWP_NOSIZE = 0x0001
_SWP_NOMOVE = 0x0002
_SWP_NOZORDER = 0x0004
_SWP_NOACTIVATE = 0x0010
_SWP_FRAMECHANGED = 0x0020


def _hwnd_of(widget: tk.Misc) -> int:
    """Devolve o HWND da janela de topo que contém `widget`.

    Usa `GetAncestor(GA_ROOT)` em vez de `GetParent`, pois este último devolveria o
    *owner* (a janela de controle) para Toplevels popup, o que aplicaria o
    estilo na janela errada.
    """
    wid = int(widget.winfo_id())
    try:
        user32 = ctypes.windll.user32
        user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        user32.GetAncestor.restype = wintypes.HWND
        root = user32.GetAncestor(wintypes.HWND(wid), _GA_ROOT)
        return int(root) if root else wid
    except Exception:  # pragma: no cover - só em plataforma não-Windows
        return wid


def _apply_click_through(widget: tk.Misc, enabled: bool) -> bool:
    """Liga/desliga o "clique atravessa" na janela de topo de `widget`.

    Devolve True se conseguiu aplicar; em falha loga warning e devolve False
    (o app continua funcionando, apenas sem click-through).
    """
    try:
        user32 = ctypes.windll.user32
        hwnd = wintypes.HWND(_hwnd_of(widget))

        # LONG_PTR: 64 bits no Python x64. Em 32 bits caímos no GetWindowLongW.
        if hasattr(user32, "GetWindowLongPtrW"):
            getter, setter = user32.GetWindowLongPtrW, user32.SetWindowLongPtrW
            long_ptr = ctypes.c_longlong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_long
        else:  # pragma: no cover - Windows 32 bits
            getter, setter = user32.GetWindowLongW, user32.SetWindowLongW
            long_ptr = ctypes.c_long

        getter.argtypes = [wintypes.HWND, ctypes.c_int]
        getter.restype = long_ptr
        setter.argtypes = [wintypes.HWND, ctypes.c_int, long_ptr]
        setter.restype = long_ptr

        style = int(getter(hwnd, _GWL_EXSTYLE))
        if enabled:
            new_style = style | _WS_EX_LAYERED | _WS_EX_TRANSPARENT
        else:
            # Mantém LAYERED (necessário para o -alpha), remove só TRANSPARENT.
            new_style = (style | _WS_EX_LAYERED) & ~_WS_EX_TRANSPARENT
        if new_style != style:
            ctypes.set_last_error(0)
            setter(hwnd, _GWL_EXSTYLE, long_ptr(new_style))
            err = ctypes.get_last_error()
            if err:
                raise OSError(err, ctypes.FormatError(err))

        user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                        wintypes.UINT]
        user32.SetWindowPos(hwnd, None, 0, 0, 0, 0,
                            _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOZORDER
                            | _SWP_NOACTIVATE | _SWP_FRAMECHANGED)
        log.debug("Click-through %s (hwnd=%s)", "ativado" if enabled else "desativado", hwnd.value)
        return True
    except Exception as exc:
        log.warning("Não foi possível aplicar click-through: %s", exc)
        return False


# --------------------------------------------------------------------------
# Tooltip minimalista
# --------------------------------------------------------------------------

class _Tooltip:
    """Balãozinho de ajuda exibido ao pousar o mouse sobre um widget."""

    def __init__(self, widget: tk.Widget, text: str, delay_ms: int = 450) -> None:
        self._widget = widget
        self._text = text
        self._delay = delay_ms
        self._after_id: Optional[str] = None
        self._tip: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event: object = None) -> None:
        self._cancel()
        self._after_id = self._widget.after(self._delay, self._show)

    def _cancel(self) -> None:
        if self._after_id is not None:
            try:
                self._widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self) -> None:
        if self._tip is not None:
            return
        x = self._widget.winfo_rootx() + 12
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 6
        tip = tk.Toplevel(self._widget)
        tip.overrideredirect(True)
        tip.attributes("-topmost", True)
        tk.Label(tip, text=self._text, bg="#ffffe0", fg="#222222",
                 relief="solid", borderwidth=1, font=("Segoe UI", 9),
                 padx=6, pady=3).pack()
        tip.geometry(f"+{x}+{y}")
        self._tip = tip

    def _hide(self, _event: object = None) -> None:
        self._cancel()
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None


# --------------------------------------------------------------------------
# Overlay de legenda
# --------------------------------------------------------------------------

class _SubtitleOverlay:
    """Janela sem borda com a legenda: original, tradução anterior e atual."""

    def __init__(self, master: tk.Tk, config: AppConfig) -> None:
        self._config = config
        self._font_size = max(FONT_MIN, min(FONT_MAX, int(config.subtitle_font_size)))
        self._show_original = True
        self._click_through = False

        win = tk.Toplevel(master)
        win.withdraw()
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.attributes("-alpha", OVERLAY_ALPHA)
        win.configure(bg=OVERLAY_BG)
        self.win = win

        screen_w = win.winfo_screenwidth()
        screen_h = win.winfo_screenheight()
        self._width = max(320, int(screen_w * OVERLAY_WIDTH_RATIO))
        wrap = self._width - 40

        frame = tk.Frame(win, bg=OVERLAY_BG, padx=16, pady=10)
        frame.pack(fill="both", expand=True)
        # Espaçador força a largura (~70% da tela); a altura fica automática.
        tk.Frame(frame, bg=OVERLAY_BG, width=wrap, height=1).pack()

        self._lbl_original = tk.Label(
            frame, text="", bg=OVERLAY_BG, fg=COLOR_ORIGINAL,
            wraplength=wrap, justify="center")
        self._lbl_previous = tk.Label(
            frame, text="", bg=OVERLAY_BG, fg=COLOR_PREVIOUS,
            wraplength=wrap, justify="center")
        self._lbl_current = tk.Label(
            frame, text=_PLACEHOLDER, bg=OVERLAY_BG, fg=COLOR_CURRENT,
            wraplength=wrap, justify="center")
        self._lbl_original.pack(fill="x")
        self._lbl_previous.pack(fill="x")
        self._lbl_current.pack(fill="x")
        self._apply_fonts()

        pos = config.subtitle_pos
        if pos and len(pos) == 2:
            x, y = int(pos[0]), int(pos[1])
        else:
            x = (screen_w - self._width) // 2
            y = max(0, screen_h - OVERLAY_BOTTOM_MARGIN)
        win.geometry(f"+{x}+{y}")

        self._drag_dx = 0
        self._drag_dy = 0
        self._bind_drag(win)

        self._fade_after: Optional[str] = None
        self._hide_after: Optional[str] = None
        self._enabled = True
        win.deiconify()
        self.schedule_auto_hide()

    def set_enabled(self, enabled: bool) -> None:
        """Liga/desliga o overlay por completo (checkbox do painel)."""
        self._enabled = bool(enabled)
        if not self._enabled:
            for attr in ("_fade_after", "_hide_after"):
                after_id = getattr(self, attr)
                if after_id is not None:
                    try:
                        self.win.after_cancel(after_id)
                    except Exception:
                        pass
                    setattr(self, attr, None)
            try:
                self.win.withdraw()
            except Exception:
                pass
        # ligado: reaparece sozinho na próxima legenda (update_text)

    # ---------------- fontes -------------------------------------------

    def _apply_fonts(self) -> None:
        """Recalcula as três fontes a partir do tamanho principal."""
        size = self._font_size
        self._lbl_original.configure(font=("Segoe UI", max(7, int(size * 0.6))))
        self._lbl_previous.configure(font=("Segoe UI", max(7, int(size * 0.75))))
        self._lbl_current.configure(font=("Segoe UI", size, "bold"))

    def set_font_size(self, size: int) -> None:
        """Ajusta ao vivo o tamanho da fonte principal (limitado a 8..40)."""
        self._font_size = max(FONT_MIN, min(FONT_MAX, int(size)))
        self._apply_fonts()

    @property
    def font_size(self) -> int:
        return self._font_size

    def set_show_original(self, show: bool) -> None:
        """Mostra/esconde a linha com o texto no idioma original."""
        self._show_original = bool(show)
        if self._show_original:
            self._lbl_original.pack(fill="x", before=self._lbl_previous)
        else:
            self._lbl_original.pack_forget()

    # ---------------- arrasto ------------------------------------------

    def _bind_drag(self, widget: tk.Misc) -> None:
        widget.bind("<Button-1>", self._on_press, add="+")
        widget.bind("<B1-Motion>", self._on_drag, add="+")
        widget.bind("<ButtonRelease-1>", self._on_release, add="+")
        for child in widget.winfo_children():
            self._bind_drag(child)

    def _on_press(self, event: tk.Event) -> None:
        self._drag_dx = event.x_root - self.win.winfo_x()
        self._drag_dy = event.y_root - self.win.winfo_y()

    def _on_drag(self, event: tk.Event) -> None:
        self.win.geometry(f"+{event.x_root - self._drag_dx}+{event.y_root - self._drag_dy}")

    def _on_release(self, _event: tk.Event) -> None:
        self.remember_position()

    def remember_position(self) -> None:
        """Guarda a posição atual do overlay na config em memória."""
        try:
            self._config.subtitle_pos = [self.win.winfo_x(), self.win.winfo_y()]
        except Exception:  # pragma: no cover - janela já destruída
            pass

    # ---------------- click-through -------------------------------------

    def set_click_through(self, enabled: bool) -> bool:
        """Aplica o estilo WS_EX_TRANSPARENT; devolve o estado efetivo."""
        ok = _apply_click_through(self.win, enabled)
        self._click_through = bool(enabled) and ok
        return self._click_through

    # ---------------- conteúdo e auto-hide -------------------------------

    def update_text(self, original: str, translated: str) -> None:
        """Rola o histórico (1 linha) e mostra a nova legenda."""
        if not self._enabled:
            return
        if not (translated or "").strip():
            return  # legenda vazia não acende a caixa nem segura o fade
        current = self._lbl_current.cget("text")
        if current and current != _PLACEHOLDER:
            self._lbl_previous.configure(text=current)
        self._lbl_current.configure(text=translated or "")
        self._lbl_original.configure(text=original or "")
        self.show_now()
        self.schedule_auto_hide()

    def show_now(self) -> None:
        """Cancela qualquer fade em curso e reexibe o overlay opaco."""
        if self._fade_after is not None:
            try:
                self.win.after_cancel(self._fade_after)
            except Exception:
                pass
            self._fade_after = None
        try:
            self.win.attributes("-alpha", OVERLAY_ALPHA)
            if self.win.state() == "withdrawn":
                self.win.deiconify()
                self.win.attributes("-topmost", True)
                if self._click_through:
                    _apply_click_through(self.win, True)
        except tk.TclError:  # pragma: no cover
            pass

    def schedule_auto_hide(self) -> None:
        """(Re)arma o timer de 6 s de inatividade que dispara o fade."""
        if self._hide_after is not None:
            try:
                self.win.after_cancel(self._hide_after)
            except Exception:
                pass
        self._hide_after = self.win.after(HIDE_DELAY_MS, self._start_fade)

    def _start_fade(self) -> None:
        self._hide_after = None
        self._fade_step(FADE_STEPS)

    def _fade_step(self, remaining: int) -> None:
        try:
            if remaining <= 0:
                self.win.attributes("-alpha", 0.0)
                self.win.withdraw()
                self._fade_after = None
                return
            self.win.attributes("-alpha", OVERLAY_ALPHA * remaining / FADE_STEPS)
            self._fade_after = self.win.after(
                FADE_INTERVAL_MS, self._fade_step, remaining - 1)
        except tk.TclError:  # pragma: no cover - janela destruída no meio do fade
            self._fade_after = None

    def destroy(self) -> None:
        """Cancela timers e destrói a janela."""
        for attr in ("_fade_after", "_hide_after"):
            after_id = getattr(self, attr)
            if after_id is not None:
                try:
                    self.win.after_cancel(after_id)
                except Exception:
                    pass
                setattr(self, attr, None)
        try:
            self.win.destroy()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Aplicação (painel de controle + overlay)
# --------------------------------------------------------------------------

class App:
    """GUI completa do tradutor: painel de controle + overlay de legenda."""

    # vozes pt-BR do Edge-TTS (rótulo na janela, id enviado ao serviço)
    _VOICES = (
        ("Feminina (Francisca)", "pt-BR-FranciscaNeural"),
        ("Feminina (Thalita)", "pt-BR-ThalitaMultilingualNeural"),
        ("Masculina (Antônio)", "pt-BR-AntonioNeural"),
    )

    def __init__(self, controller: ControllerProtocol, config: AppConfig) -> None:
        self._controller = controller
        self._config = config
        self._subtitles: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self._running = False
        self._closing = False
        self._themes = list_themes() or [("trading", "Mercado financeiro")]

        self._root = tk.Tk()
        self._root.title("Tradutor Simultâneo")
        self._root.geometry("420x530")
        self._root.minsize(400, 500)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._style = ttk.Style(self._root)
        self._build_widgets()

        self._overlay = _SubtitleOverlay(self._root, config)
        self._overlay.set_show_original(bool(self._var_show_original.get()))
        if not config.show_subtitles:
            self._overlay.set_enabled(False)
        if config.overlay_click_through:
            effective = self._overlay.set_click_through(True)
            self._var_click_through.set(bool(effective))

        self._root.after(POLL_SUBTITLE_MS, self._drain_subtitles)
        self._root.after(POLL_STATE_MS, self._poll_state)

    # ------------------------------------------------------------------
    # Construção da interface
    # ------------------------------------------------------------------

    def _build_widgets(self) -> None:
        cfg = self._config
        outer = ttk.Frame(self._root, padding=10)
        outer.pack(fill="both", expand=True)

        # --- Linha 1: Iniciar/Pausar + Ao vivo -------------------------
        top = ttk.Frame(outer)
        top.pack(fill="x")
        self._btn_toggle = tk.Button(
            top, text="▶  Iniciar", command=self._on_toggle,
            bg=COLOR_START, fg="white", activebackground=COLOR_START,
            activeforeground="white", font=("Segoe UI", 12, "bold"),
            relief="flat", cursor="hand2", height=2, width=16)
        self._btn_toggle.pack(side="left", fill="x", expand=True)

        btn_live = ttk.Button(top, text="⏩ Ao vivo", command=self._on_skip_to_live)
        btn_live.pack(side="left", padx=(8, 0), ipady=8)
        _Tooltip(btn_live, "descarta a fila de voz atrasada")

        # --- Status ----------------------------------------------------
        self._var_status = tk.StringVar(value="⏸ Pausado")
        self._lbl_status = ttk.Label(outer, textvariable=self._var_status,
                                     font=("Segoe UI", 10, "bold"),
                                     foreground=COLOR_STATUS_IDLE, anchor="w")
        self._lbl_status.pack(fill="x", pady=(10, 6))

        ttk.Separator(outer).pack(fill="x", pady=2)

        # --- Sliders ---------------------------------------------------
        sliders = ttk.Frame(outer)
        sliders.pack(fill="x", pady=(6, 2))
        sliders.columnconfigure(1, weight=1)

        self._var_original = tk.DoubleVar(value=round(cfg.gain_original * 100))
        self._var_tts = tk.DoubleVar(value=round(cfg.gain_tts * 100))

        self._lbl_val_original = self._add_slider(
            sliders, 0, "Volume original", self._var_original, 0, 100,
            self._on_gain_original)
        self._lbl_val_tts = self._add_slider(
            sliders, 1, "Volume tradução", self._var_tts, 0, 150,
            self._on_gain_tts)

        self._var_speed = tk.DoubleVar(value=getattr(cfg, "tts_speed", 1.0))
        ttk.Label(sliders, text="Velocidade da voz").grid(
            row=3, column=0, sticky="w", pady=3)
        speed_box = ttk.Frame(sliders)
        speed_box.grid(row=3, column=1, columnspan=2, sticky="w")
        for spd, txt in ((1.0, "1.0x"), (1.25, "1.25x"), (1.5, "1.5x")):
            ttk.Radiobutton(speed_box, text=txt, value=spd,
                            variable=self._var_speed,
                            command=self._on_tts_speed).pack(
                side="left", padx=(0, 10))

        ttk.Label(sliders, text="Voz").grid(
            row=4, column=0, sticky="w", pady=3)
        self._cbo_voice = ttk.Combobox(
            sliders, state="readonly", width=24,
            values=[rotulo for rotulo, _ in self._VOICES])
        atual = getattr(cfg, "tts_voice", self._VOICES[0][1])
        ids = [vid for _, vid in self._VOICES]
        self._cbo_voice.current(ids.index(atual) if atual in ids else 0)
        self._cbo_voice.grid(row=4, column=1, columnspan=2, sticky="w")
        self._cbo_voice.bind("<<ComboboxSelected>>", self._on_voice_selected)

        lbl_theme = ttk.Label(sliders, text="Tema")
        lbl_theme.grid(row=5, column=0, sticky="w", pady=3)
        self._cbo_theme = ttk.Combobox(
            sliders, state="readonly", width=24,
            values=[rotulo for _, rotulo in self._themes])
        nomes_tema = [nome for nome, _ in self._themes]
        atual_tema = getattr(cfg, "glossario", self._themes[0][0])
        if atual_tema not in nomes_tema:
            # tema salvo foi removido da pasta: volta ao padrão de forma explícita,
            # para GUI e pipeline concordarem desde já (e não só no próximo save)
            atual_tema = self._themes[0][0]
            cfg.glossario = atual_tema
        self._cbo_theme.current(nomes_tema.index(atual_tema))
        self._cbo_theme.grid(row=5, column=1, columnspan=2, sticky="w")
        self._cbo_theme.bind("<<ComboboxSelected>>", self._on_theme_selected)
        _Tooltip(self._cbo_theme,
                 "Conjunto de regras de jargão. Novos temas: arquivos em glossarios/")

        ttk.Separator(outer).pack(fill="x", pady=6)

        # --- Dispositivos ---------------------------------------------
        devs = ttk.Frame(outer)
        devs.pack(fill="x")
        devs.columnconfigure(1, weight=1)

        ttk.Label(devs, text="Captura:").grid(row=0, column=0, sticky="w", pady=2)
        self._cbo_capture = ttk.Combobox(devs, state="readonly", width=28)
        self._cbo_capture.grid(row=0, column=1, sticky="ew", padx=(6, 0), pady=2)
        self._cbo_capture.bind("<<ComboboxSelected>>", self._on_capture_selected)

        ttk.Label(devs, text="Saída:").grid(row=1, column=0, sticky="w", pady=2)
        self._cbo_output = ttk.Combobox(devs, state="readonly", width=28)
        self._cbo_output.grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=2)
        self._cbo_output.bind("<<ComboboxSelected>>", self._on_output_selected)
        btn_test = ttk.Button(devs, text="🔊", width=3,
                              command=lambda: self._safe(self._controller.test_output))
        btn_test.grid(row=1, column=2, padx=(4, 0), pady=2)
        _Tooltip(btn_test, "toca um bip na saída selecionada")

        self._output_items: list[tuple[int, str]] = []
        self._populate_devices()

        ttk.Separator(outer).pack(fill="x", pady=6)

        # --- Controles da legenda -------------------------------------
        subs = ttk.Frame(outer)
        subs.pack(fill="x")
        ttk.Label(subs, text="Legenda:").pack(side="left")
        ttk.Button(subs, text="A-", width=4,
                   command=lambda: self._change_font(-2)).pack(side="left", padx=(6, 2))
        ttk.Button(subs, text="A+", width=4,
                   command=lambda: self._change_font(+2)).pack(side="left", padx=2)
        self._var_font = tk.StringVar(value=f"{cfg.subtitle_font_size} pt")
        ttk.Label(subs, textvariable=self._var_font, width=6).pack(side="left", padx=(6, 0))

        self._var_click_through = tk.BooleanVar(value=bool(cfg.overlay_click_through))
        chk_ct = ttk.Checkbutton(outer, text="Clique atravessa a legenda",
                                 variable=self._var_click_through,
                                 command=self._on_click_through)
        chk_ct.pack(fill="x", pady=(6, 0))
        _Tooltip(chk_ct, "o mouse passa direto pela legenda (e o arrasto para de funcionar)")

        self._var_show_original = tk.BooleanVar(value=True)
        ttk.Checkbutton(outer, text="Mostrar texto original",
                        variable=self._var_show_original,
                        command=self._on_show_original).pack(fill="x")

        self._var_show_overlay = tk.BooleanVar(value=bool(cfg.show_subtitles))
        ttk.Checkbutton(outer, text="Mostrar legenda flutuante",
                        variable=self._var_show_overlay,
                        command=self._on_show_overlay).pack(fill="x")

    def _add_slider(self, parent: ttk.Frame, row: int, label: str,
                    var: tk.DoubleVar, lo: int, hi: int, callback) -> ttk.Label:
        """Cria rótulo + ttk.Scale + leitura numérica; devolve o label do valor."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        scale = ttk.Scale(parent, from_=lo, to=hi, orient="horizontal",
                          variable=var, command=callback)
        scale.grid(row=row, column=1, sticky="ew", padx=6, pady=3)
        value_lbl = ttk.Label(parent, text=f"{int(var.get())}", width=4, anchor="e")
        value_lbl.grid(row=row, column=2, sticky="e", pady=3)
        return value_lbl

    def _populate_devices(self) -> None:
        """Preenche os combos consultando o controller (tolerante a erro)."""
        cfg = self._config
        try:
            captures = list(self._controller.list_capture_devices())
        except Exception as exc:
            log.warning("Falha ao listar dispositivos de captura: %s", exc)
            captures = []
        self._cbo_capture["values"] = [CAPTURE_DEFAULT_LABEL] + captures
        hint = cfg.capture_device_hint
        self._cbo_capture.current(
            captures.index(hint) + 1 if hint in captures else 0)

        try:
            outputs = list(self._controller.list_output_devices())
        except Exception as exc:
            log.warning("Falha ao listar dispositivos de saída: %s", exc)
            outputs = []
        self._output_items = outputs
        self._cbo_output["values"] = [OUTPUT_DEFAULT_LABEL] + [n for _, n in outputs]
        # Seleciona pelo NOME salvo (índices mudam entre máquinas/boots).
        names = [n for _, n in outputs]
        self._cbo_output.current(
            names.index(cfg.output_device_name) + 1
            if cfg.output_device_name in names else 0)

    # ------------------------------------------------------------------
    # Callbacks do painel
    # ------------------------------------------------------------------

    def _on_toggle(self) -> None:
        """Alterna entre iniciar e pausar o pipeline."""
        try:
            if self._running:
                self._controller.stop_pipeline()
            else:
                self._controller.start_pipeline()
        except Exception as exc:
            log.exception("Erro ao alternar o pipeline: %s", exc)
            self._set_status(f"erro: {exc}", error=True)
            return
        self._running = not self._running
        self._refresh_toggle_button()

    def _refresh_toggle_button(self) -> None:
        if self._running:
            self._btn_toggle.configure(text="⏸  Pausar", bg=COLOR_STOP,
                                       activebackground=COLOR_STOP)
        else:
            self._btn_toggle.configure(text="▶  Iniciar", bg=COLOR_START,
                                       activebackground=COLOR_START)

    def _on_skip_to_live(self) -> None:
        try:
            self._controller.skip_to_live()
        except Exception as exc:
            log.exception("Erro em skip_to_live: %s", exc)

    def _on_gain_original(self, _value: str) -> None:
        v = int(round(self._var_original.get()))
        self._lbl_val_original.configure(text=str(v))
        self._safe(self._controller.set_gain_original, v / 100.0)

    def _on_gain_tts(self, _value: str) -> None:
        v = int(round(self._var_tts.get()))
        self._lbl_val_tts.configure(text=str(v))
        self._safe(self._controller.set_gain_tts, v / 100.0)

    def _on_tts_speed(self) -> None:
        self._safe(self._controller.set_tts_speed,
                   float(self._var_speed.get()))

    def _on_voice_selected(self, _event: object = None) -> None:
        idx = max(0, self._cbo_voice.current())
        self._safe(self._controller.set_tts_voice, self._VOICES[idx][1])

    def _on_theme_selected(self, _event: object = None) -> None:
        idx = max(0, self._cbo_theme.current())
        self._safe(self._controller.set_glossary, self._themes[idx][0])

    def _on_capture_selected(self, _event: object = None) -> None:
        """Item 0 = padrão do sistema, enviado como string vazia."""
        idx = self._cbo_capture.current()
        name = "" if idx <= 0 else self._cbo_capture.get()
        self._config.capture_device_hint = name or None
        self._safe(self._controller.set_capture_device, name)

    def _on_output_selected(self, _event: object = None) -> None:
        """Item 0 = padrão do sistema, enviado como índice -1."""
        idx = self._cbo_output.current()
        if idx <= 0:
            device_index = -1
            self._config.output_device = None
        else:
            device_index = self._output_items[idx - 1][0]
            self._config.output_device = device_index
        self._safe(self._controller.set_output_device, device_index)

    def _change_font(self, delta: int) -> None:
        new_size = max(FONT_MIN, min(FONT_MAX, self._overlay.font_size + delta))
        self._overlay.set_font_size(new_size)
        self._config.subtitle_font_size = new_size
        self._var_font.set(f"{new_size} pt")

    def _on_click_through(self) -> None:
        wanted = bool(self._var_click_through.get())
        effective = self._overlay.set_click_through(wanted)
        if wanted and not effective:
            self._var_click_through.set(False)
            self._set_status("erro: click-through indisponível nesta janela", error=True)
        self._config.overlay_click_through = effective

    def _on_show_original(self) -> None:
        self._overlay.set_show_original(bool(self._var_show_original.get()))

    def _on_show_overlay(self) -> None:
        self._overlay.set_enabled(bool(self._var_show_overlay.get()))

    def _safe(self, func, *args) -> None:
        """Chama o controller sem deixar exceção derrubar o loop Tk."""
        try:
            func(*args)
        except Exception as exc:
            log.exception("Erro ao chamar %s: %s", getattr(func, "__name__", func), exc)

    # ------------------------------------------------------------------
    # Loops periódicos
    # ------------------------------------------------------------------

    def _drain_subtitles(self) -> None:
        """Consome a fila de legendas na main thread e atualiza o overlay."""
        if self._closing:
            return
        try:
            while True:
                original, translated = self._subtitles.get_nowait()
                self._overlay.update_text(original, translated)
        except queue.Empty:
            pass
        except Exception as exc:  # pragma: no cover
            log.exception("Erro ao atualizar a legenda: %s", exc)
        finally:
            if not self._closing:
                self._root.after(POLL_SUBTITLE_MS, self._drain_subtitles)

    def _poll_state(self) -> None:
        """Lê `get_ui_state()` a cada 500 ms e atualiza a linha de status."""
        if self._closing:
            return
        try:
            state = self._controller.get_ui_state()
            self._render_state(state)
        except Exception as exc:
            log.warning("Falha em get_ui_state(): %s", exc)
            self._set_status(f"erro: {exc}", error=True)
        finally:
            if not self._closing:
                self._root.after(POLL_STATE_MS, self._poll_state)

    def _render_state(self, state: UiState) -> None:
        if state.running != self._running:
            self._running = bool(state.running)
            self._refresh_toggle_button()

        status = (state.status or "").strip()
        if status.lower().startswith("erro"):
            self._set_status(status, error=True)
            return
        if not state.running:
            # Mostra mensagens de progresso (ex.: "carregando modelos…")
            # mesmo antes de o pipeline entrar em execução.
            self._set_status(status if status else "⏸ Pausado")
            return

        lang = (state.detected_lang or "??").upper()
        atraso = f"{state.backlog_seconds:.1f}".replace(".", ",")
        texto = f"● Ouvindo ({lang}) · atraso {atraso}s"
        if state.rate_pct:
            texto += f" · voz +{int(state.rate_pct)}%"
        self._set_status(texto, running=True)

    def _set_status(self, text: str, running: bool = False, error: bool = False) -> None:
        self._var_status.set(text)
        color = COLOR_STATUS_ERROR if error else (
            COLOR_STATUS_OK if running else COLOR_STATUS_IDLE)
        self._lbl_status.configure(foreground=color)

    # ------------------------------------------------------------------
    # API pública (GuiProtocol)
    # ------------------------------------------------------------------

    def push_subtitle(self, original: str, translated: str) -> None:
        """Enfileira uma legenda. Seguro para chamar de qualquer thread."""
        self._subtitles.put((original, translated))

    def run(self) -> None:
        """Bloqueia no mainloop do Tk até a janela de controle ser fechada."""
        if os.environ.get("TRADUTOR_GUI_AUTOCLOSE") == "1":
            log.info("TRADUTOR_GUI_AUTOCLOSE=1: fechando em 6 s.")
            self._root.after(6000, lambda: self._shutdown(save=False))
        self._root.mainloop()

    # ------------------------------------------------------------------
    # Encerramento
    # ------------------------------------------------------------------

    def _on_close(self) -> None:
        self._shutdown(save=True)

    def _shutdown(self, save: bool) -> None:
        """Para o pipeline, persiste a config e destrói as janelas."""
        if self._closing:
            return
        self._closing = True
        self._safe(self._controller.stop_pipeline)
        self._overlay.remember_position()
        self._collect_config()
        if save:
            try:
                self._config.save()
                log.info("Configuração salva.")
            except Exception as exc:
                log.warning("Falha ao salvar a configuração: %s", exc)
        # `stop_pipeline` retorna na hora; o desmonte roda em paralelo ao
        # salvamento acima. Damos um tempo curto para o áudio fechar direito
        # e seguimos de qualquer jeito, pois a janela não pode ficar pendurada.
        wait = getattr(self._controller, "wait_stopped", None)
        if wait is not None:
            self._safe(wait, 2.5)
        self._overlay.destroy()
        try:
            self._root.destroy()
        except Exception:
            pass

    def _collect_config(self) -> None:
        """Copia o estado atual dos controles para o AppConfig."""
        cfg = self._config
        cfg.gain_original = round(self._var_original.get() / 100.0, 3)
        cfg.gain_tts = round(self._var_tts.get() / 100.0, 3)
        cfg.tts_speed = float(self._var_speed.get())
        cfg.tts_voice = self._VOICES[max(0, self._cbo_voice.current())][1]
        cfg.glossario = self._themes[max(0, self._cbo_theme.current())][0]
        cfg.subtitle_font_size = self._overlay.font_size
        cfg.overlay_click_through = bool(self._var_click_through.get())
        cfg.show_subtitles = bool(self._var_show_overlay.get())
