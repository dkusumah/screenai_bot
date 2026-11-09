"""ScreenAI Bot — a desktop bot that reads text from a screen region (OCR),
sends it to an AI API, and delivers the answer back into a chosen screen area
by clicking and/or typing.

Modules
-------
config      Typed access to ``config.json`` (AI, OCR, actions, hotkeys, prompt).
capture     Screen capture via ``mss`` (monitors, virtual-desktop coordinates).
ocr         Tesseract OCR wrapper with preprocessing + word/line geometry.
ai_client   OpenAI-compatible / Anthropic REST client (no vendor SDK needed).
actions     Click/type/hotkey simulation via ``pyautogui`` with safety rails.
hotkeys     Global hotkey manager (``pynput``, optional ``keyboard`` backend).
native      macOS permissions (Accessibility / Input Monitoring) + Dock icon policy.
tray        Menu-bar / tray icon and its menu (background mode, no window needed).
overlay     Frameless region selector + always-on-top region marker (Qt).
pipeline    Orchestrator: capture -> OCR -> AI -> action, running in a QThread.
ui          PySide6 control panel (main entry point of the GUI).
cli         Headless entry point for scripting / smoke tests.

Run the GUI:  ``python -m screenai_bot``            (or ``--background`` for the tray)
Run the CLI:  ``python -m screenai_bot.cli --help``
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__", "main"]


def _setup_logging() -> None:
    """Log to ``~/.screenai_bot/screenai.log`` (a tray app has no console).

    Without this, warnings such as "macOS blocks the global hotkeys" vanished
    into a stderr nobody sees once the window is closed.
    """
    import logging
    from logging.handlers import RotatingFileHandler

    from .config import APP_DIR

    root = logging.getLogger("screenai")
    if root.handlers:
        return
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        file_handler = RotatingFileHandler(
            APP_DIR / "screenai.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except Exception:  # read-only home, weird filesystem: stderr is enough
        pass
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root.addHandler(stream)


def run_gui(start_hidden: bool = False) -> int:
    """Launch the desktop control panel (PySide6). Returns the process exit code.

    ``start_hidden=True`` skips showing the window and drops straight into the
    menu bar / tray ("Docker Dashboard" mode), so the bot is usable only through
    the hotkeys and the status icon.
    """
    import sys

    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        raise SystemExit(
            f"GUI dependencies missing ({exc}).\n"
            "Install them with:  pip install -r requirements.txt\n"
            "Or run the headless CLI instead:  python -m screenai_bot.cli --help"
        )

    from .ui import MainWindow

    _setup_logging()
    app = QApplication(sys.argv)
    app.setApplicationName("ScreenAI Bot")
    app.setOrganizationName("screenai-bot")
    window = MainWindow()
    # MainWindow already decided whether closing the panel may quit the process
    # (see _apply_background_policy), so nothing to force here.
    app.aboutToQuit.connect(window._shutdown)
    if start_hidden or bool(window.cfg.get_path("app.start_minimized", False)):
        window.hide_to_tray(log_hint=False)
    else:
        window.show()
    return app.exec()


def main(argv=None):
    """Entry point used by ``python -m screenai_bot``.

    With no arguments it launches the GUI; with CLI flags (or when Qt is not
    installed) it forwards to the headless CLI so scripting still works.
    ``--background`` / ``--tray`` start the GUI hidden in the menu bar.
    """
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    background = bool(args) and all(arg in ("--background", "--tray") for arg in args)
    if background:
        try:
            return run_gui(start_hidden=True)
        except ImportError as exc:  # pragma: no cover - depends on install state
            print(f"GUI unavailable ({exc}). Use 'python -m screenai_bot.cli --help'.",
                  file=sys.stderr)
            return 1
    if args:
        from .cli import main as cli_main

        return cli_main(args)
    try:
        return run_gui()
    except ImportError as exc:  # pragma: no cover - depends on install state
        print(f"GUI unavailable ({exc}). Use 'python -m screenai_bot.cli --help'.",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
