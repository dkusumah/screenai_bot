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
overlay     Frameless region selector + always-on-top region marker (Qt).
pipeline    Orchestrator: capture -> OCR -> AI -> action, running in a QThread.
ui          PySide6 control panel (main entry point of the GUI).
cli         Headless entry point for scripting / smoke tests.

Run the GUI:  ``python -m screenai_bot``
Run the CLI:  ``python -m screenai_bot.cli --help``
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__", "main"]


def run_gui() -> int:
    """Launch the desktop control panel (PySide6). Returns the process exit code."""
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

    app = QApplication(sys.argv)
    app.setApplicationName("ScreenAI Bot")
    app.setOrganizationName("screenai-bot")
    window = MainWindow()
    window.show()
    return app.exec()


def main(argv=None):
    """Entry point used by ``python -m screenai_bot``.

    With no arguments it launches the GUI; with CLI flags (or when Qt is not
    installed) it forwards to the headless CLI so scripting still works.
    """
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
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
