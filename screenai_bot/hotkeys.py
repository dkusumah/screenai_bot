"""Global hotkeys (system-wide, work even when the bot window is not focused).

Primary backend: ``pynput``. On Windows you can optionally switch to ``keyboard``
for lower-latency hooks via ``hotkey_backend`` in config.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Dict, Optional

log = logging.getLogger("screenai.hotkeys")

try:
    from pynput import keyboard as pn_keyboard
except Exception as exc:  # pragma: no cover
    pn_keyboard = None
    _PYNPUT_ERROR = exc
else:
    _PYNPUT_ERROR = None

HotkeyHandler = Callable[[], None]


def parse_combo(combo: str) -> frozenset:
    """'ctrl+alt+enter' -> frozenset of normalized pynput key names."""
    aliases = {
        "ctrl": "ctrl", "control": "ctrl",
        "alt": "alt", "option": "alt",
        "shift": "shift",
        "cmd": "cmd", "command": "cmd", "super": "cmd", "win": "cmd", "meta": "cmd",
        "enter": "enter", "return": "enter",
        "esc": "esc", "escape": "esc",
        "space": "space",
        "pgup": "page_up", "pageup": "page_up",
        "pgdn": "page_down", "pagedown": "page_down",
    }
    parts = [p.strip().lower() for p in str(combo).replace(" ", "").split("+") if p.strip()]
    normalized = set()
    for part in parts:
        if len(part) == 1:
            normalized.add(part)
            continue
        normalized.add(aliases.get(part, part))
    return frozenset(normalized)


class HotkeyManager:
    """Registers global shortcuts and fires callbacks on the UI/main thread-safe path."""

    def __init__(self, backend: str = "pynput") -> None:
        self.backend = backend
        self._combos: Dict[frozenset, HotkeyHandler] = {}
        self._listener = None
        self._pressed = set()
        self._lock = threading.RLock()
        self.enabled = True

    # ------------------------------------------------------------------ registration
    def register(self, combo: str, handler: HotkeyHandler) -> bool:
        if pn_keyboard is None and self.backend != "keyboard":
            log.warning("pynput unavailable (%s); hotkeys disabled", _PYNPUT_ERROR)
            return False
        parsed = parse_combo(combo)
        if not parsed:
            return False
        with self._lock:
            self._combos[parsed] = handler
        log.info("Registered hotkey %s -> %s", combo, getattr(handler, "__name__", "handler"))
        return True

    def clear(self) -> None:
        with self._lock:
            self._combos.clear()

    # ----------------------------------------------------------------------- lifecycle
    def start(self) -> bool:
        if self._listener is not None:
            return True
        if self.backend == "keyboard":
            return self._start_keyboard_backend()
        if pn_keyboard is None:
            return False
        try:
            self._listener = pn_keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
            self._listener.daemon = True
            self._listener.start()
            log.info("Global hotkey listener started (pynput)")
            return True
        except Exception:
            log.exception("Could not start global hotkey listener (needs accessibility permission on macOS)")
            return False

    def stop(self) -> None:
        listener, self._listener = self._listener, None
        if listener is not None:
            try:
                listener.stop()
            except Exception:
                pass
        if self.backend == "keyboard":
            try:
                import keyboard as kb_module

                kb_module.unhook_all()
            except Exception:
                pass

    # ------------------------------------------------------------------------ events
    def _normalize(self, key) -> str:
        char = getattr(key, "char", None)
        if char:
            return char.lower()
        name = getattr(key, "name", "") or ""
        return {"key.enter": "enter", "cmd": "cmd", "cmd_r": "cmd",
                "ctrl_l": "ctrl", "ctrl_r": "ctrl", "alt_l": "alt", "alt_r": "alt",
                "alt_gr": "alt", "shift_l": "shift", "shift_r": "shift"}.get(name, name)

    def _on_press(self, key) -> None:
        self._pressed.add(self._normalize(key))

    def _on_release(self, key) -> None:
        released = self._normalize(key)
        current = set(self._pressed)
        self._pressed.discard(released)
        if not self.enabled:
            return
        snapshot = frozenset(current | {released})
        if released not in current:
            # Nothing that was actually held just came up -> ignore plain key taps
            # so typing normal text never triggers a modifier combo.
            return
        with self._lock:
            handlers = list(self._combos.items())
        # Exact match first; then allow supersets so e.g. ctrl+alt+enter still fires
        # when Shift is held (users often hit the wrong modifier).
        for combo, handler in sorted(handlers, key=lambda item: len(item[0])):
            if combo == snapshot or (combo.issubset(snapshot) and len(snapshot) - len(combo) <= 1):
                try:
                    handler()
                except Exception:
                    log.exception("Hotkey handler crashed for %s", "+".join(sorted(combo)))
                return

    # --------------------------------------------------------------- keyboard backend
    def _start_keyboard_backend(self) -> bool:
        try:
            import keyboard as kb_module
        except Exception:
            log.warning("python 'keyboard' package not installed")
            return False
        for combo, handler in list(self._combos.items()):
            spec = "+".join(sorted(combo))
            kb_module.add_hotkey(spec, handler, suppress=False)
        self._listener = object()  # marks "started"
        kb_module.wait()  # blocks; caller runs this in a worker thread
        return True


class LocalHotkeyFallback:
    """Qt shortcut fallback so the app still works if OS-level hooks are blocked."""

    def __init__(self, parent_window) -> None:
        self.parent = parent_window

    def bind(self, combo: str, slot: Callable[[], None]) -> None:
        # QShortcut is part of QtGui in Qt 6; fall back to QtWidgets for
        # legacy PySide6 releases that re-exported it there.
        try:
            from PySide6.QtGui import QShortcut
        except ImportError:  # pragma: no cover
            from PySide6.QtWidgets import QShortcut
        from PySide6.QtGui import QKeySequence

        shortcut = QShortcut(QKeySequence(_qt_sequence(combo)), self.parent)
        shortcut.setContext(2)  # Qt.ApplicationShortcut
        shortcut.activated.connect(slot)


def _qt_sequence(combo: str) -> str:
    mapping = {"ctrl": "Ctrl", "alt": "Alt", "shift": "Shift", "cmd": "Meta",
               "enter": "Return", "esc": "Esc", "space": "Space"}
    parts = [p.strip().lower() for p in combo.replace(" ", "").split("+") if p.strip()]
    out = []
    for part in parts:
        out.append(mapping.get(part, part.upper() if len(part) == 1 else part))
    return "+".join(out)
