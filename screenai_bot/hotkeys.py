"""Global hotkeys (system-wide, work even when the bot window is not focused).

Primary backend: ``pynput``. On Windows you can optionally switch to ``keyboard``
for lower-latency hooks via ``hotkey_backend`` in config.

Two details make this reliable on macOS:

* Keys fire on **press**, not release. macOS and Windows drop release events when
  the app loses focus mid-combination, which used to swallow the shortcut.
* ``char`` becomes a control code while Ctrl is held (``Ctrl+R`` arrives as
  ``"\\x12"``), so the character is folded back to a letter before matching.

Handlers registered here run on the *listener* thread. Qt widgets must never be
touched from there, so the UI routes the callback onto the GUI thread itself.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict, FrozenSet, Optional

from .native import (
    IS_MACOS, hotkey_permission_hint, input_monitoring_trusted, system_key_age,
)

log = logging.getLogger("screenai.hotkeys")

try:
    from pynput import keyboard as pn_keyboard
except Exception as exc:  # pragma: no cover
    pn_keyboard = None
    _PYNPUT_ERROR = exc
else:
    _PYNPUT_ERROR = None

HotkeyHandler = Callable[[], None]

#: Spelling variants for the same physical key, per platform.
_KEY_ALIASES: Dict[str, str] = {
    "control": "ctrl", "ctrl_l": "ctrl", "ctrl_r": "ctrl", "control_l": "ctrl",
    "control_r": "ctrl",
    "option": "alt", "alt_l": "alt", "alt_r": "alt", "alt_gr": "alt",
    "shift_l": "shift", "shift_r": "shift",
    "command": "cmd", "cmd_l": "cmd", "cmd_r": "cmd",
    "super": "cmd", "super_l": "cmd", "super_r": "cmd",
    "win": "cmd", "win_l": "cmd", "win_r": "cmd",
    "meta": "cmd", "meta_l": "cmd", "meta_r": "cmd",
    "return": "enter", "escape": "esc", "esc": "esc",
    "spacebar": "space",
    "kp_enter": "enter", "numpad_enter": "enter", "pad_enter": "enter",
    "pgup": "page_up", "pageup": "page_up",
    "pgdn": "page_down", "pagedown": "page_down",
    "del": "delete", "ins": "insert",
}

#: Keys that can never be part of a shortcut (they would stick in ``_pressed``).
IGNORED_KEYS = frozenset({"caps_lock", "num_lock", "scroll_lock", "fn", "unknown"})

#: Pure modifiers: one of these may be pressed by accident without breaking a combo.
MODIFIERS = frozenset({"ctrl", "alt", "shift", "cmd"})

#: Control characters that are *keys* of their own, not Ctrl+<letter>.
#: Return arrives as "\r" on some layouts/keyboards; without this it was folded
#: into a letter ("\r" -> "m") and "ctrl+alt+enter" could never match.
_CONTROL_CHAR_KEYS: Dict[str, str] = {
    "\r": "enter", "\n": "enter", "\t": "tab", "\x1b": "esc",
    "\x7f": "backspace", "\x08": "backspace",
}

#: How each key is written on a Mac keyboard (⌃⌥↩ ...), used in menus/prompts.
MODIFIER_SYMBOLS = {"ctrl": "\u2303", "alt": "\u2325", "shift": "\u21e7", "cmd": "\u2318"}
KEY_SYMBOLS = {
    "enter": "\u21a9", "esc": "\u238b", "tab": "\u21e5", "space": "\u2423",
    "backspace": "\u232b", "delete": "\u2326",
    "up": "\u2191", "down": "\u2193", "left": "\u2190", "right": "\u2192",
    "page_up": "\u21de", "page_down": "\u21df", "home": "\u2196", "end": "\u2198",
}


def describe_combo(combo: str, macos: Optional[bool] = None) -> str:
    """Render a combo the way the user's keyboard labels it.

    ``"ctrl+alt+enter"`` becomes ``"⌃⌥↩"`` on macOS and ``"Ctrl+Alt+Enter"``
    elsewhere, which is what makes "Ctrl+Alt is Control+Option, not Command"
    obvious without a manual.
    """
    if macos is None:
        macos = IS_MACOS
    parts = [p.strip().lower() for p in str(combo).replace(" ", "").split("+") if p.strip()]
    if not parts:
        return str(combo)
    if macos:
        out = []
        for part in parts:
            name = _KEY_ALIASES.get(part, part)
            out.append(MODIFIER_SYMBOLS.get(name) or KEY_SYMBOLS.get(name)
                       or (name.upper() if len(name) == 1 else name.title()))
        return "".join(out)
    out = []
    for part in parts:
        name = _KEY_ALIASES.get(part, part)
        if name in MODIFIER_SYMBOLS:
            out.append("Ctrl" if name == "ctrl" else name.capitalize())
            continue
        # Plain words on Windows/Linux ("Enter"), symbols only on macOS.
        out.append(name.upper() if len(name) == 1 else name.replace("_", " ").title())
    return "+".join(out)

#: macOS hardware keycodes (ANSI) -> key name.
#:
#: Needed because macOS composes the character while Option is held: pressing
#: Option+R delivers ``char="®"`` (and Option+S gives "ß"), so a binding such as
#: ``ctrl+alt+r`` could *never* match on a Mac. The keycode stays stable, so it
#: is used to recover the physical key.
_MAC_VK_NAMES: Dict[int, str] = {
    0: "a", 1: "s", 2: "d", 3: "f", 4: "h", 5: "g", 6: "z", 7: "x", 8: "c", 9: "v",
    11: "b", 12: "q", 13: "w", 14: "e", 15: "r", 16: "y", 17: "t",
    18: "1", 19: "2", 20: "3", 21: "4", 22: "6", 23: "5", 24: "=", 25: "9", 26: "7",
    27: "-", 28: "8", 29: "0", 30: "]", 31: "o", 32: "u", 33: "[", 34: "i", 35: "p",
    37: "l", 38: "j", 39: "'", 40: "k", 41: ";", 42: "\\", 43: ",", 44: "/",
    45: "n", 46: "m", 47: ".", 50: "`",
    36: "enter", 76: "enter", 48: "tab", 49: "space", 51: "backspace", 53: "esc",
    117: "delete",
    115: "home", 116: "page_up", 119: "end", 121: "page_down",
    123: "left", 124: "right", 125: "down", 126: "up",
    122: "f1", 120: "f2", 99: "f3", 118: "f4", 96: "f5", 97: "f6", 98: "f7",
    100: "f8", 101: "f9", 109: "f10", 103: "f11", 111: "f12",
}


def parse_combo(combo: str) -> FrozenSet[str]:
    """'ctrl+alt+enter' -> frozenset of normalized pynput key names."""
    parts = [p.strip().lower() for p in str(combo).replace(" ", "").split("+") if p.strip()]
    normalized = set()
    for part in parts:
        if len(part) == 1:
            normalized.add(part)
            continue
        normalized.add(_KEY_ALIASES.get(part, part))
    return frozenset(normalized)


def _normalize_key(key) -> Optional[str]:
    """pynput key -> short lowercase name understood by :func:`parse_combo`."""
    if key is None:
        return None
    char = getattr(key, "char", None)
    vk = getattr(key, "vk", None)
    if char:
        # Return/Tab/Esc/Backspace can arrive as control characters; they must
        # not be mistaken for Ctrl+<letter> shortcuts.
        if char in _CONTROL_CHAR_KEYS:
            return _CONTROL_CHAR_KEYS[char]
        # Ctrl+letter arrives as a control character ("\x12" for ctrl+r), which
        # used to make every letter-based combo silently unmatchable.
        if len(char) == 1 and ord(char) < 0x20:
            # Trust the physical key first: macOS reports keypad Enter as "\x03",
            # the same control code Ctrl+C produces, so the char alone is
            # ambiguous while the keycode is not.
            mapped = _MAC_VK_NAMES.get(vk) if IS_MACOS else None
            if mapped:
                return mapped
            code = ord(char)
            return chr(code + 0x60) if 1 <= code <= 26 else None
        if not char.strip():
            return "space"
        if char.isascii():
            return char.lower()
        # macOS composes the character while Option is held (Option+R -> "®"),
        # so the composed char can never match "ctrl+alt+r": use the keycode.
        mapped = _MAC_VK_NAMES.get(vk) if IS_MACOS else None
        return mapped or char.lower()
    name = getattr(key, "name", None)
    if name:
        name = str(name).lower()
        return _KEY_ALIASES.get(name, name)
    # Some special keys arrive as a bare KeyCode (no name, no char): the macOS
    # keycode table still identifies them (Return 36, keypad Enter 76, ...).
    mapped = _MAC_VK_NAMES.get(vk) if IS_MACOS else None
    if mapped:
        return mapped
    text = str(key).split(".")[-1].strip().lower()
    return _KEY_ALIASES.get(text, text) if text and text != "none" else None


class HotkeyManager:
    """Registers global shortcuts and fires callbacks on the UI/main thread-safe path."""

    def __init__(self, backend: str = "pynput", max_restarts: int = 2) -> None:
        self.backend = backend
        self._combos: Dict[frozenset, HotkeyHandler] = {}
        self._listener = None
        self._pressed: set = set()
        self._fired: set = set()
        self._press_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        self.enabled = True
        # --- watchdog (macOS disables event taps without telling pynput) ----
        self._seen_at_last_check = 0
        self._dead_checks = 0
        self.restart_count = 0
        self._last_restart = 0.0
        self._stopped = False
        #: How many *automatic* rebuilds are allowed before asking the user to
        #: press "Restart Hotkeys" (0 disables automatic recovery entirely).
        self.max_restarts = max(0, int(max_restarts))
        self.gave_up = False
        self._restart_lock = threading.Lock()
        self._restart_thread: Optional[threading.Thread] = None
        #: Optional hook ``f(key_name)`` fired when a key matched no shortcut
        #: (used for the optional "miss" debug sound). Runs on the listener thread.
        self.on_unmatched: Optional[Callable[[str], None]] = None
        # --- diagnostics, shown by the "Test Hotkeys" dialog ---------------
        self.event_count = 0
        self.fire_count = 0
        self.last_key: Optional[str] = None
        self.last_combo: Optional[str] = None
        self.last_fire = 0.0
        self.started_at = 0.0

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

    @property
    def combos(self) -> Dict[str, str]:
        """Human readable view of what is bound (used in logs and the tray menu)."""
        with self._lock:
            return {
                "+".join(sorted(combo)): getattr(handler, "__name__", "handler")
                for combo, handler in self._combos.items()
            }

    @property
    def running(self) -> bool:
        return self._listener is not None

    @property
    def receiving_events(self) -> bool:
        """True once at least one key press arrived: proves the OS delivers them."""
        return self.event_count > 0

    def activity_text(self) -> str:
        """Live one-liner for the hotkey diagnostics dialog."""
        with self._lock:
            held = "+".join(sorted(self._pressed)) or "-"
        state = "running" if self.running else "stopped"
        if not self.receiving_events and self.running:
            state += " (no key events received yet)"
        return (
            f"listener: {state} | keys seen: {self.event_count} | "
            f"last key: {self.last_key or '-'} | held: {held} | "
            f"triggers: {self.fire_count}"
            + (f" | last trigger: {self.last_combo}" if self.last_combo else "")
        )

    # ----------------------------------------------------------------------- lifecycle
    def start(self) -> bool:
        if self._listener is not None:
            return True
        self._stopped = False
        if self.backend == "keyboard":
            return self._start_keyboard_backend()
        if pn_keyboard is None:
            return False
        try:
            self._listener = pn_keyboard.Listener(
                on_press=self._on_press, on_release=self._on_release
            )
            self._listener.daemon = True
            self._listener.start()
            self.started_at = time.time()
            log.info("Global hotkey listener started (pynput)")
            self._log_permission_state()
            return True
        except Exception:
            log.exception(
                "Could not start the global hotkey listener (macOS needs Input Monitoring)"
            )
            return False

    def _log_permission_state(self) -> None:
        """Say out loud when the OS is going to swallow every event.

        pynput still reports a running listener without permission, which is how
        the bot used to claim "Global hotkeys active" while nothing happened.
        """
        trusted = input_monitoring_trusted()
        if trusted is False:
            log.warning("Global hotkeys are blocked by macOS: %s", hotkey_permission_hint())
        elif trusted is None:
            log.debug("Input Monitoring state unknown on this platform")

    def stop(self) -> None:
        self._stopped = True
        listener, self._listener = self._listener, None
        if listener is not None:
            try:
                listener.stop()
            except Exception:
                pass
            # Let the CFRunLoop thread finish before anything else touches it.
            self._join_listener(listener, timeout=1.5)
        if self.backend == "keyboard":
            try:
                import keyboard as kb_module

                kb_module.unhook_all()
            except Exception:
                pass

    # ------------------------------------------------------------------- watchdog
    def _join_listener(self, listener, timeout: float = 3.0) -> None:
        """Wait for a stopped listener thread to finish.

        Destroying the CoreFoundation run loop of a ``pynput`` listener while it
        is still unwinding is what turns a "restart" into a ``SIGTRAP`` crash,
        so the old thread is always joined before a new listener is created.
        """
        if listener is None:
            return
        thread = getattr(listener, "_thread", None)
        try:
            if thread is not None and thread is not threading.current_thread():
                if thread.is_alive():
                    thread.join(timeout)
                    if thread.is_alive():
                        log.warning("Old hotkey listener did not stop within %.0fs", timeout)
        except Exception:  # pragma: no cover - defensive
            log.debug("Joining the old listener failed", exc_info=True)

    def _note_gave_up(self) -> None:
        """Tell the user once that automatic recovery has stopped."""
        if self.gave_up:
            return
        self.gave_up = True
        log.error(
            "Giving up on automatic hotkey recovery after %d attempt(s) - "
            "use 'Restart Hotkeys' in the tray menu or the Test Hotkeys dialog.",
            self.restart_count,
        )

    def restart(self, reason: str = "", manual: bool = False) -> bool:
        """Request a listener rebuild. Asynchronous and safe from the GUI thread.

        ``manual`` (the Restart Hotkeys button) ignores the automatic budget.
        """
        if self.backend != "pynput":
            return False
        if manual:
            self.restart_count = 0
            self.gave_up = False
        elif self.max_restarts <= 0 or self.restart_count >= self.max_restarts:
            self._note_gave_up()
            return False
        if self._restart_thread is not None and self._restart_thread.is_alive():
            log.debug("A hotkey restart is already running")
            return False
        thread = threading.Thread(
            target=self._restart_worker, args=(reason,), name="hotkey-restart", daemon=True
        )
        self._restart_thread = thread
        thread.start()
        return True

    def _restart_worker(self, reason: str) -> None:
        with self._restart_lock:
            self.restart_count += 1
            self._last_restart = time.monotonic()
            self._dead_checks = 0
            log.warning(
                "Restarting the global hotkey listener (%s, attempt %d/%d)",
                reason or "requested", self.restart_count, max(self.max_restarts, 1),
            )
            old = self._listener
            try:
                self.stop()          # stop the run loop ...
                self._join_listener(old)   # ... and let its thread finish
                self.reset_state()
                self._stopped = False
                self.start()
            except Exception:  # pragma: no cover - never kill the app over this
                log.exception("Could not rebuild the hotkey listener")

    def ensure_alive(self) -> bool:
        """Watchdog tick: revive the listener when macOS silently killed it.

        ``pynput`` never handles ``kCGEventTapDisabledByTimeout``, so after a
        timeout (heavy input, sleep/wake, a secure-input dialog) the tap stays
        dead until the listener is recreated - hotkeys simply stop working.

        Only a listener that *used to* deliver keys is rebuilt: a tap that never
        saw a single key is blocked by permissions or by another app's secure
        input, where restarting cannot help (and would loop forever).
        """
        if self.backend != "pynput" or self._stopped:
            return False
        listener = self._listener
        if listener is None:
            return self.restart("listener missing")
        if not getattr(listener, "running", True):
            return self.restart("listener stopped")
        thread = getattr(listener, "_thread", None)
        if thread is not None and not thread.is_alive():
            return self.restart("listener thread is gone")
        # Compare with the counter stored by the *previous* tick, then update it.
        previous = self._seen_at_last_check
        self._seen_at_last_check = self.event_count
        if not self.receiving_events:
            # Blind since startup: a permission / secure-input problem, not a
            # disabled tap. Do not thrash the listener.
            self._dead_checks = 0
            return False
        # The OS knows about a key press we never received -> the tap is dead.
        age = system_key_age()
        if age is not None and age < 2.0 and previous == self.event_count:
            self._dead_checks += 1
            if self._dead_checks >= 2:
                if self.max_restarts <= 0 or self.restart_count >= self.max_restarts:
                    # Budget spent: say so now instead of waiting for the cooldown.
                    self._note_gave_up()
                elif (time.monotonic() - self._last_restart) > 10:
                    return self.restart("no events while the OS saw key presses")
        else:
            self._dead_checks = 0
        return False
        return False

    def blocked_hint(self) -> str:
        """Explain a listener that is running but receives nothing.

        Two causes exist on macOS: another app holds *secure keyboard entry*
        (password fields, banking apps) or a kiosk browser such as Safe Exam
        Browser is running. Both make macOS drop the key stream for everyone.
        """
        if not self.receiving_events and self.restart_count >= 2:
            return (
                "Another application is blocking key monitoring - a password field "
                "with secure input, or a kiosk browser such as Safe Exam Browser. "
                "Close it, then press 'Restart Hotkeys'."
            )
        return ""

    # ------------------------------------------------------------------------ events
    def _normalize(self, key) -> Optional[str]:
        return _normalize_key(key)

    def _on_press(self, key) -> None:
        """Fire the shortcut as soon as its last key goes down.

        Firing on press (instead of release) is what makes the combos work on
        macOS, where the release event is often lost when focus changes. The
        ``_fired`` set stops auto-repeat from firing the same combo repeatedly
        while a key is held down.
        """
        name = self._normalize(key)
        if not name:
            return
        # Count *every* key, including the ones shortcuts ignore (Caps Lock...):
        # this counter is the watchdog's proof that the tap still delivers events.
        self.event_count += 1
        self.last_key = name
        if name in IGNORED_KEYS:
            return
        now = time.monotonic()
        with self._lock:
            # Forget old non-modifier keys: macOS can lose their key-up events, and
            # a stale entry would otherwise linger in the "held" readout forever.
            for stale in [k for k, t in self._press_times.items()
                          if k not in MODIFIERS and now - t > 2.5]:
                self._pressed.discard(stale)
                self._press_times.pop(stale, None)
            self._pressed.add(name)
            self._press_times[name] = now
            # Matching looks at "the key just pressed + the modifiers currently
            # held", so leftover keys can never block a shortcut again. Switching
            # apps (Cmd+Tab) used to leave ``cmd``/``tab`` stuck in the pressed
            # set, after which every combo silently stopped matching.
            held = frozenset(k for k in self._pressed if k in MODIFIERS)
            snapshot = frozenset(set(held) | {name})
        if not self.enabled:
            return
        if name in MODIFIERS:
            # Modifiers alone never fire a combo; wait for the real key.
            return
        with self._lock:
            handlers = list(self._combos.items())
        # Smallest combo first so an exact match always wins over a superset.
        for combo, handler in sorted(handlers, key=lambda item: len(item[0])):
            if name not in combo or not combo.issubset(snapshot):
                continue
            # Extra pressed *modifiers* are tolerated (a stale modifier from
            # Cmd+Tab, or a finger resting on Shift). Any other extra key means
            # this is a different shortcut, so it is skipped.
            if any(key not in MODIFIERS for key in snapshot - combo):
                continue
            if combo in self._fired:
                return
            self._fired.add(combo)
            self.fire_count += 1
            self.last_fire = time.time()
            self.last_combo = "+".join(sorted(combo))
            log.debug("Hotkey %s", self.last_combo)
            try:
                handler()
            except Exception:
                log.exception("Hotkey handler crashed for %s", self.last_combo)
            return
        # Nothing matched: let the UI beep about it (debug aid, opt-in).
        callback = self.on_unmatched
        if callback is not None:
            try:
                callback(name)
            except Exception:  # pragma: no cover - a debug hook must never break input
                log.debug("on_unmatched hook failed", exc_info=True)

    def _on_release(self, key) -> None:
        name = self._normalize(key)
        if not name:
            return
        with self._lock:
            self._pressed.discard(name)
            self._press_times.pop(name, None)
            # A combo can fire again once one of its own keys came back up.
            self._fired = {combo for combo in self._fired if combo.issubset(self._pressed)}

    def reset_state(self) -> None:
        """Forget held keys (used when the window/app is activated again).

        macOS regularly loses key-release events while another app is being
        activated, which is how a combo could stay "blocked" until a restart.
        """
        with self._lock:
            dropped = len(self._pressed)
            self._pressed.clear()
            self._press_times.clear()
            self._fired.clear()
        if dropped:
            log.debug("Hotkey state reset (%d stale key(s) forgotten)", dropped)

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
