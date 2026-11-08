"""Input simulation: clicks and typing inside (or at) a screen region.

Two backends are used on purpose:

* ``pyautogui`` performs the actual mouse/keyboard events (cross-platform).
* ``pynput`` supplies the *currently focused* window coordinates, which lets the
  bot act on "wherever my text box is" without the user re-selecting a region.

Safety rails
------------
* ``failsafe`` keeps pyautogui's move-to-corner abort enabled by default.
* Every action goes through :class:`ActionPlanner` so it can be previewed and
  confirmed before anything touches the desktop.
"""

from __future__ import annotations

import logging
import platform
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

from .config import Config, Region

log = logging.getLogger("screenai.actions")

try:
    import pyautogui
except Exception as exc:  # pragma: no cover
    pyautogui = None  # type: ignore
    _PG_ERROR = exc
else:
    _PG_ERROR = None

try:
    from pynput import mouse as pn_mouse
except Exception:  # pragma: no cover
    pn_mouse = None  # type: ignore


@dataclass
class ActionStep:
    kind: str                      # click | double_click | right_click | type | hotkey | key | paste | wait | move
    x: int = 0
    y: int = 0
    text: str = ""
    keys: Tuple[str, ...] = ()
    button: str = "left"
    interval: float = 0.012
    duration: float = 0.25
    note: str = ""

    def describe(self) -> str:
        if self.kind in ("type", "paste"):
            preview = self.text.replace("\n", "\\n")
            if len(preview) > 60:
                preview = preview[:57] + "..."
            return f"{self.kind} \"{preview}\" @({self.x},{self.y})"
        if self.kind == "hotkey":
            return f"hotkey {'+'.join(self.keys)}"
        if self.kind == "key":
            return f"key {self.keys[0] if self.keys else '?'}"
        if self.kind == "wait":
            return f"wait {self.text or '0.3s'}"
        return f"{self.kind} @({self.x},{self.y})"


@dataclass
class ActionResult:
    ok: bool
    steps: List[ActionStep] = field(default_factory=list)
    error: str = ""
    executed: int = 0

    def __bool__(self) -> bool:
        return self.ok


def require_pyautogui() -> Any:
    if pyautogui is None:
        raise RuntimeError(f"pyautogui is required for input actions ({_PG_ERROR}).")
    return pyautogui


def pointer_position() -> Tuple[int, int]:
    """Current cursor position in virtual-desktop coordinates."""
    if pn_mouse is not None:
        try:
            pos = pn_mouse.Controller().position
            return (int(pos.x), int(pos.y))
        except Exception:
            pass
    if pyautogui is not None:
        point = pyautogui.position()
        return (int(point.x), int(point.y))
    return (0, 0)


def screen_size() -> Tuple[int, int]:
    if pyautogui is not None:
        size = pyautogui.size()
        return (int(size.width), int(size.height))
    return (1920, 1080)


class InputController:
    """Thin, testable wrapper around pyautogui with our config defaults applied."""

    def __init__(
        self, cfg: Config, on_pointer: Optional[Callable[[int, int, str], None]] = None
    ) -> None:
        self.cfg = cfg
        self._on_pointer = on_pointer
        pg = require_pyautogui()
        pg.FAILSAFE = bool(cfg.get_path("actions.failsafe", True))
        pg.PAUSE = 0.0
        self._pg = pg

    # ------------------------------------------------------------------ primitives
    def move(self, x: int, y: int, duration: Optional[float] = None) -> None:
        self._pg.moveTo(x, y, duration=duration if duration is not None
                        else float(self.cfg.get_path("actions.move_duration", 0.25)))

    def click(self, x: int, y: int, clicks: int = 1, button: str = "left",
              duration: Optional[float] = None) -> None:
        self._pg.click(
            x=x, y=y, clicks=clicks, button=button,
            interval=float(self.cfg.get_path("actions.click_duration", 0.06)),
        )
        # The OS can clamp/misapply a position (scaled display, edge of screen,
        # dead zone). Verify where the pointer really ended up so a "wrong spot"
        # click is visible in the log instead of failing silently.
        try:
            actual = tuple(int(v) for v in self._pg.position())
            if abs(actual[0] - int(x)) > 2 or abs(actual[1] - int(y)) > 2:
                log.warning(
                    "Pointer landed at %s instead of (%s,%s) - click went to the "
                    "wrong place!", actual, int(x), int(y),
                )
            else:
                log.info("click delivered at %s (button=%s, clicks=%s)", actual, button, clicks)
        except Exception:  # pragma: no cover - position() unavailable
            log.debug("could not read back the pointer position", exc_info=True)
        if duration:
            time.sleep(duration)

    def type_text(self, text: str, interval: Optional[float] = None) -> None:
        """Type ``text``; newlines become Enter and non-ASCII chars go via clipboard."""
        interval = interval if interval is not None else float(
            self.cfg.get_path("actions.type_interval", 0.012))
        chunks, unsupported = _split_supported(text)
        for chunk in chunks:
            if chunk is None:
                self._pg.press("enter")
            elif chunk:
                self._pg.typewrite(chunk, interval=interval)
        for char in unsupported:
            _type_char_via_clipboard(self._pg, char)

    def press_key(self, key: str) -> None:
        self._pg.press(key)

    def hotkey(self, *keys: str) -> None:
        keys = tuple(k for k in keys if k)
        if keys:
            self._pg.hotkey(*keys)

    def copy_to_clipboard(self, text: str) -> bool:
        return _set_clipboard(text)

    def paste(self, text: str) -> bool:
        """Clipboard-based paste - fastest and unicode-safe (Cmd+C/V on macOS)."""
        if not _set_clipboard(text):
            return False
        time.sleep(0.05)
        if platform.system() == "Darwin":
            self.hotkey("command", "v")
        else:
            self.hotkey("ctrl", "v")
        return True

    def _notify_pointer(self, step: ActionStep) -> None:
        """Debug aid: ask the UI to flash the click location before it happens."""
        if self._on_pointer is None or not self.cfg.get_path("actions.debug_pointer", False):
            return
        try:
            self._on_pointer(int(step.x), int(step.y), step.kind)
        except Exception:
            log.debug("pointer debug callback failed", exc_info=True)
            return
        # Give the GUI a moment to draw the ripple before the real click lands.
        delay_ms = int(self.cfg.get_path("actions.debug_pointer_delay_ms", 120) or 0)
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)

    # --------------------------------------------------------------------- steps
    def run_steps(self, steps: List[ActionStep], stop_flag: Optional[Any] = None) -> ActionResult:
        result = ActionResult(ok=True, steps=list(steps))
        for step in steps:
            if stop_flag is not None and stop_flag.is_set():
                result.ok = False
                result.error = "Stopped by user."
                return result
            try:
                self._run_step(step)
                result.executed += 1
            except Exception as exc:
                log.exception("Action failed: %s", step.describe())
                result.ok = False
                result.error = f"{step.describe()} failed: {exc}"
                return result
        return result

    def _run_step(self, step: ActionStep) -> None:
        if step.kind in ("click", "double_click", "right_click"):
            clicks = 2 if step.kind == "double_click" else 1
            button = "right" if step.kind == "right_click" else step.button
            log.info("ACTION %s", step.describe())
            self._notify_pointer(step)
            self.click(step.x, step.y, clicks=clicks, button=button)
            time.sleep(0.12)
        elif step.kind == "move":
            self.move(step.x, step.y)
        elif step.kind == "type":
            log.info("ACTION %s", step.describe())
            self.type_text(step.text, step.interval)
        elif step.kind == "paste":
            log.info("ACTION paste @(%d,%d)", step.x, step.y)
            if not self.paste(step.text):
                self.type_text(step.text)
        elif step.kind == "hotkey":
            self.hotkey(*step.keys)
        elif step.kind == "key":
            self.press_key(step.keys[0] if step.keys else "enter")
        elif step.kind == "wait":
            seconds = _parse_seconds(step.text)
            log.info("ACTION wait %.2fs", seconds)
            time.sleep(seconds)
        else:
            raise ValueError(f"Unknown action kind: {step.kind}")


# ---------------------------------------------------------------------- planning
class ActionPlanner:
    """Turns high level intent ("type this answer into that area") into steps."""

    MODES = ("type_at_cursor", "type_in_region", "click_and_type", "click_only",
             "double_click_and_type", "copy_only", "press_enter_after")

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def build(
        self,
        text: str,
        mode: str = "type_in_region",
        source_region: Optional[Region] = None,
        target_region: Optional[Region] = None,
        click_point: Optional[Tuple[int, int]] = None,
        clear_field: bool = True,
        press_enter: bool = False,
    ) -> List[ActionStep]:
        steps: List[ActionStep] = []
        target = target_region or source_region
        use_cursor = click_point is None and mode in ("type_at_cursor",)
        x, y = click_point if click_point else (target.center if target else pointer_position())

        if mode == "copy_only":
            return steps

        focus_click = bool(self.cfg.get_path("actions.focus_click", True))
        if mode in ("click_and_type", "double_click_and_type", "click_only", "type_in_region") \
                and target is not None and focus_click:
            kind = "double_click" if mode == "double_click_and_type" else "click"
            steps.append(ActionStep(kind=kind, x=int(x), y=int(y)))
            steps.append(ActionStep(kind="wait", text="0.15"))

        if mode == "click_only":
            return steps

        if clear_field and mode != "click_only":
            shortcut = str(self.cfg.get_path("actions.clear_field_shortcut", "ctrl+a"))
            steps.append(ActionStep(kind="hotkey", keys=normalize_clear_shortcut(shortcut)))
            steps.append(ActionStep(kind="key", keys=("delete",)))

        if text:
            prefer_paste = any(ord(ch) > 126 for ch in text) or "\n" in text
            steps.append(
                ActionStep(
                    kind="paste" if prefer_paste else "type",
                    x=int(x), y=int(y), text=text,
                    interval=float(self.cfg.get_path("actions.type_interval", 0.012)),
                )
            )
        if press_enter or mode == "press_enter_after":
            steps.append(ActionStep(kind="key", keys=("enter",)))
        if use_cursor:
            log.debug("Typing at current cursor %s", (x, y))
        return steps


# ----------------------------------------------------------------------- helpers
_SUPPORTED_RANGES = [(0x20, 0x7E)]


def _is_supported(char: str) -> bool:
    code = ord(char)
    return any(lo <= code <= hi for lo, hi in _SUPPORTED_RANGES)


def _split_supported(text: str) -> Tuple[List[str], List[str]]:
    """Split ASCII-runnable chunks (newlines become Enter presses) from the rest.

    Returns ``(chunks, unsupported_chars)`` where a ``None`` entry in ``chunks``
    means "press Enter", which pyautogui's typewrite cannot emit directly.
    """
    chunks: List[Any] = []
    unsupported: List[str] = []
    buffer = ""
    for char in text:
        if char == "\n":
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.append(None)  # newline -> Enter key
            continue
        if _is_supported(char):
            buffer += char
        else:
            unsupported.append(char)
    if buffer:
        chunks.append(buffer)
    return chunks, unsupported


def _type_char_via_clipboard(pg: Any, char: str) -> None:
    if _set_clipboard(char):
        if platform.system() == "Darwin":
            pg.hotkey("command", "v")
        else:
            pg.hotkey("ctrl", "v")
        time.sleep(0.03)


def normalize_clear_shortcut(shortcut: str) -> Tuple[str, ...]:
    """Map a configured clear shortcut onto real key names for this OS.

    ``ctrl+a`` on macOS must become ``cmd+a`` (select-all), otherwise it moves the
    caret to the start of the line instead of selecting the field contents.
    """
    keys = tuple(k.strip().lower() for k in str(shortcut or "").split("+") if k.strip())
    if platform.system() == "Darwin":
        keys = tuple("command" if k in ("ctrl", "control", "cmd") else k for k in keys)
    return keys


def _set_clipboard(text: str) -> bool:
    """Best-effort clipboard write across platforms (no extra dependency)."""
    import os
    import subprocess

    system = platform.system()
    try:
        if system == "Windows":
            # clip.exe reads bytes and decodes them using the console OEM codepage,
            # which mangles non-ASCII. Prefer PowerShell Set-Clipboard when present.
            try:
                subprocess.run(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                     "Set-Clipboard -Value ([Console]::In.ReadToEnd())"],
                    input=text.encode("utf-8"), check=True, timeout=6,
                )
                return True
            except Exception:
                subprocess.run(["clip.exe"], input=text.encode("utf-8", "ignore"),
                               check=True, env=os.environ, timeout=6)
                return True
        if system == "Darwin":
            subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
            return True
        for tool in (["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "-b", "-i"]):
            try:
                subprocess.run(tool, input=text.encode("utf-8"), check=True)
                return True
            except Exception:
                continue
    except Exception:
        log.exception("Clipboard write failed")
    return False


def _parse_seconds(value: str) -> float:
    try:
        raw = str(value).strip().lower().rstrip("s")
        return max(0.0, float(raw))
    except Exception:
        return 0.3
