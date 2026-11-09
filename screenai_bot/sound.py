"""Debug sounds: audible feedback for hotkeys, clicks and failures.

Off by default. Enable with the **Debug Sounds** checkbox in *Options*
(``debug.sounds`` in config), then use the per-event switches to choose what is
audible - hotkey fired, hotkey that matched nothing, mouse clicks, typing,
run start/finish, failures and cancelled actions.

On macOS the cues are played through the system sound bank (``NSSound``); other
platforms fall back to the platform beep. Playback is best-effort and never
raises, so a missing audio device cannot break the pipeline.
"""

from __future__ import annotations

import logging
import sys
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("screenai.sound")

IS_MACOS = sys.platform == "darwin"
MACOS_SOUND_DIR = "/System/Library/Sounds"

#: cue -> (macOS system sound, meaning shown in the UI/README)
CUES: Dict[str, Tuple[str, str]] = {
    "hotkey": ("Tink", "a shortcut fired"),
    "hotkey_miss": ("Pop", "a key press that matched no shortcut"),
    "click": ("Morse", "the bot is clicking the mouse"),
    "type": ("Tink", "the bot is typing/pasting"),
    "start": ("Ping", "a run started"),
    "done": ("Glass", "a run finished"),
    "error": ("Basso", "something failed or was blocked"),
    "cancel": ("Bottle", "an action was declined"),
}


class SoundPlayer:
    """Plays short cues, honouring the config switches and volume."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._active: List[object] = []
        self._warned = False
        self.play_count = 0
        self.last_cue: Optional[str] = None

    # ------------------------------------------------------------------ settings
    @property
    def enabled(self) -> bool:
        """Master switch (``debug.sounds``)."""
        return bool(self.cfg.get_path("debug.sounds", False))

    @property
    def volume(self) -> float:
        try:
            return max(0.0, min(1.0, float(self.cfg.get_path("debug.sound_volume", 0.5) or 0)))
        except (TypeError, ValueError):
            return 0.5

    def cue_enabled(self, cue: str) -> bool:
        """True when the master switch and this cue's own switch are both on."""
        if not self.enabled:
            return False
        return bool(self.cfg.get_path(f"debug.sound_{cue}", True))

    def available(self) -> bool:
        """Whether audio output is usable at all on this system."""
        if IS_MACOS:
            try:
                import AppKit  # noqa: F401
                return True
            except Exception:
                return False
        return True

    # ------------------------------------------------------------------ playback
    def play(self, cue: str) -> bool:
        """Play ``cue`` if enabled. Never raises; returns True when audio played."""
        if not self.cue_enabled(cue):
            return False
        played = False
        if IS_MACOS:
            played = self._play_macos(cue)
        if not played:
            played = self._play_fallback()
        if played:
            self.play_count += 1
            self.last_cue = cue
        elif not self._warned:
            self._warned = True
            log.debug("Sound playback unavailable; debug sounds stay silent")
        return played

    def _play_macos(self, cue: str) -> bool:
        try:
            from AppKit import NSSound  # type: ignore
        except Exception:  # PyObjC missing
            return False
        name = CUES.get(cue, ("Tink", ""))[0]
        try:
            sound = NSSound.alloc().initWithContentsOfFile_byReference_(
                f"{MACOS_SOUND_DIR}/{name}.aiff", True
            )
            if sound is None:  # unknown file: let AppKit resolve the name
                sound = NSSound.soundNamed_(name)
            if sound is None:
                return False
            sound.setVolume_(self.volume)
            sound.stop()
            sound.play()
            # Hold a reference while it plays; NSSound stops if it is released.
            self._active.append(sound)
            del self._active[:-8]
            return True
        except Exception:  # pragma: no cover - audio device problems
            log.debug("Could not play the %s sound", name, exc_info=True)
            return False

    def _play_fallback(self) -> bool:
        if sys.platform.startswith("win"):
            try:
                import winsound

                winsound.MessageBeep(winsound.MB_ICONASTERISK)
                return True
            except Exception:
                pass
        try:
            from PySide6.QtWidgets import QApplication

            app = QApplication.instance()
            if app is not None:
                app.beep()
                return True
        except Exception:
            pass
        return False

    # --------------------------------------------------------------- diagnostics
    def describe(self) -> str:
        """One-line state for the UI."""
        if not self.enabled:
            return "off"
        on = [cue for cue in CUES if self.cue_enabled(cue)]
        return f"on ({self.volume:.0%}) - {', '.join(on) if on else 'no cues selected'}"
