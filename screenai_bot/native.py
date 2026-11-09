"""macOS-only helpers: permission checks, privacy panes and the Dock icon policy.

Everything here is best-effort and never fatal. On other platforms, or when
PyObjC is missing, each function reports ``None``/"unavailable" and the rest of
the app keeps working exactly as before.

Why this module exists
----------------------
macOS gates keyboard monitoring behind two switches (System Settings >
Privacy & Security):

* **Accessibility** (``AXIsProcessTrusted``) - needed to *send* clicks/keys.
* **Input Monitoring** (``CGPreflightListenEventAccess``) - needed to *listen*
  for the global hotkeys.

When either is off, ``pynput`` starts happily but never delivers an event, so the
bot used to claim "Global hotkeys active" while nothing worked. These helpers let
the UI tell the truth and send the user straight to the right settings pane.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from typing import Dict, Optional

log = logging.getLogger("screenai.native")

IS_MACOS = sys.platform == "darwin"

#: ``open`` URLs that jump directly to the matching privacy pane.
PRIVACY_PANES: Dict[str, str] = {
    "accessibility": (
        "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
    ),
    "input_monitoring": (
        "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent"
    ),
    "screen_recording": (
        "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"
    ),
}


def accessibility_trusted() -> Optional[bool]:
    """True/False when macOS can tell us, ``None`` elsewhere or without PyObjC."""
    if not IS_MACOS:
        return None
    try:
        from ApplicationServices import AXIsProcessTrusted
    except Exception:  # PyObjC missing
        log.debug("AXIsProcessTrusted unavailable", exc_info=True)
        return None
    try:
        return bool(AXIsProcessTrusted())
    except Exception:  # pragma: no cover - defensive
        log.debug("AXIsProcessTrusted failed", exc_info=True)
        return None


def input_monitoring_trusted() -> Optional[bool]:
    """Whether this process may listen to keyboard events (TCC ``ListenEvent``)."""
    if not IS_MACOS:
        return None
    try:
        import Quartz
    except Exception:
        log.debug("Quartz unavailable", exc_info=True)
        return None
    preflight = getattr(Quartz, "CGPreflightListenEventAccess", None)
    if preflight is None:  # macOS < 10.15
        return None
    try:
        return bool(preflight())
    except Exception:  # pragma: no cover - defensive
        log.debug("CGPreflightListenEventAccess failed", exc_info=True)
        return None


def request_permissions(accessibility: bool = True, input_monitoring: bool = True) -> None:
    """Ask macOS to show its permission dialogs (each appears once per process)."""
    if not IS_MACOS:
        return
    if accessibility:
        try:
            from ApplicationServices import (  # noqa: F401
                AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt,
            )
            AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True})
        except Exception:
            log.debug("Could not prompt for Accessibility", exc_info=True)
    if input_monitoring:
        try:
            import Quartz
            request = getattr(Quartz, "CGRequestListenEventAccess", None)
            if request is not None:
                request()
        except Exception:
            log.debug("Could not prompt for Input Monitoring", exc_info=True)


def open_privacy_pane(kind: str = "accessibility") -> bool:
    """Open System Settings on the given privacy pane. Returns True when opened."""
    if not IS_MACOS:
        return False
    url = PRIVACY_PANES.get(kind, PRIVACY_PANES["accessibility"])
    try:
        subprocess.Popen(["open", url])
        return True
    except Exception:
        log.warning("Could not open %s", url, exc_info=True)
        return False



def set_dock_icon_visible(visible: bool) -> bool:
    """Show/hide the Dock icon (menu-bar-only mode, like Docker Dashboard).

    ``Accessory`` keeps the process running with only a menu-bar item; ``Regular``
    brings the normal Dock icon and app-switcher entry back.
    """
    if not IS_MACOS:
        return False
    try:
        from AppKit import NSApplication  # type: ignore
    except Exception:
        log.debug("AppKit unavailable", exc_info=True)
        return False
    accessory, regular = 1, 0  # NSApplicationActivationPolicy*
    try:
        from AppKit import (  # type: ignore
            NSApplicationActivationPolicyAccessory,
            NSApplicationActivationPolicyRegular,
        )
        accessory = NSApplicationActivationPolicyAccessory
        regular = NSApplicationActivationPolicyRegular
    except Exception:  # older PyObjC without the named constants
        pass
    try:
        app = NSApplication.sharedApplication()
        if app is None:
            return False
        app.setActivationPolicy_(regular if visible else accessory)
        if visible:
            app.activateIgnoringOtherApps_(True)
        return True
    except Exception:  # pragma: no cover - defensive
        log.debug("Could not change the Dock icon policy", exc_info=True)
        return False


def system_key_age() -> Optional[float]:
    """Seconds since the OS last saw *any* key press, system-wide.

    An independent witness: if macOS knows about a key press but the bot's
    listener never saw it, the event tap was disabled (or is being blocked by a
    secure-input app). Off macOS this returns ``None``.
    """
    if not IS_MACOS:
        return None
    try:
        import Quartz
    except Exception:
        return None
    try:
        return float(Quartz.CGEventSourceSecondsSinceLastEventType(
            Quartz.kCGEventSourceStateHIDSystemState, Quartz.kCGEventKeyDown
        ))
    except Exception:  # pragma: no cover - defensive
        return None


def permission_report() -> str:
    """One-line summary of the macOS permissions the bot depends on."""
    if not IS_MACOS:
        return "not macOS - no permissions required"
    parts = []
    for label, value in (
        ("accessibility", accessibility_trusted()),
        ("input monitoring", input_monitoring_trusted()),
    ):
        state = "granted" if value else ("DENIED" if value is False else "unknown")
        parts.append(f"{label}: {state}")
    return ", ".join(parts)


def hotkey_permission_hint() -> str:
    """Plain-language hint shown when global hotkeys cannot receive events."""
    if not IS_MACOS:
        return ""
    trusted = input_monitoring_trusted()
    if trusted is True:
        return ""
    who = "the app that launched the bot (Terminal, iTerm, VS Code, ...)"
    if trusted is False:
        return (
            f"macOS blocks the global hotkeys: enable {who} in System Settings > "
            "Privacy & Security > Input Monitoring (and Accessibility), then restart "
            "the bot."
        )
    return (
        f"Global hotkeys need Input Monitoring permission for {who}: System Settings "
        "> Privacy & Security > Input Monitoring."
    )
