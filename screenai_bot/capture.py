"""Screen geometry + capture for Windows / macOS / Linux (X11 & Wayland fallbacks).

``mss`` gives us the *virtual desktop* coordinate space, which is what we need:
regions are stored in absolute coordinates spanning all monitors, so a region
selected on monitor 2 still captures correctly after a restart.
"""

from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .config import Region

log = logging.getLogger("screenai.capture")

try:  # pragma: no cover - import guard for headless installs
    import mss
    from PIL import Image
except Exception as exc:  # pragma: no cover
    mss = None
    Image = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass
class Monitor:
    index: int
    left: int
    top: int
    width: int
    height: int
    name: str = ""

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def contains(self, x: int, y: int) -> bool:
        return self.left <= x < self.right and self.top <= y < self.bottom

    def clamp(self, region: Region) -> Region:
        left = max(region.left, self.left)
        top = max(region.top, self.top)
        right = min(region.right, self.right)
        bottom = min(region.bottom, self.bottom)
        return Region(region.name, left, top, max(1, right - left), max(1, bottom - top))

    def to_qt(self) -> Tuple[int, int, int, int]:
        return (self.left, self.top, self.width, self.height)


class ScreenCapture:
    """Thread-safe screen grabber.

    ``mss`` sessions are bound to the thread that created them, so we keep one
    session per thread instead of sharing a global instance.
    """

    def __init__(self) -> None:
        if mss is None:
            raise RuntimeError(
                "mss/Pillow are required for screen capture. "
                f"Install with `pip install mss Pillow` (error: {_IMPORT_ERROR})."
            )
        self._local = threading.local()
        self._monitors: List[Monitor] = []
        self.refresh_monitors()

    # ------------------------------------------------------------------ session
    @property
    def _sct(self) -> "mss.MSS":
        sct = getattr(self._local, "sct", None)
        if sct is None:
            try:
                sct = mss.mss()
            except Exception as exc:  # Wayland without XWayland, headless, etc.
                raise RuntimeError(
                    "Could not open a screen-capture session. On Linux/Wayland run "
                    "under XWayland (e.g. `QT_QPA_PLATFORM=xcb`) or use the X11 session."
                ) from exc
            self._local.sct = sct
        return sct

    # ---------------------------------------------------------------- monitors
    def refresh_monitors(self) -> List[Monitor]:
        raw = self._sct.monitors
        out: List[Monitor] = []
        for idx, mon in enumerate(raw):
            if idx == 0:
                name = "ALL"
            else:
                name = f"Monitor {idx}"
            out.append(
                Monitor(
                    index=idx,
                    left=int(mon["left"]),
                    top=int(mon["top"]),
                    width=int(mon["width"]),
                    height=int(mon["height"]),
                    name=name,
                )
            )
        self._monitors = out
        log.info("Detected monitors: %s", [(m.name, m.to_qt()) for m in out])
        return out

    @property
    def monitors(self) -> List[Monitor]:
        return self._monitors or self.refresh_monitors()

    @property
    def virtual_bounds(self) -> Monitor:
        whole = self.monitors[0]
        return whole

    def monitor_for(self, region: Region) -> Monitor:
        cx, cy = region.center
        for mon in self.monitors[1:] or self.monitors:
            if mon.contains(cx, cy):
                return mon
        return self.monitors[0]

    def monitor_at(self, x: int, y: int) -> Monitor:
        for mon in self.monitors[1:] or self.monitors:
            if mon.contains(x, y):
                return mon
        return self.monitors[0]

    def fullscreen_region(self) -> Region:
        whole = self.monitors[0]
        return Region("full-desktop", whole.left, whole.top, whole.width, whole.height)

    # ----------------------------------------------------------------- capture
    def grab(self, region: Region) -> "Image.Image":
        """Capture ``region`` and return a PIL image (RGB)."""
        region = region.normalized()
        if region.width < 2 or region.height < 2:
            raise ValueError(f"Region too small to capture: {region}")
        box = {"left": region.left, "top": region.top,
               "width": region.width, "height": region.height}
        shot = self._sct.grab(box)
        return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

    def grab_point_color(self, x: int, y: int) -> Tuple[int, int, int]:
        tiny = Region("probe", x, y, 1, 1).normalized()
        img = self.grab(tiny)
        return img.getpixel((0, 0))[:3]

    def close(self) -> None:
        sct = getattr(self._local, "sct", None)
        if sct is not None:
            try:
                sct.close()
            except Exception:
                pass
            self._local.sct = None


# --------------------------------------------------------------------- helpers
def scale_factor() -> float:
    """Return the OS/UI scale factor (Retina/HiDPI) when detectable.

    Advisory only: OCR accuracy improves when small text is upscaled, so the OCR
    layer uses this to decide how much magnification to apply.
    """
    import os

    if sys.platform == "darwin":
        try:
            from AppKit import NSScreen  # type: ignore

            backing = float(NSScreen.mainScreen().backingScaleFactor())
            if backing > 0:
                return backing
        except Exception:
            pass
    for var in ("QT_SCALE_FACTOR", "GDK_SCALE"):
        try:
            value = float(os.environ.get(var, "") or 0)
            if value > 0:
                return value
        except ValueError:
            continue
    return 1.0
