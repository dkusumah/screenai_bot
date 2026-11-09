"""Menu-bar / system-tray presence so the bot can keep running in the background.

Closing the panel no longer kills the bot: it drops to a *status item* (macOS
menu bar, Windows notification area, Linux tray) exactly like Docker Dashboard.
The icon is drawn here in code, so there is no image file to install, and on
macOS it is a *template* image - the system recolours it for light/dark modes.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Slot
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from .native import IS_MACOS

log = logging.getLogger("screenai.tray")


def make_icon(size: int = 22, color: str = "#ffffff") -> QIcon:
    """Draw the bot glyph: a screen with two eyes (no asset files needed).

    macOS menu bars expect a template image, so the artwork is monochrome and
    marked with ``setIsMask`` - macOS then inverts it for light/dark mode.
    """
    scale = 2  # render at 2x so retina displays get a crisp icon
    pixmap = QPixmap(size * scale, size * scale)
    # With a device pixel ratio set, QPainter already works in logical (22px)
    # coordinates - scaling again here would draw the glyph 4x too large.
    pixmap.setDevicePixelRatio(scale)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen(QColor(color), max(1.4, size * 0.085))
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        # Monitor outline
        painter.drawRoundedRect(QRectF(2.2, 4.0, 17.6, 12.2), 2.6, 2.6)
        # Stand
        painter.drawLine(QPointF(8.4, 17.8), QPointF(13.6, 17.8))
        # Eyes
        painter.setBrush(QColor(color))
        painter.setPen(Qt.NoPen)
        eye = max(1.1, size * 0.055)
        painter.drawEllipse(QRectF(6.8, 9.2, eye * 2, eye * 2))
        painter.drawEllipse(QRectF(13.0, 9.2, eye * 2, eye * 2))
    finally:
        painter.end()
    icon = QIcon(pixmap)
    try:
        icon.setIsMask(True)  # macOS: follow the menu-bar light/dark appearance
    except Exception:  # pragma: no cover - older Qt without template support
        pass
    return icon


class TrayController(QObject):
    """Owns the status icon and its menu; the window supplies the slots."""

    def __init__(self, parent=None, title: str = "ScreenAI Bot") -> None:
        super().__init__(parent)
        self.title = title
        self.available = QSystemTrayIcon.isSystemTrayAvailable()
        self._actions: Dict[str, QAction] = {}
        self._hint_shown = False
        self.icon = QSystemTrayIcon(make_icon(), self)
        self.icon.setToolTip(title)
        self.menu = QMenu()
        self.icon.setContextMenu(self.menu)
        self.icon.activated.connect(self._on_activated)
        self.on_open = None  # callable set by the window (icon click / double click)

    # ------------------------------------------------------------------- building
    def add_action(self, key: str, text: str, slot, tooltip: str = "") -> Optional[QAction]:
        action = QAction(text, self.menu)
        if tooltip:
            action.setToolTip(tooltip)
        action.triggered.connect(lambda _checked=False, s=slot: s())
        self.menu.addAction(action)
        self._actions[key] = action
        return action

    def add_separator(self) -> None:
        self.menu.addSeparator()

    def set_text(self, key: str, text: str) -> None:
        action = self._actions.get(key)
        if action is not None and action.text() != text:
            action.setText(text)

    def set_action_enabled(self, key: str, enabled: bool) -> None:
        action = self._actions.get(key)
        if action is not None:
            action.setEnabled(bool(enabled))

    # ------------------------------------------------------------------ lifecycle
    def show(self) -> bool:
        if not self.available:
            return False
        self.icon.show()
        return True

    def hide(self) -> None:
        self.icon.hide()

    def set_tooltip(self, text: str) -> None:
        self.icon.setToolTip(f"{self.title}\n{text}" if text else self.title)

    def notify(self, message: str, detail: str = "") -> bool:
        """Show a tray balloon. macOS does not support these (Qt ignores them)."""
        if not self.available:
            return False
        try:
            self.icon.showMessage(message, detail, self.icon.icon(), 4000)
            return True
        except Exception:  # pragma: no cover - platform specific
            log.debug("Tray notification failed", exc_info=True)
            return False

    @Slot(QSystemTrayIcon.ActivationReason)
    def _on_activated(self, reason=None) -> None:
        """Open the panel from the icon (macOS shows the menu on a click instead)."""
        if reason is None:
            return
        # On macOS a single click already opens the context menu, so only a
        # double click reopens the window there; elsewhere a plain click does.
        wanted = ((QSystemTrayIcon.DoubleClick,) if IS_MACOS
                  else (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick))
        if reason not in wanted:
            return
        if callable(self.on_open):
            self.on_open()
