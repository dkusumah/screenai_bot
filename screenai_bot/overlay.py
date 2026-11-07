"""Frameless, click-through-ish overlays drawn on top of everything.

Two widgets live here:

* :class:`RegionSelector` - drag-a-box overlay used to define a screen area
  (multi-monitor aware; supports arrow-key nudging and Esc to cancel).
* :class:`RegionMarker`   - persistent thin outline that shows where the saved
  regions currently are, so you can see what the bot will read.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from .config import Region


class RegionSelector(QWidget):
    """Semi-transparent full-screen overlay for drawing a rectangle."""

    selected = Signal(object)   # Region
    cancelled = Signal()

    MIN_SIZE = 8

    def __init__(self, title: str = "Drag to select a screen area",
                 preset: Optional[Region] = None, hint: str = "") -> None:
        super().__init__()
        self.setWindowTitle("ScreenAI - Select Region")
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        self._title = title
        self._hint = hint or "Drag a box  |  Arrow keys nudge  |  Enter confirm  |  Esc cancel"
        self._origin: Optional[Tuple[int, int]] = None
        self._current = QRect()
        self._preset = preset
        self._presets: List[Region] = []
        self.setGeometry(_virtual_desktop_geometry())
        self.showFullScreen()
        self.activateWindow()
        self.raise_()

    # --------------------------------------------------------------------- state
    @property
    def selection_rect(self) -> QRect:
        return self._current.normalized() if self._current else QRect()

    def set_presets(self, regions: List[Region]) -> None:
        self._presets = list(regions)
        self.update()

    # ---------------------------------------------------------------------- events
    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._origin = (int(event.globalPosition().x()), int(event.globalPosition().y()))
            self._current = QRect(self._map_from_global(self._origin), QPoint(0, 0))
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._origin is None:
            return
        point = self._map_from_global((int(event.globalPosition().x()), int(event.globalPosition().y())))
        start = self._map_from_global(self._origin)
        self._current = QRect(start, point)
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton or self._origin is None:
            return
        self._origin = None
        rect = self._current.normalized()
        if rect.width() >= self.MIN_SIZE and rect.height() >= self.MIN_SIZE:
            self.selected.emit(self._to_region(rect))
            self.close()
        else:
            self.update()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        if key == Qt.Key_Escape:
            self.cancelled.emit()
            self.close()
        elif key in (Qt.Key_Return, Qt.Key_Enter):
            rect = self._current.normalized()
            if rect.width() >= self.MIN_SIZE and rect.height() >= self.MIN_SIZE:
                self.selected.emit(self._to_region(rect))
                self.close()
        elif key in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down) and self._current:
            step = 1 if not event.modifiers() & Qt.ShiftModifier else 10
            dx = {Qt.Key_Left: -step, Qt.Key_Right: step}.get(key, 0)
            dy = {Qt.Key_Up: -step, Qt.Key_Down: step}.get(key, 0)
            self._current.translate(dx, dy)
            self.update()
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------- painting
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 130))
        rect = self._current.normalized()
        if rect.isValid() and not rect.isEmpty():
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.fillRect(rect, Qt.black)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            pen = QPen(QColor(80, 200, 255), 2)
            painter.setPen(pen)
            painter.drawRect(rect)
            self._draw_magnifier_label(painter, rect)
        for region in self._presets:
            mapped = self._map_region_local(region)
            painter.setPen(QPen(QColor(255, 200, 60, 190), 1, Qt.DashLine))
            painter.drawRect(mapped)
        if self._preset:
            painter.setPen(QPen(QColor(120, 255, 160), 2, Qt.DashLine))
            painter.drawRect(self._map_region_local(self._preset))
        self._draw_hud(painter, rect)

    def _draw_hud(self, painter: QPainter, rect: QRect) -> None:
        painter.setFont(QFont("Segoe UI", 13, QFont.Bold))
        painter.setPen(QColor(240, 240, 240))
        painter.drawText(self.rect().adjusted(0, 40, 0, 0), Qt.AlignHCenter | Qt.AlignTop, self._title)
        painter.setFont(QFont("Segoe UI", 10))
        painter.setPen(QColor(200, 200, 200))
        painter.drawText(self.rect().adjusted(0, 70, 0, 0), Qt.AlignHCenter | Qt.AlignTop, self._hint)
        if rect and rect.width() > 4:
            offset = self._offset()
            info = (f"{rect.width()} x {rect.height()} px   "
                    f"at ({rect.left() + offset.x()},{rect.top() + offset.y()})")
            painter.setPen(QColor(80, 200, 255))
            painter.drawText(rect.adjusted(0, -26, 0, -6), Qt.AlignLeft | Qt.AlignVCenter, info)

    def _draw_magnifier_label(self, painter: QPainter, rect: QRect) -> None:
        pass

    # -------------------------------------------------------------- coord helpers
    def _offset(self) -> QPoint:
        return _virtual_desktop_geometry().topLeft()

    def _map_from_global(self, point: Tuple[int, int]) -> QPoint:
        offset = self._offset()
        return QPoint(int(point[0]) - offset.x(), int(point[1]) - offset.y())

    def _to_global(self, local_value: int) -> int:
        """Convert a widget-local coordinate to virtual-desktop (global) coordinates."""
        return local_value + self._offset().x()

    def _map_region_local(self, region: Region) -> QRect:
        offset = self._offset()
        return QRect(region.left - offset.x(), region.top - offset.y(), region.width, region.height)

    def _to_region(self, rect: QRect) -> Region:
        offset = self._offset()
        return Region(
            name="selection",
            left=rect.x() + offset.x(),
            top=rect.y() + offset.y(),
            width=rect.width(),
            height=rect.height(),
        )


class RegionMarker(QWidget):
    """Non-interactive outlines showing saved regions on the desktop."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool | Qt.WindowTransparentForInput
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setGeometry(_virtual_desktop_geometry())
        self._regions: List[Tuple[Region, str]] = []
        self._active: Optional[str] = None

    def set_regions(self, regions: List[Region], active_name: str = "") -> None:
        self._regions = [(r, r.name) for r in regions]
        self._active = active_name
        self.setGeometry(_virtual_desktop_geometry())
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        offset = _virtual_desktop_geometry().topLeft()
        for region, name in self._regions:
            rect = QRect(region.left - offset.x(), region.top - offset.y(),
                         region.width, region.height)
            active = name == self._active
            color = QColor(80, 200, 255) if active else QColor(150, 150, 150, 160)
            painter.setPen(QPen(color, 2 if active else 1, Qt.DashLine if not active else Qt.SolidLine))
            painter.drawRect(rect)
            if active:
                painter.fillRect(rect, QColor(80, 200, 255, 24))
            painter.setPen(color)
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.drawText(rect.adjusted(4, 2, 0, 0), Qt.AlignLeft | Qt.AlignTop, name)


def _virtual_desktop_geometry() -> QRect:
    """Bounding rect of all screens, in *global* coordinates (multi-monitor safe)."""
    try:
        union = QRect()
        for screen in QApplication.screens():
            union = union.united(screen.geometry())
        if union.isValid() and not union.isEmpty():
            return union
    except Exception:
        pass
    primary = QApplication.primaryScreen()
    return primary.geometry() if primary else QRect(0, 0, 1920, 1080)
