"""Main control window (PySide6).

Layout:
  * top bar: region picker / task / mode, Run + Stop buttons
  * left:    captured preview + saved regions list
  * right:   OCR text (editable), AI answer (editable), action log
  * bottom:  status bar with hotkey hints

Everything the worker does is driven from here; the pipeline runs in a QThread so
the UI never blocks while capturing, waiting for OCR or calling the API.
"""

from __future__ import annotations

import io
import logging
import threading
import time
from typing import List, Optional

from PySide6.QtCore import (
    Q_ARG, QMetaObject, QObject, QPoint, QPointF, QRect, QSize, Qt, QThread, QTimer,
    Signal, Slot,
)
from PySide6.QtGui import QColor, QFontDatabase, QIcon, QPainter, QPen, QPixmap
# QAction and QShortcut live in QtGui since Qt 6 (they are NOT widgets), but
# some older PySide6 releases re-exported them from QtWidgets. Import from the
# canonical location with a fallback so both old and new PySide6 work.
try:
    from PySide6.QtGui import QAction, QShortcut
except ImportError:  # pragma: no cover - legacy PySide6 (< ~6.4)
    from PySide6.QtWidgets import QAction, QShortcut

from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton,
    QSizePolicy, QStatusBar, QTabWidget, QTextEdit, QToolButton, QVBoxLayout, QWidget,
)

from .actions import ActionPlanner
from .config import APP_DIR, BUNDLED_CONFIG, DEFAULTS, Config, Region, deep_merge, load_config
from .ocr import OcrResult
from .overlay import ClickFlash, RegionMarker, RegionSelector
from .pipeline import PipelineRequest, PipelineWorker, preview_prompt

log = logging.getLogger("screenai.ui")

# Chevrons for the collapsible section headers (kept as constants so the
# f-strings below stay valid on Python 3.10/3.11, where escapes are not
# allowed inside f-string expressions).
_ARROW_OPEN = "\u25be"
_ARROW_CLOSED = "\u25b8"


class ChevronComboBox(QComboBox):
    """Combo box with a Bootstrap/Select2 style chevron instead of a button.

    Qt paints its own boxy drop-down button. We keep the click area (so the
    popup still opens by clicking the right-hand side) but hide the button
    chrome and the built-in arrow, then draw a thin chevron ourselves so the
    control blends into the rest of the dark theme.
    """

    CHEVRON = QColor("#93a4bb")
    CHEVRON_HOVER = QColor("#dbe4f0")

    CHEVRON_STYLE = (
        "QComboBox { padding-right: 26px; }"
        "QComboBox::drop-down {"
        " subcontrol-origin: padding; subcontrol-position: center right;"
        " width: 24px; border: none; background: transparent; }"
        "QComboBox::down-arrow { image: none; width: 0px; height: 0px; }"
        "QComboBox QLineEdit { background: transparent; border: none;"
        " padding-right: 22px; }"
    )

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMaxVisibleItems(14)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(self.CHEVRON_STYLE)
        self._hovered = False

    # ------------------------------------------------------------------ events
    def enterEvent(self, event) -> None:  # noqa: N802
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        colour = self.CHEVRON_HOVER if self._hovered else self.CHEVRON
        if not self.isEnabled():
            colour = QColor("#5c6678")
        pen = QPen(colour, 1.7)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        centre_x = float(self.rect().right() - 14)
        centre_y = float(self.rect().center().y()) + 0.5
        half_width, half_height = 4.2, 2.4
        painter.drawLine(
            QPointF(centre_x - half_width, centre_y - half_height),
            QPointF(centre_x, centre_y + half_height),
        )
        painter.drawLine(
            QPointF(centre_x, centre_y + half_height),
            QPointF(centre_x + half_width, centre_y - half_height),
        )
        painter.end()


def pil_to_pixmap(image) -> QPixmap:
    """PIL image -> QPixmap without depending on Qt image plugins."""
    from PIL import Image as PILImage

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    pixmap = QPixmap()
    pixmap.loadFromData(buffer.getvalue())
    del PILImage
    return pixmap


def _mask_secret(value: str) -> str:
    """Return a masked placeholder so keys/tokens never leak into diagnostics."""
    value = (value or "").strip()
    if not value:
        return ""
    return f"<<masked {len(value)}-char secret>>"


class WorkerThread(QThread):
    """Runs PipelineWorker.run() off the GUI thread."""

    def __init__(self, worker: PipelineWorker, request: PipelineRequest, parent=None) -> None:
        super().__init__(parent)
        self.worker = worker
        self.request = request

    def run(self) -> None:  # noqa: D102 - QThread override
        self.worker.run(self.request)


# Per-provider defaults used by the AI settings row (provider -> base URL, the
# env var to fall back on, and a few model suggestions for the editable combo).
PROVIDER_DEFAULTS = {
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1", "o4-mini"],
    },
    "anthropic": {
        "label": "Anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "api_key_env": "ANTHROPIC_API_KEY",
        "models": [
            "claude-3-5-haiku-latest",
            "claude-3-5-sonnet-latest",
            "claude-sonnet-4-5",
            "claude-opus-4-1",
        ],
    },
    "gemini": {
        "label": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "api_key_env": "GEMINI_API_KEY",
        "models": [
            "gemini-3.8-flash",
            "gemini-flash-latest",
            "gemini-3.7-flash",
            "gemini-3.5-flash",
            "gemini-2.5-flash",
            "gemini-pro-latest",
            "gemini-2.5-pro",
        ],
    },
}


class MainWindow(QMainWindow):
    THEME = """
    QMainWindow, QDialog { background: #101419; }
    QWidget { color: #dbe4f0; font-size: 13px; }
    QLabel#appTitle { font-size: 21px; font-weight: 700; color: #f4f8ff; }
    QLabel#appHint { color: #8492a6; }
    QGroupBox {
        background: #151a21; border: 1px solid #232b37; border-radius: 10px;
        margin-top: 15px; padding: 12px 10px 10px 10px;
    }
    QGroupBox::title {
        subcontrol-origin: margin; subcontrol-position: top left; left: 12px;
        top: -2px; padding: 0 5px; color: #a8bbd4; font-weight: 600;
    }
    QFrame#navBar { background: #0b0f14; border-bottom: 1px solid #1c232d; }
    QToolButton#navButton {
        background: #161d27; border: 1px solid #263041; border-radius: 8px;
        padding: 8px 14px; color: #c8d4e4; font-weight: 600;
    }
    QToolButton#navButton:hover { background: #1e2734; border-color: #38465a; color: #ffffff; }
    QToolButton#navButton:pressed { background: #131a23; }
    QToolButton#navButton:checked {
        background: #1d3f6e; border-color: #3b82f6; color: #ffffff;
    }
    QFrame#navSeparator { background: #232b37; border: none; }
    QPushButton#sectionHeader {
        text-align: left; color: #cfe0ff; font-weight: 600; font-size: 13px;
        background: #141a22; border: 1px solid #212a36; border-radius: 8px;
        padding: 9px 14px;
    }
    QPushButton#sectionHeader:hover { background: #1a222d; border-color: #2f3b4b; }
    QPushButton#sectionHeader:checked { background: #141a22; border-color: #212a36; }
    QPushButton {
        background: #1d2531; border: 1px solid #2b3646; border-radius: 8px;
        padding: 7px 14px; color: #e3eaf5;
    }
    QPushButton:hover { background: #26303f; border-color: #3a4759; }
    QPushButton:pressed { background: #171e28; }
    QPushButton:disabled { color: #5c6678; background: #151a21; border-color: #222a35; }
    QPushButton#primary {
        background: #2563eb; border: 1px solid #3b82f6; border-radius: 10px;
        color: #ffffff; font-size: 15px; font-weight: 700; padding: 10px 26px;
    }
    QPushButton#primary:hover { background: #3172f5; }
    QPushButton#primary:disabled { background: #1b2a44; border-color: #24344f; color: #6d7f9b; }
    QPushButton#danger {
        background: #3a1d24; border: 1px solid #7f2b3a; color: #ffb3c0;
        font-weight: 600; padding: 10px 18px; border-radius: 10px;
    }
    QPushButton#danger:hover { background: #4a2029; border-color: #a03848; }
    QPushButton#danger:disabled { background: #1d171a; border-color: #33242a; color: #6f565e; }
    QComboBox, QLineEdit, QPlainTextEdit, QTextEdit, QListWidget {
        background: #0d1117; border: 1px solid #262f3d; border-radius: 8px;
        padding: 6px 8px; selection-background-color: #2563eb; selection-color: #ffffff;
    }
    QComboBox:hover, QLineEdit:hover { border-color: #334051; }
    QComboBox:focus, QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus {
        border-color: #3b82f6;
    }
    QComboBox QAbstractItemView {
        background: #12171f; border: 1px solid #2b3646; selection-background-color: #2563eb;
    }
    QCheckBox { spacing: 8px; color: #cbd6e5; }
    QCheckBox::indicator, QGroupBox::indicator {
        width: 15px; height: 15px; border-radius: 4px;
        border: 1px solid #38465a; background: #0d1117;
    }
    QCheckBox::indicator:checked { background: #2563eb; border-color: #3b82f6; }
    QTabWidget::pane { border: 1px solid #232b37; border-radius: 8px; top: -1px; }
    QTabBar::tab {
        background: transparent; color: #8b9ab0; padding: 6px 10px;
        border-radius: 7px; margin-right: 2px;
    }
    QTabBar::tab:hover { color: #cfdcee; }
    QTabBar::tab:selected { color: #ffffff; background: #1d2531; }
    QTabBar::scroller { width: 0px; }
    QTabBar QToolButton {
        background: #1d2531; border: 1px solid #2b3646; border-radius: 6px; padding: 2px;
    }
    QStatusBar { color: #8492a6; background: #0c1015; }
    QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
    QScrollBar::handle:vertical { background: #2b3646; border-radius: 5px; min-height: 26px; }
    QScrollBar::handle:vertical:hover { background: #3a4759; }
    QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
    """
    MODE_LABELS = {
        "type_in_region": "Click Inside Area, Clear It, Type Answer",
        "click_and_type": "Click Target Once, Then Type",
        "double_click_and_type": "Double-Click Target, Then Type",
        "click_matching_line": "Click The Line Matching The AI Answer (MCQ)",
        "type_at_cursor": "Type At Current Cursor (No Click)",
        "click_only": "Just Click The Area (No Typing)",
        "press_enter_after": "Type Answer, Then Press Enter",
        "copy_only": "Copy Answer To Clipboard Only",
    }

    def __init__(self, cfg: Optional[Config] = None) -> None:
        super().__init__()
        self.cfg = cfg or load_config()
        self.setWindowTitle("ScreenAI Bot - read screen, ask AI, act")
        self.resize(1180, 760)
        self.worker = PipelineWorker(self.cfg)
        self._hidden_for_action = False
        self._hide_gen = 0
        self._hide_event: Optional[threading.Event] = None
        self._capture_panel_rect: Optional[tuple] = None
        # Let the worker ask us to move our own window out of the way before it
        # captures the screen or sends input (otherwise the screenshot and the
        # click both hit the bot window instead of the app underneath).
        self.worker.pre_action_hook = self._hide_for_input_sync
        self.worker.post_action_hook = self._restore_input_window_async
        self.worker.pre_capture_hook = self._hide_for_capture_sync
        self.worker.post_capture_hook = self._restore_input_window_async
        self.worker.panel_rect_provider = self._panel_rect
        self.thread: Optional[WorkerThread] = None
        self.selector: Optional[RegionSelector] = None
        self.marker = RegionMarker()
        self.click_flash = ClickFlash()
        self.source_region: Optional[Region] = self.cfg.regions[0] if self.cfg.regions else None
        self.target_region: Optional[Region] = None
        self.last_image = None
        self._busy = False

        self._build_ui()
        self._connect_worker()
        self._register_hotkeys()
        self._refresh_regions()
        QTimer.singleShot(400, self._health_check)

    # ------------------------------------------------------------------- UI build
    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)

        root.addWidget(self._build_nav_bar())

        # --- title + one-line instructions -----------------------------------
        title = QLabel("ScreenAI Bot")
        title.setObjectName("appTitle")
        root.addWidget(title)
        hint = QLabel("Select an area to read, choose what to do, then press Run.")
        hint.setObjectName("appHint")
        root.addWidget(hint)

        # --- run bar: the two choices that matter, plus the main actions -----
        controls = QHBoxLayout()
        controls.setSpacing(8)
        controls.addWidget(QLabel("Task"))
        self.task_combo = ChevronComboBox()
        tasks = self.cfg.get_path("prompt.tasks", {}) or {}
        for key in tasks.keys():
            self.task_combo.addItem(f"{key}", key)
        self.task_combo.setMinimumWidth(170)
        controls.addWidget(self.task_combo)
        controls.addWidget(QLabel("Action"))
        self.mode_combo = ChevronComboBox()
        for key, label in self.MODE_LABELS.items():
            self.mode_combo.addItem(label, key)
        self.mode_combo.setMinimumWidth(300)
        controls.addWidget(self.mode_combo)
        # Restore the previous Action/Task choice before wiring the change
        # handlers. Without this the dropdown silently falls back to
        # "type_in_region" (a click at the region center) on every launch.
        self._restore_ui_selection()
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.task_combo.currentIndexChanged.connect(self._on_task_changed)
        controls.addStretch(1)

        # All three main actions share one size so the row reads as a group.
        action_size = QSize(132, 40)
        self.repeat_btn = QPushButton("Repeat Last")
        self.repeat_btn.setFixedSize(action_size)
        self.repeat_btn.setToolTip("Run again with the same settings.")
        self.repeat_btn.clicked.connect(self.run_pipeline)
        controls.addWidget(self.repeat_btn)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setFixedSize(action_size)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_pipeline)
        controls.addWidget(self.stop_btn)
        self.run_btn = QPushButton("Run")
        self.run_btn.setObjectName("primary")
        self.run_btn.setFixedSize(action_size)
        self.run_btn.setToolTip("Read the area, ask the AI and act on the answer.")
        self.run_btn.clicked.connect(self.run_pipeline)
        controls.addWidget(self.run_btn)
        root.addLayout(controls)

        # --- collapsible Options: every behaviour switch lives here ----------
        root.addWidget(
            self._make_section("Options", self._build_options_body(),
                               "ui.show_options", False)
        )

        # --- collapsible AI settings (expanded if no key is configured yet) ---
        root.addWidget(
            self._make_section("AI Settings", self._build_ai_body(), "ui.show_ai",
                               not bool(self.cfg.api_key))
        )

        # --- fixed three-column layout (no draggable handles) ----------------
        content = QHBoxLayout()
        content.setSpacing(12)

        left = QWidget()
        left.setFixedWidth(300)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        areas_box = QGroupBox("Areas")
        areas_lay = QVBoxLayout(areas_box)
        areas_lay.setSpacing(6)
        top_hint = QLabel("Top item is the default read area.")
        top_hint.setObjectName("appHint")
        areas_lay.addWidget(top_hint)
        self.region_list = QListWidget()
        self.region_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.region_list.itemDoubleClicked.connect(lambda *_: self.apply_selected_region())
        areas_lay.addWidget(self.region_list, 1)
        row = QHBoxLayout()
        add_btn = QPushButton("Add Area")
        add_btn.clicked.connect(self.add_current_region)
        rename_btn = QPushButton("Rename")
        rename_btn.clicked.connect(self.rename_region)
        del_btn = QPushButton("Delete")
        del_btn.clicked.connect(self.delete_region)
        use_btn = QPushButton("Use As Source")
        use_btn.clicked.connect(self.apply_selected_region)
        pick_btn = QPushButton("Pick Target Here")
        pick_btn.clicked.connect(self.use_selected_as_target)
        for button in (add_btn, rename_btn, del_btn):
            row.addWidget(button)
        areas_lay.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(use_btn)
        row2.addWidget(pick_btn)
        areas_lay.addLayout(row2)
        left_lay.addWidget(areas_box, 1)

        preview_box = QGroupBox("Captured Preview")
        preview_lay = QVBoxLayout(preview_box)
        self.preview = QLabel("No capture yet")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(140)
        self.preview.setStyleSheet(
            "background:#0b0f14;color:#7d8798;border:1px dashed #2a3140;border-radius:8px;"
        )
        preview_lay.addWidget(self.preview, 1)
        left_lay.addWidget(preview_box, 1)
        content.addWidget(left)

        middle = QWidget()
        mid_lay = QVBoxLayout(middle)
        mid_lay.setContentsMargins(0, 0, 0, 0)
        read_box = QGroupBox("Text Read From Screen (Editable)")
        read_lay = QVBoxLayout(read_box)
        self.ocr_view = QPlainTextEdit()
        self.ocr_view.setPlaceholderText("Select an area and press Run...")
        read_lay.addWidget(self.ocr_view, 1)
        mid_lay.addWidget(read_box, 1)

        answer_box = QGroupBox("AI Answer (Edit Before Applying)")
        answer_lay = QVBoxLayout(answer_box)
        self.answer_view = QTextEdit()
        self.answer_view.setPlaceholderText("The model's reply appears here.")
        answer_lay.addWidget(self.answer_view, 1)
        btns = QHBoxLayout()
        self.ask_btn = QPushButton("Ask AI With Edited Text")
        self.ask_btn.clicked.connect(self.ask_with_edited_text)
        self.apply_btn = QPushButton("Apply Answer Now")
        self.apply_btn.clicked.connect(self.apply_answer_now)
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.clicked.connect(self.copy_answer)
        self.prompt_btn = QPushButton("Preview Prompt")
        self.prompt_btn.clicked.connect(self.show_prompt)
        for button in (self.ask_btn, self.apply_btn, self.copy_btn, self.prompt_btn):
            btns.addWidget(button)
        answer_lay.addLayout(btns)
        mid_lay.addWidget(answer_box, 1)
        content.addWidget(middle, 1)

        self.log_panel = QGroupBox("Activity")
        self.log_panel.setMinimumWidth(330)
        self.log_panel.setMaximumWidth(420)
        log_lay = QVBoxLayout(self.log_panel)
        tabs = QTabWidget()
        tabs.setUsesScrollButtons(False)   # shrink the tabs instead of a scroll arrow
        tabs.tabBar().setExpanding(False)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        log_font = self.log_view.font()
        log_font.setPointSize(11)
        available = set(QFontDatabase.families())
        for family in ("Menlo", "Consolas", "DejaVu Sans Mono", "Liberation Mono", "Courier New"):
            if family in available:
                log_font.setFamily(family)
                break
        self.log_view.setFont(log_font)
        tabs.addTab(self.log_view, "Activity Log")
        self.steps_view = QListWidget()
        tabs.addTab(self.steps_view, "Last Actions")
        self.config_view = QPlainTextEdit()
        self.config_view.setReadOnly(True)
        tabs.addTab(self.config_view, "Config")
        log_lay.addWidget(tabs)
        content.addWidget(self.log_panel)

        root.addLayout(content, 1)
        self.log_panel.setVisible(bool(self.cfg.get_path("ui.show_log", True)))

        status = QStatusBar()
        status.setSizeGripEnabled(False)
        self.setStatusBar(status)
        hotkeys = self.cfg.get_path("hotkeys", {}) or {}
        status.showMessage(
            f"Run: {hotkeys.get('run')}    \u2022    Stop: {hotkeys.get('stop')}    \u2022    "
            f"Select area: {hotkeys.get('select_region')}    \u2022    "
            "Failsafe: slam the mouse into the top-left corner"
        )
        self.setCentralWidget(central)
        self._init_ai_fields()
        self._dump_config_view()
        self._apply_theme()

    # ------------------------------------------------------------ ui sub-builders
    def _build_nav_bar(self) -> QWidget:
        """Top navigation bar made of real pill buttons (not toolbar actions)."""
        bar = QFrame()
        bar.setObjectName("navBar")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(12, 9, 12, 9)
        lay.setSpacing(8)

        def nav_button(text: str, slot, *, checkable: bool = False,
                       checked: bool = False, tooltip: str = "",
                       signal: str = "toggled") -> QToolButton:
            btn = QToolButton()
            btn.setObjectName("navButton")
            btn.setText(text)
            btn.setCheckable(checkable)
            btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            if tooltip:
                btn.setToolTip(tooltip)
            if checkable:
                btn.setChecked(bool(checked))
                # "clicked" fires only on a real user click, so slots that call
                # setChecked() themselves cannot re-enter through the signal.
                getattr(btn, signal).connect(slot)
            else:
                btn.clicked.connect(slot)
            lay.addWidget(btn)
            return btn

        def separator() -> None:
            line = QFrame()
            line.setObjectName("navSeparator")
            line.setFrameShape(QFrame.VLine)
            line.setFixedWidth(1)
            lay.addWidget(line)

        self.act_select_source = nav_button(
            "Select Area To Read", self.select_source_region,
            tooltip="Drag a box over the area the bot should read.",
        )
        self.act_select_target = nav_button(
            "Select Area To Write", self.select_target_region,
            tooltip="Optional: drag a box where the answer should be typed/clicked.",
        )
        separator()
        self.act_show_regions = nav_button(
            "Outlines", lambda _checked: self.toggle_marker(), checkable=True,
            signal="clicked", tooltip="Show the saved areas on screen.",
        )
        self.act_click_preview = nav_button(
            "Click Preview", self._on_debug_pointer_toggled, checkable=True,
            checked=bool(self.cfg.get_path("actions.debug_pointer", False)),
            tooltip="Flash the exact click point before every click (debug aid).",
        )
        self.act_show_log = nav_button(
            "Log Panel", self.toggle_log_panel, checkable=True,
            checked=bool(self.cfg.get_path("ui.show_log", True)),
            tooltip="Show or hide the activity/log column.",
        )
        lay.addStretch(1)
        separator()
        nav_button("Open Config", lambda: open_path(APP_DIR),
                   tooltip="Open the folder that holds config.json.")
        nav_button("Quit", self.close)
        return bar

    def _build_options_body(self) -> QWidget:
        """All behaviour switches, tucked into one collapsible section."""
        body = QWidget()
        grid = QGridLayout(body)
        grid.setContentsMargins(10, 0, 10, 8)
        grid.setHorizontalSpacing(22)
        grid.setVerticalSpacing(8)

        self.clear_check = QCheckBox("Clear Field First")
        self.clear_check.setToolTip("Select-all + delete inside the write area before typing.")
        self.clear_check.setChecked(
            bool(self.cfg.get_path("actions.clear_field_shortcut", True))
        )
        self.enter_check = QCheckBox("Press Enter After")
        self.enter_check.setToolTip("Submit the answer once it has been typed.")
        self.dry_check = QCheckBox("Dry Run (No Clicks)")
        self.dry_check.setToolTip("Show the plan without touching the mouse or keyboard.")
        self.cursor_check = QCheckBox("Type At Cursor")
        self.cursor_check.setToolTip("Skip the click and type wherever the pointer is.")
        self.confirm_check = QCheckBox("Confirm Before Acting")
        self.confirm_check.setToolTip(
            "ON: show the Confirm dialog listing the clicks and wait for your approval.\n"
            "OFF: the bot clicks the answer immediately, with no dialog."
        )
        self.confirm_check.setChecked(
            bool(self.cfg.get_path("actions.confirm_before_action", True))
        )
        self.confirm_check.toggled.connect(self._on_confirm_toggled)
        self.hide_check = QCheckBox("Hide While Acting")
        self.hide_check.setToolTip(
            "Hide this window while the bot captures the screen and while it clicks "
            "or types, so the screenshot and the click both reach the application "
            "underneath instead of the bot window."
        )
        self.hide_check.setChecked(bool(self.cfg.get_path("actions.auto_hide_window", True)))
        self.hide_check.toggled.connect(self._on_hide_window_toggled)

        boxes = (self.clear_check, self.enter_check, self.dry_check,
                 self.cursor_check, self.confirm_check, self.hide_check)
        for index, box in enumerate(boxes):
            grid.addWidget(box, index // 3, index % 3)

        grid.addWidget(QLabel("Extra Context"), 2, 0)
        self.context_edit = QLineEdit()
        self.context_edit.setPlaceholderText(
            "Optional hint for the AI, e.g. 'this is a login form' or 'answer in one word'"
        )
        grid.addWidget(self.context_edit, 2, 1, 1, 2)
        grid.setColumnStretch(2, 1)
        return body

    def _build_ai_body(self) -> QWidget:
        """Provider / model / key / base URL, tucked into a collapsible section."""
        body = QWidget()
        grid = QGridLayout(body)
        grid.setContentsMargins(10, 0, 10, 8)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        grid.addWidget(QLabel("Provider"), 0, 0)
        self.provider_combo = ChevronComboBox()
        for key, info in PROVIDER_DEFAULTS.items():
            self.provider_combo.addItem(info["label"], key)
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        grid.addWidget(self.provider_combo, 0, 1)

        grid.addWidget(QLabel("Model"), 0, 2)
        self.model_combo = ChevronComboBox()
        self.model_combo.setEditable(True)  # pick a suggestion or type any model
        self.model_combo.setInsertPolicy(QComboBox.NoInsert)
        self.model_combo.setMinimumWidth(210)
        grid.addWidget(self.model_combo, 0, 3, 1, 3)

        grid.addWidget(QLabel("API Key"), 1, 0)
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setPlaceholderText("Paste the provider API key")
        grid.addWidget(self.api_key_edit, 1, 1, 1, 3)

        self.show_key_check = QCheckBox("Show")
        self.show_key_check.setToolTip("Reveal the key while typing.")
        self.show_key_check.toggled.connect(self._toggle_key_visibility)
        grid.addWidget(self.show_key_check, 1, 4)

        self.save_ai_btn = QPushButton("Save")
        self.save_ai_btn.setToolTip("Save provider, model, key and base URL to config.json.")
        self.save_ai_btn.clicked.connect(self._save_ai_settings)
        grid.addWidget(self.save_ai_btn, 1, 5)

        self.reload_cfg_btn = QPushButton("Reload Config")
        self.reload_cfg_btn.setFixedWidth(140)
        self.reload_cfg_btn.setToolTip("Re-read config.json from disk.")
        self.reload_cfg_btn.clicked.connect(self.reload_config)
        grid.addWidget(self.reload_cfg_btn, 0, 6, Qt.AlignRight)

        grid.addWidget(QLabel("Base URL"), 2, 0)
        self.base_url_edit = QLineEdit()
        self.base_url_edit.setPlaceholderText("https://api.openai.com/v1")
        grid.addWidget(self.base_url_edit, 2, 1, 1, 5)
        url_hint = QLabel("auto-filled by provider; edit for a custom gateway")
        url_hint.setObjectName("appHint")
        grid.addWidget(url_hint, 2, 6)

        grid.setColumnStretch(3, 1)
        return body

    def _make_section(self, title: str, body: QWidget, cfg_key: str,
                      expanded: bool = False) -> QWidget:
        """A collapsible header + body whose open/closed state is remembered."""
        wrapper = QWidget()
        lay = QVBoxLayout(wrapper)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        header = QPushButton()
        header.setObjectName("sectionHeader")
        header.setCheckable(True)
        header.setCursor(Qt.PointingHandCursor)
        header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        is_open = bool(self.cfg.get_path(cfg_key, expanded))
        header.setChecked(is_open)
        header.setText(f"{_ARROW_OPEN if is_open else _ARROW_CLOSED}   {title}")
        body.setVisible(is_open)

        def _on_toggle(open_now: bool) -> None:
            body.setVisible(open_now)
            header.setText(f"{_ARROW_OPEN if open_now else _ARROW_CLOSED}   {title}")
            self.cfg.set_path(cfg_key, bool(open_now))
            try:
                self.cfg.save()
            except Exception as exc:  # pragma: no cover - filesystem failures
                log.warning("Could not persist %s: %s", cfg_key, exc)

        header.toggled.connect(_on_toggle)
        lay.addWidget(header)
        lay.addWidget(body)
        return wrapper

    def toggle_log_panel(self, show: bool) -> None:
        """Show or hide the right-hand log area and remember the choice."""
        panel = getattr(self, "log_panel", None)
        if panel is None:
            return
        panel.setVisible(bool(show))
        action = getattr(self, "act_show_log", None)
        if action is not None and action.isChecked() != bool(show):
            action.blockSignals(True)
            action.setChecked(bool(show))
            action.blockSignals(False)
        self.cfg.set_path("ui.show_log", bool(show))
        try:
            self.cfg.save()
        except Exception as exc:  # pragma: no cover - filesystem failures
            log.warning("Could not persist ui.show_log: %s", exc)
        self.on_log(f"[ui] Log panel {'shown' if show else 'hidden'}.")

    def _apply_theme(self) -> None:
        """Apply the modern dark stylesheet (see :attr:`THEME`)."""
        self.setStyleSheet(self.THEME)

    # ------------------------------------------------------------------ signals
    def _connect_worker(self) -> None:
        w = self.worker
        w.status.connect(self.on_status)
        w.log_line.connect(self.on_log)
        w.captured.connect(self.on_captured)
        w.ocr_done.connect(self.on_ocr)
        w.answer_ready.connect(self.on_answer)
        w.action_preview.connect(self.on_action_preview)
        w.action_done.connect(self.on_action_done)
        w.failed.connect(self.on_failed)
        w.finished.connect(self.on_finished)
        w.action_pointer.connect(self._flash_pointer)
        # Safety net: whatever happens, the panel must come back.
        w.action_done.connect(lambda _result: self._restore_input_window())
        w.failed.connect(lambda _msg: self._restore_input_window())
        w.finished.connect(self._restore_input_window)

    def _register_hotkeys(self) -> None:
        from .hotkeys import HotkeyManager

        self.hotkeys = HotkeyManager(str(self.cfg.get_path("hotkey_backend", "pynput")))
        hk = self.cfg.get_path("hotkeys", {}) or {}
        mapping = {
            "select_region": self.select_source_region,
            "run": self.run_pipeline,
            "stop": self.stop_pipeline,
            "repeat_last": self.run_pipeline,
            "paste_answer": self.paste_last_answer,
            "toggle_overlay": self.toggle_marker,
        }
        started = 0
        for key, handler in mapping.items():
            combo = hk.get(key)
            if combo and self.hotkeys.register(combo, handler):
                started += 1
        if started and self.hotkeys.start():
            self.on_log(f"Global hotkeys active ({started} bindings).")
        else:
            self.hotkeys.stop()
            self._bind_local_shortcuts(hk)
            self.on_log("Global hotkeys unavailable - using app-window shortcuts instead.")

    def _bind_local_shortcuts(self, hk: dict) -> None:
        from .hotkeys import LocalHotkeyFallback, _qt_sequence

        fallback = LocalHotkeyFallback(self)
        mapping = {
            "select_region": self.select_source_region,
            "run": self.run_pipeline,
            "stop": self.stop_pipeline,
            "paste_answer": self.paste_last_answer,
            "toggle_overlay": self.toggle_marker,
        }
        for key, handler in mapping.items():
            combo = hk.get(key)
            if not combo:
                continue
            shortcut = QShortcut(_qt_sequence(combo), self)
            shortcut.setContext(Qt.ApplicationShortcut)
            shortcut.activated.connect(handler)

    # ------------------------------------------------------------- AI settings
    def _init_ai_fields(self) -> None:
        """Fill the AI provider/model/key widgets from the loaded config."""
        self._sync_ai_fields_from_cfg()

    def _sync_ai_fields_from_cfg(self) -> None:
        provider = str(self.cfg.get_path("ai.provider", "openai")).lower()
        index = self.provider_combo.findData(provider)
        if index < 0:
            index = 0
        self.provider_combo.blockSignals(True)
        self.provider_combo.setCurrentIndex(index)
        self.provider_combo.blockSignals(False)
        self._populate_models(provider, str(self.cfg.get_path("ai.model", "")))
        self.api_key_edit.setText(str(self.cfg.get_path("ai.api_key", "") or ""))
        self.base_url_edit.setText(str(self.cfg.get_path("ai.base_url", "")))

    def _populate_models(self, provider: str, current: str) -> None:
        """Fill the editable model combo with suggestions for ``provider``."""
        models = list(PROVIDER_DEFAULTS.get(provider, {}).get("models", []))
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(models)
        idx = self.model_combo.findText(current) if current else -1
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        else:
            self.model_combo.setCurrentIndex(-1)
            self.model_combo.setEditText(current or "")
        self.model_combo.blockSignals(False)

    def _on_provider_changed(self, _index: int) -> None:
        provider = self.provider_combo.currentData() or "openai"
        info = PROVIDER_DEFAULTS.get(provider, {})
        # Reset base_url to the provider default unless the user customised it.
        known = {cfg["base_url"] for cfg in PROVIDER_DEFAULTS.values()}
        current_base = self.base_url_edit.text().strip()
        if not current_base or current_base in known:
            self.base_url_edit.setText(info.get("base_url", ""))
        # Pick a model that is valid for the newly selected provider.
        current_model = self.model_combo.currentText().strip()
        models = info.get("models", [])
        if current_model not in models:
            current_model = models[0] if models else current_model
        self._populate_models(provider, current_model)
        self.statusBar().showMessage(
            f"Provider set to {info.get('label', provider)} - review and click "
            "'Save AI settings'.",
            6000,
        )

    def _toggle_key_visibility(self, shown: bool) -> None:
        self.api_key_edit.setEchoMode(QLineEdit.Normal if shown else QLineEdit.Password)

    def _save_ai_settings(self) -> None:
        """Persist the provider/model/key/base-URL shown in the control panel."""
        provider = self.provider_combo.currentData() or "openai"
        info = PROVIDER_DEFAULTS.get(provider, {})
        model = self.model_combo.currentText().strip()
        api_key = self.api_key_edit.text().strip()
        base_url = self.base_url_edit.text().strip() or info.get("base_url", "")
        if not model:
            QMessageBox.warning(self, "AI settings", "Please choose or type a model name.")
            return
        self.cfg.set_path("ai.provider", provider)
        self.cfg.set_path("ai.model", model)
        self.cfg.set_path("ai.api_key", api_key)
        self.cfg.set_path("ai.base_url", base_url)
        self.cfg.set_path("ai.api_key_env", info.get("api_key_env", "OPENAI_API_KEY"))
        try:
            self.cfg.save()
        except OSError as exc:
            QMessageBox.critical(self, "AI settings", f"Could not save config:\n{exc}")
            return
        self._dump_config_view()
        self.on_log(
            f"[ai] Saved provider={provider} model={model} "
            f"key={'set' if api_key else 'NOT set'} base_url={base_url}"
        )
        self.statusBar().showMessage(
            f"AI settings saved: {info.get('label', provider)} / {model}", 6000
        )
        self._health_check()

    def reload_config(self) -> None:
        """Reload config.json from disk and refresh the widgets that mirror it."""
        import json

        try:
            raw = json.loads(self.cfg.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raw = {}
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.critical(self, "Reload config", f"Could not read {self.cfg.path}:\n{exc}")
            return
        base = DEFAULTS
        if BUNDLED_CONFIG.exists():
            base = deep_merge(DEFAULTS, json.loads(BUNDLED_CONFIG.read_text(encoding="utf-8")))
        merged = deep_merge(base, raw)
        # Update in place so the PipelineWorker keeps using the same Config.
        self.cfg.data.clear()
        self.cfg.data.update(merged)
        self._sync_ai_fields_from_cfg()
        self._sync_action_fields_from_cfg()
        self._refresh_regions()
        self._dump_config_view()
        self.on_log(f"[config] Reloaded {self.cfg.path}.")
        self._health_check()

    # ------------------------------------------------------------------- regions
    def _refresh_regions(self) -> None:
        self.region_list.clear()
        for region in self.cfg.regions:
            item = QListWidgetItem(f"{region.name}  {region.width}x{region.height} @({region.left},{region.top})")
            item.setData(Qt.UserRole, region.to_dict())
            self.region_list.addItem(item)
        if self.source_region is None and self.cfg.regions:
            self.source_region = self.cfg.regions[0]
        self._set_active_item()
        self.marker.set_regions(self.cfg.regions, self.source_region.name if self.source_region else "")
        self._update_status()

    def _set_active_item(self) -> None:
        for index in range(self.region_list.count()):
            item = self.region_list.item(index)
            data = Region.from_dict(item.data(Qt.UserRole))
            if self.source_region and data.name == self.source_region.name:
                self.region_list.setCurrentItem(item)
                return

    def _update_status(self) -> None:
        """Show the current source/target areas in the status bar."""
        if self.source_region is None:
            self.on_status("No area selected - use 'Select Area To Read' first.")
            return
        msg = (f"Read: {self.source_region.name} "
               f"{self.source_region.width}x{self.source_region.height} "
               f"@({self.source_region.left},{self.source_region.top})")
        if self.target_region is not None:
            msg += (f"   |   Write: {self.target_region.name} "
                    f"{self.target_region.width}x{self.target_region.height} "
                    f"@({self.target_region.left},{self.target_region.top})")
        self.on_status(msg)

    def _on_debug_pointer_toggled(self, enabled: bool) -> None:
        """Turn the on-screen click flash on/off and persist the choice."""
        self.cfg.set_path("actions.debug_pointer", bool(enabled))
        try:
            self.cfg.save()
        except OSError:
            log.warning("Could not persist debug_pointer setting", exc_info=True)
        self.on_log(f"[debug] Click preview {'ON' if enabled else 'OFF'}.")
        if not enabled:
            self.click_flash.hide()

    def _flash_pointer(self, x: int, y: int, kind: str) -> None:
        if not self.cfg.get_path("actions.debug_pointer", False):
            return
        self.click_flash.flash(x, y, kind)

    def selected_region(self) -> Optional[Region]:
        item = self.region_list.currentItem()
        if item is None:
            return None
        return Region.from_dict(item.data(Qt.UserRole))

    def apply_selected_region(self) -> None:
        region = self.selected_region()
        if region:
            self.source_region = region
            self._refresh_regions()
            self.on_status(f"Source area set to \"{region.name}\"")

    def use_selected_as_target(self) -> None:
        region = self.selected_region()
        if region:
            self.target_region = region
            self.on_status(f"Write target set to \"{region.name}\"")

    def add_current_region(self) -> None:
        if self.source_region is None:
            QMessageBox.information(self, "ScreenAI", "Select an area first.")
            return
        name, ok = QInputDialog.getText(self, "Name this area", "Area name:",
                                        text=f"area-{len(self.cfg.regions) + 1}")
        if not ok or not name.strip():
            return
        region = Region(name.strip(), self.source_region.left, self.source_region.top,
                        self.source_region.width, self.source_region.height)
        self.cfg.add_region(region)
        self.cfg.save()
        self.source_region = region
        self._refresh_regions()

    def rename_region(self) -> None:
        region = self.selected_region()
        if region is None:
            return
        name, ok = QInputDialog.getText(self, "Rename area", "New name:", text=region.name)
        if not ok or not name.strip():
            return
        regions = self.cfg.regions
        for index, item in enumerate(regions):
            if item.name == region.name:
                regions[index] = Region(name.strip(), item.left, item.top, item.width, item.height)
        self.cfg.regions = regions
        if self.source_region and self.source_region.name == region.name:
            self.source_region = regions[index]
        self.cfg.save()
        self._refresh_regions()

    def delete_region(self) -> None:
        region = self.selected_region()
        if region is None:
            return
        regions = [r for r in self.cfg.regions if r.name != region.name]
        self.cfg.regions = regions
        self.cfg.save()
        if self.source_region and self.source_region.name == region.name:
            self.source_region = regions[0] if regions else None
        self._refresh_regions()

    # ----------------------------------------------------------------- selection
    def select_source_region(self) -> None:
        self._open_selector("Drag a box around the text to read", self._on_source_selected)

    def select_target_region(self) -> None:
        self._open_selector("Drag a box where the answer should be typed", self._on_target_selected)

    def _open_selector(self, title: str, callback) -> None:
        if self.selector is not None:
            self.selector.close()
        self.hide_marker_for_capture()
        self.selector = RegionSelector(title=title, preset=self.source_region)
        self.selector.set_presets(self.cfg.regions)
        self.selector.selected.connect(callback)
        self.selector.cancelled.connect(self._selection_cancelled)
        self.selector.destroyed.connect(lambda *_: setattr(self, "selector", None))

    def _on_source_selected(self, region: Region) -> None:
        self.source_region = region
        self.cfg.add_region(region)
        self.cfg.save()
        self._refresh_regions()
        self.on_status(f"Read area: {region}")
        self.show_marker_if_needed()

    def _on_target_selected(self, region: Region) -> None:
        self.target_region = region
        self.on_status(f"Write target: {region}")
        self.show_marker_if_needed()

    def _selection_cancelled(self) -> None:
        self.show_marker_if_needed()

    def hide_marker_for_capture(self) -> None:
        if self.marker.isVisible():
            self.marker.hide()
            self._marker_was_visible = True

    def show_marker_if_needed(self) -> None:
        if getattr(self, "_marker_was_visible", False):
            self.marker.setGeometry(self.marker.geometry())
            self.marker.show()
            self._marker_was_visible = False

    def toggle_marker(self) -> None:
        if self.marker.isVisible():
            self.marker.hide()
            checked = False
        else:
            self.marker.set_regions(self.cfg.regions,
                                    self.source_region.name if self.source_region else "")
            self.marker.show()
            checked = True
        if self.act_show_regions.isChecked() != checked:
            self.act_show_regions.blockSignals(True)
            self.act_show_regions.setChecked(checked)
            self.act_show_regions.blockSignals(False)

    # ---------------------------------------------------------------------- run
    def _restore_ui_selection(self) -> None:
        """Apply the persisted Action/Task dropdown choices at startup."""
        saved_mode = str(self.cfg.get_path("ui.action_mode", "type_in_region") or "")
        idx = self.mode_combo.findData(saved_mode)
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)
        else:
            log.warning("Unknown ui.action_mode %r; keeping the default", saved_mode)
        saved_task = str(self.cfg.get_path("ui.task_key", "answer") or "")
        tidx = self.task_combo.findData(saved_task)
        if tidx >= 0:
            self.task_combo.setCurrentIndex(tidx)
        log.info(
            "UI restored: action=%s task=%s",
            self.mode_combo.currentData(), self.task_combo.currentData(),
        )

    def _remember_ui_selection(self) -> None:
        """Persist the current Action/Task dropdown choices to config.json."""
        mode = self.mode_combo.currentData()
        task = self.task_combo.currentData()
        changed = False
        if mode and self.cfg.get_path("ui.action_mode") != str(mode):
            self.cfg.set_path("ui.action_mode", str(mode))
            changed = True
        if task and self.cfg.get_path("ui.task_key") != str(task):
            self.cfg.set_path("ui.task_key", str(task))
            changed = True
        if not changed:
            return
        try:
            self.cfg.save()
        except Exception as exc:  # pragma: no cover - filesystem failures
            log.warning("Could not persist UI selection: %s", exc)

    def _on_task_changed(self, _index: int) -> None:
        self._remember_ui_selection()

    def _on_mode_changed(self, _index: int) -> None:
        """Auto-pair the MCQ click mode with the choose_option task."""
        if self.mode_combo.currentData() == "click_matching_line":
            idx = self.task_combo.findData("choose_option")
            if idx >= 0 and self.task_combo.currentData() != "choose_option":
                self.task_combo.setCurrentIndex(idx)
                self.on_log("[mcq] Task set to 'choose_option' for the MCQ click mode.")
        self._remember_ui_selection()
        self.on_log(f"[action] Selected mode: {self.mode_combo.currentData()}")

    # ------------------------------------------------- window occlusion handling
    def _panel_rect(self) -> Optional[tuple]:
        """Rect of our panel *as included in the last capture*, else None.

        ``None`` means the screenshot was taken with the panel hidden, so any
        OCR hit is genuine screen content and must not be second-guessed.
        """
        rect = getattr(self, "_capture_panel_rect", None)
        return rect

    def _live_panel_rect(self) -> Optional[tuple]:
        """The bot window's current global rect."""
        try:
            geo = self.frameGeometry()
            if geo.width() <= 0 or geo.height() <= 0:
                return None
            return (geo.left(), geo.top(), geo.width(), geo.height())
        except Exception:  # pragma: no cover - defensive
            return None

    def _point_under_window(self, x: int, y: int, margin: int = 12) -> bool:
        """True when the given global point falls on this window (or its frame)."""
        if not self.isVisible():
            return False
        rect = self.frameGeometry().adjusted(-margin, -margin, margin, margin)
        return rect.contains(QPoint(int(x), int(y)))

    def _hide_delay_ms(self) -> int:
        return max(0, int(self.cfg.get_path("actions.auto_hide_delay_ms", 220)))

    def _wait_for_window_server(self) -> None:
        """Pause (on the calling thread) so the OS can remove the window."""
        delay = self._hide_delay_ms()
        if delay:
            time.sleep(delay / 1000.0)

    def _begin_hide(self, reason: str) -> None:
        """Hide the window and arm a watchdog so it always comes back."""
        if self._hidden_for_action:
            return
        self._hidden_for_action = True
        self._hide_gen += 1
        gen = self._hide_gen
        self.hide()
        log.info("Hiding bot window (%s)", reason)
        self.on_log(f"[window] Hidden ({reason}) - the click reaches the app underneath.")
        # The watchdog must never fire while a legitimate restore delay runs.
        timeout = max(
            int(self.cfg.get_path("actions.auto_hide_max_ms", 8000) or 8000),
            int(self.cfg.get_path("actions.restore_delay_ms", 1000) or 0) + 3000,
        )
        QTimer.singleShot(timeout, lambda: self._force_restore(gen))

    def _force_restore(self, gen: int) -> None:
        """Watchdog: never leave the window hidden if an action dies midway."""
        if gen != self._hide_gen or not self._hidden_for_action:
            return
        log.warning("Window still hidden after acting - forcing it back (watchdog)")
        self._restore_input_window()

    def _signal_hide_done(self) -> None:
        """Release a worker thread waiting for the hide decision (if any)."""
        event = getattr(self, "_hide_event", None)
        if event is not None:
            self._hide_event = None
            event.set()

    @Slot(int, int, bool)
    def _prepare_input_window(self, x: int, y: int, will_click: bool) -> None:
        """Hide the bot window when it covers where input is about to be sent."""
        try:
            if not bool(self.cfg.get_path("actions.auto_hide_window", True)):
                return
            covers = self._point_under_window(x, y)
            # Typing without a click goes to whichever app is frontmost, so move
            # ourselves out of the way when we are the active window.
            steals_focus = (not will_click) and self.isActiveWindow()
            if not (covers or steals_focus):
                return
            reason = ("click target is under this window" if covers
                      else "this window has keyboard focus")
            self._begin_hide(reason)
        finally:
            self._signal_hide_done()

    @Slot(int, int, int, int)
    def _prepare_capture_window(self, left: int, top: int, width: int, height: int) -> None:
        """Hide the bot window so our own UI cannot appear in the screenshot."""
        # Assume a clean capture; only record the panel when it really was on
        # screen while the pixels were grabbed.
        self._capture_panel_rect = None
        try:
            if not bool(self.cfg.get_path("actions.auto_hide_window", True)):
                self._capture_panel_rect = self._live_panel_rect()
                return
            if not self.isVisible():
                return
            mode = str(self.cfg.get_path("actions.auto_hide_capture_mode", "always") or "always")
            if mode != "always":
                region = QRect(int(left), int(top), int(width), int(height))
                if not self.frameGeometry().intersects(region):
                    return
            self._begin_hide("capturing the screen")
            if self.isVisible():  # hide did not take effect: capture is polluted
                self._capture_panel_rect = self._live_panel_rect()
        finally:
            self._signal_hide_done()

    @Slot()
    def _restore_input_window(self) -> None:
        if not getattr(self, "_hidden_for_action", False):
            return
        self._hidden_for_action = False
        if self.isMinimized():
            self.showNormal()
        self.show()
        self.setWindowState((self.windowState() & ~Qt.WindowMinimized) | Qt.WindowActive)
        self.raise_()
        self.activateWindow()
        log.info("Bot window restored")
        self.on_log("[window] Restored.")

    def _on_gui_thread(self) -> bool:
        """True when running on the GUI thread.

        Note: ``self.thread`` is the WorkerThread in this class, so the usual
        ``QThread.currentThread() == self.thread()`` check cannot be used here.
        """
        app = QApplication.instance()
        return app is not None and QThread.currentThread() == app.thread()

    def _hide_for_input_sync(self, x: int, y: int, will_click: bool = True) -> bool:
        """Worker-thread safe: hide the window before input events fire.

        The GUI thread is never blocked: the hide is posted to it, *this* thread
        waits on an Event, and the settle delay happens here on the worker.
        Returns True when the panel is hidden (so callers know to wait before
        showing it again).
        """
        if self._on_gui_thread():
            self._prepare_input_window(x, y, will_click)
            self._wait_for_window_server()
            return bool(self._hidden_for_action)
        done = threading.Event()
        self._hide_event = done
        try:
            QMetaObject.invokeMethod(
                self, "_prepare_input_window", Qt.QueuedConnection,
                Q_ARG(int, int(x)), Q_ARG(int, y), Q_ARG(bool, bool(will_click)),
            )
        except Exception as exc:  # pragma: no cover - defensive
            self._hide_event = None
            log.warning("Could not ask the GUI to hide the window: %s", exc)
            return False
        if done.wait(2.0):
            self._wait_for_window_server()
        return bool(self._hidden_for_action)

    def _hide_for_capture_sync(self, region) -> bool:
        """Worker-thread safe: hide the window while the screen is captured."""
        if self._on_gui_thread():
            self._prepare_capture_window(region.left, region.top, region.width, region.height)
            self._wait_for_window_server()
            return bool(self._hidden_for_action)
        done = threading.Event()
        self._hide_event = done
        try:
            QMetaObject.invokeMethod(
                self, "_prepare_capture_window", Qt.QueuedConnection,
                Q_ARG(int, int(region.left)), Q_ARG(int, region.top),
                Q_ARG(int, int(region.width)), Q_ARG(int, int(region.height)),
            )
        except Exception as exc:  # pragma: no cover - defensive
            self._hide_event = None
            log.warning("Could not ask the GUI to hide the window: %s", exc)
            return False
        if done.wait(2.0):
            self._wait_for_window_server()
        return bool(self._hidden_for_action)

    def _restore_input_window_async(self) -> None:
        """Worker-thread safe call into ``_restore_input_window``."""
        if self._on_gui_thread():
            self._restore_input_window()
            return
        try:
            QMetaObject.invokeMethod(self, "_restore_input_window", Qt.QueuedConnection)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Could not restore the window after acting: %s", exc)

    def _on_confirm_toggled(self, enabled: bool) -> None:
        self.cfg.set_path("actions.confirm_before_action", bool(enabled))
        try:
            self.cfg.save()
        except Exception as exc:  # pragma: no cover - filesystem failures
            log.warning("Could not persist confirm_before_action: %s", exc)
        if enabled:
            self.on_log("[action] Confirm dialog ON - you approve each click first.")
        else:
            self.on_log("[action] Confirm dialog OFF - the answer will be clicked straight away.")

    def _sync_action_fields_from_cfg(self) -> None:
        """Mirror config values back into the action checkboxes (no re-saving)."""
        pairs = (
            (self.confirm_check, "actions.confirm_before_action", True),
            (self.hide_check, "actions.auto_hide_window", True),
        )
        for box, key, default in pairs:
            value = bool(self.cfg.get_path(key, default))
            box.blockSignals(True)
            box.setChecked(value)
            box.blockSignals(False)

    def _on_hide_window_toggled(self, enabled: bool) -> None:
        self.cfg.set_path("actions.auto_hide_window", bool(enabled))
        try:
            self.cfg.save()
        except Exception as exc:  # pragma: no cover - filesystem failures
            log.warning("Could not persist auto_hide_window: %s", exc)
        self.on_log(f"[window] Auto-hide while acting {'ON' if enabled else 'OFF'}.")

    def build_request(self, text_override: Optional[str] = None) -> Optional[PipelineRequest]:
        if self.source_region is None:
            QMessageBox.warning(self, "ScreenAI", "No area selected. Use 'Select Area To Read' first.")
            return None
        mode = self.mode_combo.currentData()
        if self.cursor_check.isChecked():
            mode = "type_at_cursor"
        request = PipelineRequest(
            source_region=self.source_region,
            task_key=self.task_combo.currentData() or "answer",
            target_region=self.target_region,
            mode=mode,
            extra_context=self.context_edit.text().strip(),
            clear_field=self.clear_check.isChecked(),
            press_enter=self.enter_check.isChecked(),
            auto_action=not self.dry_check.isChecked(),
            use_cursor=self.cursor_check.isChecked(),
            dry_run=self.dry_check.isChecked(),
        )
        if text_override is not None:
            request._text_override = text_override  # consumed by run_pipeline
        return request

    def run_pipeline(self) -> None:
        if self._busy:
            self.on_status("Already running - press Stop to cancel first.")
            return
        request = self.build_request()
        if request is None:
            return
        override = getattr(request, "_text_override", None)
        self.steps_view.clear()
        self.answer_view.clear()
        if override is not None:
            self.ask_about_text(override)
            return
        self._start_worker(request)

    def _start_worker(self, request: PipelineRequest) -> None:
        self._busy = True
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.thread = WorkerThread(self.worker, request, parent=self)
        self.thread.start()
        self.on_log(f"RUN start: {request.source_region} task={request.task_key} mode={request.mode}")

    def ask_about_text(self, text: str) -> None:
        """Send manually provided text to the AI (no capture/OCR)."""
        ocr = OcrResult(text=text, confidence=100.0, engine="manual")
        self.on_ocr(ocr)
        request = self.build_request()
        if request is None:
            return

        def worker_call() -> None:
            try:
                answer = self.worker.ai.ask(
                    ocr, task_key=request.task_key, region=request.source_region,
                    extra_context=request.extra_context,
                )
                self.worker._last_answer = answer
                self.worker.answer_ready.emit(answer)
                self.worker._stage_action(answer, request, ocr)
            except Exception as exc:
                self.worker.failed.emit(str(exc))
            finally:
                self.worker.finished.emit()

        class _SimpleThread(QThread):
            def run(self_inner):  # noqa: N802
                worker_call()

        self._busy = True
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.thread = _SimpleThread(parent=self)  # type: ignore[assignment]
        self.thread.start()

    def ask_with_edited_text(self) -> None:
        text = self.ocr_view.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "ScreenAI", "Nothing in the OCR box to send.")
            return
        self.ask_about_text(text)

    def apply_answer_now(self) -> None:
        text = self.answer_view.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "ScreenAI", "No answer to apply.")
            return
        if self._busy:
            self.on_status("Busy - wait for the current run to finish.")
            return
        request = self.build_request()
        if request is None:
            return
        self.on_log(f"APPLY: mode={request.mode}")
        if request.mode == "click_matching_line":
            from .ai_client import AiAnswer
            self.worker._act_click_matching_line(
                AiAnswer(text=text), request, self.worker.last_ocr
            )
            return
        steps = self.worker.planner.build(
            text, mode=request.mode, source_region=request.source_region,
            target_region=request.target_region, clear_field=request.clear_field,
            press_enter=request.press_enter,
        )
        # Queue the steps BEFORE previewing: with confirmation enabled,
        # on_action_preview() opens the dialog and executes them straight away.
        self.worker._pending_steps = steps
        self.on_action_preview([step.describe() for step in steps])
        if not bool(self.cfg.get_path("actions.confirm_before_action", True)):
            self.worker.execute_steps()

    def stop_pipeline(self) -> None:
        self.worker.stop()
        self.on_status("Stop requested.")

    def paste_last_answer(self) -> None:
        if self.worker.last_answer is None:
            self.on_status("No previous answer to paste.")
            return
        self.worker.paste_last_answer_at_cursor()

    def copy_answer(self) -> None:
        text = self.answer_view.toPlainText().strip()
        if text:
            QApplication.clipboard().setText(text)
            self.on_status("Answer copied to clipboard.")

    def show_prompt(self) -> None:
        ocr = OcrResult(text=self.ocr_view.toPlainText(), confidence=0.0, engine="manual")
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Prompt preview")
        dialog.setTextFormat(Qt.PlainText)
        dialog.setText(preview_prompt(self.cfg, ocr, self.task_combo.currentData(), self.source_region))
        dialog.exec()

    # ------------------------------------------------------------------ callbacks
    def on_status(self, message: str) -> None:
        self.statusBar().showMessage(message)

    def on_log(self, line: str) -> None:
        self.log_view.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {line}")

    def on_captured(self, image) -> None:
        self.last_image = image
        pixmap = pil_to_pixmap(image)
        scaled = pixmap.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation) \
            if self.preview.width() > 10 else pixmap
        self.preview.setPixmap(scaled)
        self.preview.setText("")

    def on_ocr(self, ocr: OcrResult) -> None:
        self.ocr_view.setPlainText(ocr.text)
        suffix = f" ({ocr.engine}, conf {ocr.confidence:.0f}%)"
        if ocr.note:
            suffix += f"  ! {ocr.note}"
        existing = self.ocr_view.placeholderText()
        self.ocr_view.setPlaceholderText("OCR" + suffix)
        self.on_log(f"OCR done{suffix}")

    def on_answer(self, answer) -> None:
        self.answer_view.setPlainText(answer.text)
        self.on_log(f"Answer ready from {answer.model} in {answer.latency_ms}ms")

    def on_action_preview(self, descriptions: List[str]) -> None:
        self.steps_view.clear()
        for line in descriptions:
            self.steps_view.addItem(line)
        if self.cfg.get_path("actions.confirm_before_action", True):
            self._confirm_pending()

    def _confirm_pending(self) -> None:
        lines = "\n".join(f"  - {self.steps_view.item(i).text()}" for i in range(self.steps_view.count()))
        choice = QMessageBox.question(
            self, "Confirm actions",
            f"The bot is about to control your mouse and keyboard:\n\n{lines}\n\nProceed?",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Abort, QMessageBox.No,
        )
        if choice == QMessageBox.Yes:
            self.worker.execute_steps()
        elif choice == QMessageBox.Abort:
            self.worker._pending_steps = []
            self.on_status("Cancelled - nothing was typed.")
        else:
            self.worker._pending_steps = []
            if self.worker.last_answer:
                self.worker.input.copy_to_clipboard(self.worker.last_answer.text)
            self.on_status("Declined - answer copied to clipboard instead.")

    def on_action_done(self, result) -> None:
        if result.ok:
            self.on_log(f"Actions OK ({result.executed} steps)")
        else:
            self.on_log(f"Actions incomplete: {result.error}")

    def on_failed(self, message: str) -> None:
        self.on_log(f"ERROR: {message}")
        self.statusBar().showMessage(f"Failed: {message}")

    def on_finished(self) -> None:
        self._busy = False
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if self.thread is not None:
            self.thread.wait(1500)
            self.thread = None
        self.statusBar().showMessage("Ready.")

    # ------------------------------------------------------------------- health
    def _health_check(self) -> None:
        checks = []
        checks.append(("Tesseract OCR", "found" if self.worker.ocr.available else "MISSING (vision fallback used)"))
        checks.append(("API key", "configured" if self.cfg.api_key else "NOT SET"))
        checks.append(("Regions", f"{len(self.cfg.regions)} saved" if self.cfg.regions else "none yet"))
        checks.append(("Provider", f"{self.cfg.get_path('ai.provider')}/{self.cfg.get_path('ai.model')}"))
        checks.append(("Vision mode", "on" if self.cfg.get_path("ai.use_vision") else "off"))
        for name, value in checks:
            self.on_log(f"[check] {name}: {value}")
        if not self.cfg.api_key:
            env_name = str(self.cfg.get_path("ai.api_key_env", "OPENAI_API_KEY"))
            self.on_log(
                f"[hint] Set ai.api_key (or export {env_name}) in "
                f"{APP_DIR / 'config.json'}, or use the AI settings row above."
            )
        import sys

        if sys.platform == "darwin":
            self.on_log(
                "[macos] If clicks or global hotkeys are ignored, enable Accessibility "
                "(and Input Monitoring) for your terminal/IDE in "
                "System Settings > Privacy & Security, then restart the app."
            )

    def _dump_config_view(self) -> None:
        import copy
        import json

        view = copy.deepcopy(self.cfg.data)
        ai = view.get("ai", {}) if isinstance(view, dict) else {}
        if isinstance(ai, dict) and ai.get("api_key"):
            ai["api_key"] = _mask_secret(str(ai["api_key"]))
        self.config_view.setPlainText(json.dumps(view, indent=2, ensure_ascii=False))

    def closeEvent(self, event) -> None:  # noqa: N802
        try:
            self.worker.stop()
            if self.thread is not None:
                self.thread.wait(2000)
            self.hotkeys.stop()
            self.worker.capture.close()
            self.click_flash.close()
        except Exception:
            log.exception("shutdown issue")
        super().closeEvent(event)


def open_path(path) -> None:
    import os
    import subprocess
    import sys

    path = str(path)
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        log.warning("Could not open %s", path)
