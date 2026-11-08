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
import time
from typing import List, Optional

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAction, QAbstractItemView, QApplication, QCheckBox, QComboBox, QHBoxLayout, QInputDialog,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit,
    QPushButton, QShortcut, QSplitter, QStatusBar, QTabWidget, QTextEdit, QToolBar, QVBoxLayout, QWidget,
)

from .actions import ActionPlanner
from .config import APP_DIR, Config, Region, load_config
from .ocr import OcrResult
from .overlay import RegionMarker, RegionSelector
from .pipeline import PipelineRequest, PipelineWorker, preview_prompt

log = logging.getLogger("screenai.ui")


def pil_to_pixmap(image) -> QPixmap:
    """PIL image -> QPixmap without depending on Qt image plugins."""
    from PIL import Image as PILImage

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    pixmap = QPixmap()
    pixmap.loadFromData(buffer.getvalue())
    del PILImage
    return pixmap


class WorkerThread(QThread):
    """Runs PipelineWorker.run() off the GUI thread."""

    def __init__(self, worker: PipelineWorker, request: PipelineRequest, parent=None) -> None:
        super().__init__(parent)
        self.worker = worker
        self.request = request

    def run(self) -> None:  # noqa: D102 - QThread override
        self.worker.run(self.request)


class MainWindow(QMainWindow):
    MODE_LABELS = {
        "type_in_region": "Click inside area, clear it, type answer",
        "click_and_type": "Click target once, then type",
        "double_click_and_type": "Double-click target, then type",
        "type_at_cursor": "Type at current cursor (no click)",
        "click_only": "Just click the area (no typing)",
        "press_enter_after": "Type answer, then press Enter",
        "copy_only": "Copy answer to clipboard only",
    }

    def __init__(self, cfg: Optional[Config] = None) -> None:
        super().__init__()
        self.cfg = cfg or load_config()
        self.setWindowTitle("ScreenAI Bot - read screen, ask AI, act")
        self.resize(1080, 720)
        self.worker = PipelineWorker(self.cfg)
        self.thread: Optional[WorkerThread] = None
        self.selector: Optional[RegionSelector] = None
        self.marker = RegionMarker()
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

        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        self.act_select_source = toolbar.addAction("1. Select area to READ", self.select_source_region)
        self.act_select_target = toolbar.addAction("2. Select area to WRITE (optional)", self.select_target_region)
        self.act_show_regions = toolbar.addAction("Show outlines", self.toggle_marker)
        self.act_show_regions.setCheckable(True)
        toolbar.addSeparator()
        self.act_run = toolbar.addAction("Run", self.run_pipeline)
        self.act_stop = toolbar.addAction("Stop", self.stop_pipeline)
        self.act_repeat = toolbar.addAction("Repeat last", self.run_pipeline)
        toolbar.addSeparator()
        toolbar.addAction("Open config", lambda: open_path(APP_DIR))
        toolbar.addAction("Quit", self.close)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Task"))
        self.task_combo = QComboBox()
        tasks = self.cfg.get_path("prompt.tasks", {}) or {}
        for key in tasks.keys():
            self.task_combo.addItem(f"{key}", key)
        controls.addWidget(self.task_combo)
        controls.addWidget(QLabel("Action"))
        self.mode_combo = QComboBox()
        for key, label in self.MODE_LABELS.items():
            self.mode_combo.addItem(label, key)
        controls.addWidget(self.mode_combo)
        self.clear_check = QCheckBox("Clear field first")
        self.clear_check.setChecked(bool(self.cfg.get_path("actions.clear_field_shortcut", True)))
        controls.addWidget(self.clear_check)
        self.enter_check = QCheckBox("Press Enter after")
        controls.addWidget(self.enter_check)
        self.dry_check = QCheckBox("Dry run (no clicks)")
        controls.addWidget(self.dry_check)
        self.cursor_check = QCheckBox("Type at cursor")
        controls.addWidget(self.cursor_check)
        controls.addStretch(1)
        root.addLayout(controls)

        context_row = QHBoxLayout()
        context_row.addWidget(QLabel("Extra context"))
        self.context_edit = QLineEdit()
        self.context_edit.setPlaceholderText("Optional: e.g. 'this is a login form', 'answer in one word'")
        context_row.addWidget(self.context_edit, 1)
        root.addLayout(context_row)

        splitter = QSplitter(Qt.Horizontal)

        left = QVBoxLayout()
        left.addWidget(QLabel("Saved regions (top = default source)"))
        self.region_list = QListWidget()
        self.region_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.region_list.itemDoubleClicked.connect(lambda *_: self.apply_selected_region())
        left.addWidget(self.region_list, 1)
        row = QHBoxLayout()
        add_btn = QPushButton("Add current selection")
        add_btn.clicked.connect(self.add_current_region)
        rename_btn = QPushButton("Rename")
        rename_btn.clicked.connect(self.rename_region)
        del_btn = QPushButton("Delete")
        del_btn.clicked.connect(self.delete_region)
        use_btn = QPushButton("Use as source")
        use_btn.clicked.connect(self.apply_selected_region)
        pick_btn = QPushButton("Pick target here")
        pick_btn.clicked.connect(self.use_selected_as_target)
        for button in (add_btn, rename_btn, del_btn):
            row.addWidget(button)
        left.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(use_btn)
        row2.addWidget(pick_btn)
        left.addLayout(row2)
        left.addWidget(QLabel("Captured preview"))
        self.preview = QLabel("no capture yet")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(150)
        self.preview.setStyleSheet("background:#101418;color:#8a94a3;border:1px solid #2a3140;border-radius:6px;")
        left.addWidget(self.preview, 1)
        left_panel = QWidget()
        left_panel.setLayout(left)
        splitter.addWidget(left_panel)

        middle = QVBoxLayout()
        middle.addWidget(QLabel("Text read from screen (OCR) - editable"))
        self.ocr_view = QPlainTextEdit()
        self.ocr_view.setPlaceholderText("Select an area and press Run...")
        middle.addWidget(self.ocr_view, 2)
        middle.addWidget(QLabel("AI answer - edit before applying"))
        self.answer_view = QTextEdit()
        self.answer_view.setPlaceholderText("The model's reply appears here.")
        middle.addWidget(self.answer_view, 2)
        btns = QHBoxLayout()
        self.ask_btn = QPushButton("Ask AI with edited text")
        self.ask_btn.clicked.connect(self.ask_with_edited_text)
        self.apply_btn = QPushButton("Apply answer now")
        self.apply_btn.clicked.connect(self.apply_answer_now)
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.clicked.connect(self.copy_answer)
        self.prompt_btn = QPushButton("Preview prompt")
        self.prompt_btn.clicked.connect(self.show_prompt)
        for button in (self.ask_btn, self.apply_btn, self.copy_btn, self.prompt_btn):
            btns.addWidget(button)
        middle.addLayout(btns)
        mid_panel = QWidget()
        mid_panel.setLayout(middle)
        splitter.addWidget(mid_panel)

        right = QVBoxLayout()
        tabs = QTabWidget()
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        tabs.addTab(self.log_view, "Activity log")
        self.steps_view = QListWidget()
        tabs.addTab(self.steps_view, "Last actions")
        self.config_view = QPlainTextEdit()
        self.config_view.setReadOnly(True)
        tabs.addTab(self.config_view, "Config")
        right.addWidget(tabs)
        self.run_btn = QPushButton("RUN (Ctrl+Alt+Enter)")
        self.run_btn.setMinimumHeight(46)
        self.run_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.run_btn.clicked.connect(self.run_pipeline)
        right.addWidget(self.run_btn)
        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setMinimumHeight(30)
        self.stop_btn.clicked.connect(self.stop_pipeline)
        self.stop_btn.setEnabled(False)
        right.addWidget(self.stop_btn)
        right_panel = QWidget()
        right_panel.setLayout(right)
        splitter.addWidget(right_panel)

        splitter.setSizes([280, 460, 320])
        root.addWidget(splitter, 1)

        status = QStatusBar()
        self.setStatusBar(status)
        hotkeys = self.cfg.get_path("hotkeys", {}) or {}
        status.showMessage(
            f"Select area: {hotkeys.get('select_region')}   |   Run: {hotkeys.get('run')}   |   "
            f"Stop: {hotkeys.get('stop')}   |   Paste last answer: {hotkeys.get('paste_answer')}   |   "
            "Failsafe: slam mouse into the top-left corner"
        )
        self.setCentralWidget(central)
        self._dump_config_view()

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
            self.act_show_regions.setChecked(False)
        else:
            self.marker.set_regions(self.cfg.regions,
                                     self.source_region.name if self.source_region else "")
            self.marker.show()
            self.act_show_regions.setChecked(True)

    # ---------------------------------------------------------------------- run
    def build_request(self, text_override: Optional[str] = None) -> Optional[PipelineRequest]:
        if self.source_region is None:
            QMessageBox.warning(self, "ScreenAI", "No area selected. Use 'Select area to READ' first.")
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
            self.on_status("Already running - press STOP to cancel first.")
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
        steps = self.worker.planner.build(
            text, mode=request.mode, source_region=request.source_region,
            target_region=request.target_region, clear_field=request.clear_field,
            press_enter=request.press_enter,
        )
        descriptions = [step.describe() for step in steps]
        self.on_action_preview(descriptions)
        self.worker._pending_steps = steps
        self._confirm_pending()

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
            self.on_log("[hint] Set ai.api_key (or export OPENAI_API_KEY) in "
                        f"{APP_DIR / 'config.json'} then use 'Reload config'.")

    def _dump_config_view(self) -> None:
        import json

        self.config_view.setPlainText(json.dumps(self.cfg.data, indent=2, ensure_ascii=False))

    def closeEvent(self, event) -> None:  # noqa: N802
        try:
            self.worker.stop()
            if self.thread is not None:
                self.thread.wait(2000)
            self.hotkeys.stop()
            self.worker.capture.close()
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
