"""The orchestrator: capture -> OCR -> AI -> click/type, off the UI thread.

Signals emitted (all safe to connect to Qt slots; the worker lives in a QThread):

* ``status(str)``                - one-line progress text
* ``log_line(str)``              - verbose activity log
* ``captured(object)``           - PIL image of the region
* ``ocr_done(object)``           - :class:`OcrResult`
* ``answer_ready(object)``       - :class:`AiAnswer`
* ``action_preview(list)``       - list of step descriptions awaiting confirmation
* ``action_done(object)``        - :class:`ActionResult`
* ``failed(str)``                - user-facing error
* ``action_pointer(int,int,str)``- click location relayed for the debug flash
* ``finished()``                 - pipeline complete (success or not)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, List, Optional

from PySide6.QtCore import QObject, Signal

from .actions import ActionPlanner, ActionResult, InputController, pointer_position
from .ai_client import AIClient, AiAnswer, build_prompt
from .capture import ScreenCapture
from .config import Config, Region
from .ocr import OcrEngine, OcrResult

log = logging.getLogger("screenai.pipeline")


class PipelineRequest:
    """Everything needed for one run; built by the UI from current widget state."""

    def __init__(
        self,
        source_region: Region,
        task_key: str = "answer",
        target_region: Optional[Region] = None,
        mode: str = "type_in_region",
        extra_context: str = "",
        clear_field: bool = True,
        press_enter: bool = False,
        auto_action: bool = True,
        use_cursor: bool = False,
        dry_run: bool = False,
        label: str = "",
    ) -> None:
        self.source_region = source_region.normalized()
        self.task_key = task_key
        self.target_region = target_region.normalized() if target_region else None
        self.mode = mode
        self.extra_context = extra_context
        self.clear_field = clear_field
        self.press_enter = press_enter
        self.auto_action = auto_action
        self.use_cursor = use_cursor
        self.dry_run = dry_run
        self.label = label or source_region.name


class PipelineWorker(QObject):
    status = Signal(str)
    log_line = Signal(str)
    captured = Signal(object)
    ocr_done = Signal(object)
    answer_ready = Signal(object)
    action_preview = Signal(list)
    action_done = Signal(object)
    failed = Signal(str)
    finished = Signal()
    action_pointer = Signal(int, int, str)  # debug: where a click will land

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.capture = ScreenCapture()
        self.ocr = OcrEngine(cfg)
        self.ai = AIClient(cfg)
        self.input = InputController(cfg, on_pointer=self._emit_pointer)
        self.planner = ActionPlanner(cfg)
        self._stop = False
        self._request: Optional[PipelineRequest] = None
        self._last_answer: Optional[AiAnswer] = None
        self._last_ocr: Optional[OcrResult] = None
        self._pending_steps: List[Any] = []
        # Optional callbacks supplied by the GUI so it can move its own window
        # out of the way before capturing or sending input (occlusion handling).
        self.pre_action_hook: Optional[Any] = None   # f(x, y, will_click) -> None
        self.post_action_hook: Optional[Any] = None  # f() -> None
        self.pre_capture_hook: Optional[Any] = None  # f(region) -> None
        self.post_capture_hook: Optional[Any] = None  # f() -> None
        # Returns the bot panel's global (left, top, width, height), if known.
        self.panel_rect_provider: Optional[Any] = None

    # ------------------------------------------------------------------ controls
    def stop(self) -> None:
        self._stop = True
        self.status.emit("Stopping...")

    def _emit_pointer(self, x: int, y: int, kind: str) -> None:
        """Relay a click location to the GUI so it can flash it (debug aid)."""
        self.action_pointer.emit(int(x), int(y), str(kind))

    @property
    def last_answer(self) -> Optional[AiAnswer]:
        return self._last_answer

    @property
    def last_ocr(self) -> Optional[OcrResult]:
        return self._last_ocr

    # --------------------------------------------------------------------- stages
    def run(self, request: PipelineRequest) -> None:
        """Full pipeline: read screen -> ask AI -> (preview) -> act."""
        self._stop = False
        self._request = request
        try:
            self.log_line.emit("[1/9] Run started")
            image = self._stage_capture(request)
            ocr = self._stage_ocr(image, request)
            if ocr is None:
                return
            answer = self._stage_ai(ocr, image, request)
            if answer is None:
                return
            self._stage_action(answer, request, ocr)
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("Pipeline crashed")
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()

    def _stage_capture(self, request: PipelineRequest):
        self.status.emit(f"Capturing {request.source_region}...")
        # STEP 2: hide the panel BEFORE grabbing, otherwise the screenshot (and
        # therefore OCR and the matched click point) contains our own UI.
        self.log_line.emit("[2/9] Hiding the panel while capturing")
        if self.pre_capture_hook is not None:
            try:
                self.pre_capture_hook(request.source_region)
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("pre_capture_hook failed: %s", exc)
        try:
            image = self.capture.grab(request.source_region)
        except Exception as exc:
            self.failed.emit(f"Screen capture failed: {exc}")
            return None
        finally:
            # STEP 3: bring the panel back right after the pixels are grabbed.
            if self.post_capture_hook is not None:
                try:
                    self.post_capture_hook()
                except Exception as exc:  # pragma: no cover - defensive
                    log.warning("post_capture_hook failed: %s", exc)
            self.log_line.emit("[3/9] Panel shown again")
        self.captured.emit(image)
        self.log_line.emit(f"Captured {image.width}x{image.height}px from \"{request.source_region.name}\"")
        return image

    def _stage_ocr(self, image, request: PipelineRequest) -> Optional[OcrResult]:
        if image is None:  # capture stage already reported the failure
            return None
        vision_only = bool(self.cfg.get_path("ai.use_vision", False)) and \
            bool(self.cfg.get_path("ai.vision_reads_text", False))
        if vision_only:
            ocr = self._vision_transcribe(image, request) or OcrResult(
                text="", engine="vision", image_size=getattr(image, "size", (0, 0)),
                note="Vision transcription unavailable.",
            )
        else:
            ocr = self.ocr.read(image, request.source_region)
        if ocr.engine == "unavailable":
            self.log_line.emit(ocr.note)
            fallback = self._vision_transcribe(image, request)
            ocr = fallback if fallback is not None and fallback.ok else ocr
        self.ocr_done.emit(ocr)
        if not ocr.ok:
            self.failed.emit("No readable text found in that region.")
            return None
        self.log_line.emit(f"OCR: {ocr.summary()}")
        self._last_ocr = ocr
        return ocr

    def _vision_transcribe(self, image, request: PipelineRequest) -> Optional[OcrResult]:
        """Ask the model itself to transcribe the pixels (no Tesseract installed)."""
        from .ai_client import NO_TEXT_MARKER, vision_transcribe_request

        if not self.cfg.api_key:
            return None
        self.status.emit("Reading text with the vision model...")
        system, user = vision_transcribe_request(
            "You are a precise OCR engine. Transcribe text exactly.", image
        )
        try:
            answer = self._ask_raw(system, user, image)
        except Exception as exc:
            self.log_line.emit(f"Vision transcription failed: {exc}")
            return None
        text = (answer.text or "").strip()
        if not text or NO_TEXT_MARKER in text.upper():
            return OcrResult(text="", engine="vision", image_size=image.size,
                             note="Vision model found no text.")
        return OcrResult(text=text, confidence=95.0, engine="vision",
                         image_size=image.size, note="Text read by vision model.")

    def _ask_raw(self, system: str, user: str, image=None) -> AiAnswer:
        """Send a one-off prompt by temporarily overriding the config templates."""
        ocr_stub = OcrResult(text=user, confidence=100.0, engine="raw")
        saved_system = self.cfg.get_path("prompt.system")
        saved_template = self.cfg.get_path("prompt.user_template")
        self.cfg.set_path("prompt.system", system)
        self.cfg.set_path("prompt.user_template", "{{TEXT}}")
        try:
            return self.ai.ask(ocr_stub, task_key="answer", image=image)
        finally:
            self.cfg.set_path("prompt.system", saved_system)
            self.cfg.set_path("prompt.user_template", saved_template)

    def _stage_ai(self, ocr: OcrResult, image, request: PipelineRequest) -> Optional[AiAnswer]:
        if self._stop:
            self.failed.emit("Stopped before AI request.")
            return None
        self.status.emit("Asking AI...")
        self.log_line.emit(f"[4/9] Sending the question to the AI (task={request.task_key})")
        started = time.time()
        try:
            answer = self.ai.ask(
                ocr,
                task_key=request.task_key,
                image=image if self.cfg.get_path("ai.send_image", False) else None,
                region=request.source_region,
                extra_context=request.extra_context,
                on_progress=lambda msg: self.status.emit(msg),
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return None
        answer.latency_ms = int((time.time() - started) * 1000)
        max_chars = int(self.cfg.get_path("ai.max_answer_chars", 1200))
        if len(answer.text) > max_chars:
            answer.text = answer.text[:max_chars].rstrip() + "..."
        self.answer_ready.emit(answer)
        self._last_answer = answer
        self.log_line.emit(f"[5/9] AI replied: {answer.one_liner()}")
        self.log_line.emit(
            f"AI ({answer.provider}/{answer.model}, {answer.latency_ms}ms): {answer.one_liner()}"
        )
        self._record_history(request, ocr, answer)
        return answer

    def _stage_action(self, answer: AiAnswer, request: PipelineRequest, ocr: OcrResult) -> None:
        self.log_line.emit(f"[action] mode={request.mode} task={request.task_key}")
        if request.task_key == "choose_option" and request.mode != "click_matching_line":
            self.log_line.emit(
                "[warn] Task is 'choose_option' but the Action mode is "
                f"'{request.mode}' - that mode clicks the region CENTER and types the "
                "answer. Pick 'Click the line matching the AI answer (MCQ)' in the "
                "Action dropdown to click the chosen option instead."
            )
        if request.dry_run:
            self.status.emit("Dry run: no clicks/typing performed.")
            return
        if not request.auto_action or self.cfg.get_path("output.clipboard_only", False):
            ok = self.input.copy_to_clipboard(answer.text)
            self.status.emit("Answer copied to clipboard." if ok else "Clipboard unavailable.")
            self.action_done.emit(ActionResult(ok=ok, error="" if ok else "clipboard failed"))
            return

        if request.mode == "click_matching_line":
            self._act_click_matching_line(answer, request, ocr)
            return

        target = request.target_region
        click_point = pointer_position() if request.use_cursor else None
        if request.use_cursor:
            mode = "type_at_cursor"
        elif target is None and request.mode in ("type_at_cursor",):
            mode = "type_at_cursor"
        else:
            mode = request.mode
        steps = self.planner.build(
            text=answer.text,
            mode=mode,
            source_region=request.source_region,
            target_region=target,
            click_point=click_point,
            clear_field=request.clear_field,
            press_enter=request.press_enter,
        )
        if not steps:
            self.status.emit("Nothing to do (copy-only mode).")
            return

        descriptions = [step.describe() for step in steps]
        needs_confirm = bool(self.cfg.get_path("actions.confirm_before_action", True))
        if needs_confirm:
            self._pending_steps = steps
            self.action_preview.emit(descriptions)
            self.status.emit("Waiting for confirmation of actions...")
            return
        self.execute_steps(steps)

    def _act_click_matching_line(
        self, answer: AiAnswer, request: PipelineRequest, ocr: OcrResult
    ) -> None:
        """MCQ mode: click the OCR line whose text matches the AI's chosen option."""
        from .actions import ActionStep
        from .ocr import clean_option_text, find_line_center

        target_text = clean_option_text(answer.text)
        region = request.source_region
        # The captured OCR result normally carries word boxes; fall back to the
        # last real run when this was triggered from manually-edited text.
        word_source = ocr if (ocr and ocr.words) else self._last_ocr
        if word_source is None or not word_source.words:
            self.failed.emit(
                "MCQ click mode needs OCR word positions. Run on the region once first."
            )
            return
        if word_source is not ocr:
            self.log_line.emit(
                "[warn] No OCR boxes in this result - reusing the previous capture's "
                "boxes. Coordinates may be stale; run the full pipeline instead."
            )
        match = find_line_center(
            self.ocr.lines(word_source), target_text, region.left, region.top
        )
        if match is None:
            snippet = target_text or answer.text
            self.log_line.emit(f"[6/9] No OCR line matched: {snippet!r}")
            self.failed.emit(f'Could not locate an option matching "{snippet}" on screen.')
            return
        gx, gy, matched_text = match
        # Guard: a hit inside our own panel means the capture contained our UI, so
        # the coordinates describe a place in the bot window, not the option.
        panel = self.panel_rect_provider() if self.panel_rect_provider is not None else None
        if panel is not None:
            left, top, width, height = panel
            if left <= gx < left + width and top <= gy < top + height:
                self.log_line.emit(
                    f"[6/9] Matched \"{matched_text}\" at ({gx},{gy}) - INSIDE the bot "
                    "panel, so the capture included our own window."
                )
                self.failed.emit(
                    "Refusing to click: the answer matched text inside the bot window. "
                    "Enable 'Hide while acting' and keep the panel off the read area."
                )
                return
        self.log_line.emit(f"[6/9] Answer found on screen: \"{matched_text}\" -> ({gx},{gy})")
        if not (region.left <= gx < region.right and region.top <= gy < region.bottom):
            self.log_line.emit(
                f"[warn] Click point ({gx},{gy}) is outside the read area "
                f"[{region.left},{region.top} {region.width}x{region.height}] - "
                "check display scaling / stale OCR boxes."
            )
        self.log_line.emit(f'[option] Matched "{matched_text}" -> click ({gx},{gy})')
        repeat = max(1, int(self.cfg.get_path("actions.mcq_click_repeat", 2) or 1))
        gap_ms = max(0, int(self.cfg.get_path("actions.mcq_click_gap_ms", 600) or 0))
        steps: List[Any] = []
        for index in range(repeat):
            if index:
                # A real pause keeps the OS from merging the clicks into one
                # double-click, which is what "two single left clicks" means.
                steps.append(ActionStep(kind="wait", text=f"{gap_ms / 1000.0:.2f}s"))
            steps.append(ActionStep(kind="click", x=int(gx), y=int(gy)))
        self.log_line.emit(
            f"[option] Sending {repeat} separate left click(s), {gap_ms}ms apart"
        )
        needs_confirm = bool(self.cfg.get_path("actions.confirm_before_action", True))
        if needs_confirm:
            self._pending_steps = steps
            self.action_preview.emit([step.describe() for step in steps])
            self.status.emit("Waiting for confirmation of the option click...")
            return
        self.execute_steps(steps)

    def execute_steps(self, steps: Optional[List[Any]] = None) -> None:
        steps = steps if steps is not None else self._pending_steps
        self._pending_steps = []
        if not steps:
            return
        pause_ms = int(self.cfg.get_path("actions.pause_after_answer_ms", 350))
        if pause_ms:
            time.sleep(pause_ms / 1000.0)
        self.status.emit(f"Performing {len(steps)} action(s)...")
        # Let the GUI move its own window out of the way first: if the click
        # point sits under the bot window, the click would land on the bot
        # instead of the application underneath.
        will_click = any(
            getattr(step, "kind", "") in ("click", "double_click", "right_click")
            for step in steps
        )
        point = self._first_input_point(steps)
        self.log_line.emit("[7/9] Hiding the panel for the click")
        hidden = False
        try:
            if self.pre_action_hook is not None:
                hidden = bool(self.pre_action_hook(point[0], point[1], will_click))
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("pre_action_hook failed: %s", exc)
        self.log_line.emit(f"[8/9] Clicking ({point[0]},{point[1]})")
        try:
            result = self.input.run_steps(steps, stop_flag=_FlagView(self, "_stop"))
        finally:
            # Only wait when we actually hid the panel: this gives the target
            # application time to finish reacting before our window reappears.
            delay_ms = 0
            if hidden:
                delay_ms = max(0, int(self.cfg.get_path("actions.restore_delay_ms", 1000) or 0))
            if delay_ms:
                self.log_line.emit(f"[9/9] Waiting {delay_ms}ms before showing the panel")
                time.sleep(delay_ms / 1000.0)
            if self.post_action_hook is not None:
                try:
                    self.post_action_hook()
                except Exception as exc:  # pragma: no cover - defensive
                    log.warning("post_action_hook failed: %s", exc)
            self.log_line.emit("[9/9] Panel shown again")
        self.action_done.emit(result)
        if result.ok:
            self.log_line.emit(f"Actions completed ({result.executed} steps).")
        else:
            self.failed.emit(result.error or "Action failed.")

    @staticmethod
    def _first_input_point(steps: List[Any]) -> tuple:
        """Return the (x, y) the action will touch first, for occlusion checks."""
        for step in steps:
            if getattr(step, "kind", "") == "wait":
                continue  # wait steps carry no coordinates
            x = getattr(step, "x", None)
            y = getattr(step, "y", None)
            if x is not None and y is not None:
                return int(x), int(y)
        try:
            return pointer_position()
        except Exception:  # pragma: no cover - headless
            return 0, 0

    def paste_last_answer_at_cursor(self) -> None:
        if self._last_answer is None:
            self.failed.emit("No answer available yet.")
            return
        from .actions import ActionStep

        x, y = pointer_position()
        steps = [
            ActionStep(kind="click", x=x, y=y),
            ActionStep(kind="paste", x=x, y=y, text=self._last_answer.text),
        ]
        self.execute_steps(steps)

    # -------------------------------------------------------------------- history
    def _record_history(self, request: PipelineRequest, ocr: OcrResult, answer: AiAnswer) -> None:
        if not self.cfg.get_path("output.save_history", True):
            return
        name = str(self.cfg.get_path("output.history_file", "history.jsonl"))
        path = Path(name)
        if not path.is_absolute():
            from .config import APP_DIR

            path = APP_DIR / path
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "region": request.source_region.to_dict(),
            "task": request.task_key,
            "ocr_confidence": ocr.confidence,
            "ocr_engine": ocr.engine,
            "text": ocr.text,
            "answer": answer.text,
            "model": answer.model,
            "provider": answer.provider,
            "latency_ms": answer.latency_ms,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            log.warning("Could not write history: %s", exc)


class _FlagView:
    """Minimal adapter so the worker loop can honour the stop flag."""

    def __init__(self, owner: PipelineWorker, attr: str) -> None:
        self.owner = owner
        self.attr = attr

    def is_set(self) -> bool:
        return bool(getattr(self.owner, self.attr))


def preview_prompt(cfg: Config, ocr: OcrResult, task_key: str, region: Optional[Region]) -> str:
    system, user = build_prompt(cfg, ocr, task_key, region)
    return f"[system]\n{system}\n\n[user]\n{user}"
