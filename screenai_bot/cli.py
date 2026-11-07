"""Headless / CLI entry point: run the whole pipeline without any GUI.

This is useful for servers, CI smoke tests, scripting the bot from other tools,
and for verifying that capture + OCR + AI wiring works before launching Qt.

Examples
--------
    # read a region and print the AI answer (no clicks/typing)
    python -m screenai_bot.cli --region 100,100,600,200 --task answer --dry-run

    # read the region and type the answer into another region
    python -m screenai_bot.cli --region 100,100,600,200 \
        --target 800,400,300,40 --mode click_and_type --yes

    # OCR only (debug what the bot sees)
    python -m screenai_bot.cli --region 100,100,600,200 --ocr-only

    # feed text yourself instead of reading the screen
    python -m screenai_bot.cli --text "What is the capital of France?" --task answer
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional

from .actions import ActionPlanner, InputController
from .ai_client import AIClient
from .capture import ScreenCapture
from .config import Config, Region, load_config
from .ocr import OcrEngine, OcrResult


def parse_region(raw: str, name: str = "cli") -> Optional[Region]:
    if not raw:
        return None
    parts = [p.strip() for p in raw.replace(";", ",").split(",")]
    if len(parts) != 4:
        raise ValueError(f"Region must be 'left,top,width,height' (got {raw!r})")
    left, top, width, height = (int(float(p)) for p in parts)
    return Region(name=name, left=left, top=top, width=width, height=height)


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="screenai-bot-cli",
        description="Read a screen region, ask an AI, then click/type the answer.",
    )
    parser.add_argument("--region", help="Source area as left,top,width,height (virtual desktop px)")
    parser.add_argument("--full-screen", action="store_true", help="Use the entire desktop as source")
    parser.add_argument("--monitor", type=int, default=0, help="Use monitor N as source (1-based)")
    parser.add_argument("--text", help="Skip capture/OCR and use this text as the input")
    parser.add_argument("--image", help="Run OCR on a PNG file instead of the live screen")
    parser.add_argument("--task", default="answer", help="Task key from config.prompt.tasks")
    parser.add_argument("--context", default="", help="Extra context appended to the prompt")
    parser.add_argument("--target", help="Destination area left,top,width,height for the answer")
    parser.add_argument("--mode", default="type_in_region",
                        choices=list(ActionPlanner.MODES), help="How to deliver the answer")
    parser.add_argument("--press-enter", action="store_true", help="Press Enter after typing")
    parser.add_argument("--no-clear", action="store_true", help="Do not select-all+delete first")
    parser.add_argument("--dry-run", action="store_true", help="Never click or type")
    parser.add_argument("--ocr-only", action="store_true", help="Stop after OCR")
    parser.add_argument("--ask-only", "--no-action", dest="ask_only", action="store_true",
                        help="Stop after the AI answer (copy to clipboard only)")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip the interactive confirmation")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    parser.add_argument("--model", help="Override ai.model")
    parser.add_argument("--base-url", help="Override ai.base_url (e.g. http://localhost:11434/v1)")
    parser.add_argument("--provider", help="Override ai.provider (openai|anthropic)")
    parser.add_argument("--api-key", help="Override the API key (else env/config is used)")
    parser.add_argument("--vision", action="store_true", help="Send the screenshot to the model too")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def apply_overrides(cfg: Config, args: argparse.Namespace) -> None:
    if args.model:
        cfg.set_path("ai.model", args.model)
    if args.base_url:
        cfg.set_path("ai.base_url", args.base_url)
    if args.provider:
        cfg.set_path("ai.provider", args.provider)
    if args.api_key:
        cfg.set_path("ai.api_key", args.api_key)
    if args.vision:
        cfg.set_path("ai.use_vision", True)
    if args.yes:
        cfg.set_path("actions.confirm_before_action", False)


def resolve_source(cfg: Config, args: argparse.Namespace) -> Optional[Region]:
    region = parse_region(args.region or "")
    if region:
        return region.normalized()
    if args.full_screen:
        try:
            return ScreenCapture().fullscreen_region()
        except Exception as exc:
            print(f"--full-screen unavailable: {exc}", file=sys.stderr)
            return None
    if args.monitor:
        try:
            capture = ScreenCapture()
            monitors = capture.monitors
            if args.monitor < len(monitors):
                mon = monitors[args.monitor]
                return Region(f"monitor{args.monitor}", mon.left, mon.top, mon.width, mon.height)
        except Exception as exc:
            print(f"--monitor unavailable: {exc}", file=sys.stderr)
    saved = cfg.regions
    if saved:
        print(f"Using saved region \"{saved[0].name}\" (pass --region to override)", file=sys.stderr)
        return saved[0]
    return None


def get_input_image(args: argparse.Namespace, source: Optional[Region]):
    if args.image:
        from PIL import Image

        return Image.open(args.image).convert("RGB"), source or Region("file", 0, 0, 1, 1)
    if source is None:
        raise SystemExit("No source: give --region/--full-screen/--monitor/--image/--text.")
    capture = ScreenCapture()
    return capture.grab(source), source


def confirm(steps: List[str]) -> bool:
    print("\nAbout to perform these actions:")
    for line in steps:
        print(f"  - {line}")
    try:
        answer = input("Type the answer onto your screen? [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def _vision_transcribe_cli(cfg: Config, image, verbose: bool) -> Optional[OcrResult]:
    """Transcribe an image with the AI provider when Tesseract is missing."""
    if not cfg.api_key:
        return None
    from .ai_client import NO_TEXT_MARKER, vision_transcribe_request

    system, user = vision_transcribe_request(
        "You are a precise OCR engine. Transcribe text exactly.", image
    )
    saved_system = cfg.get_path("prompt.system")
    saved_template = cfg.get_path("prompt.user_template")
    cfg.set_path("prompt.system", system)
    cfg.set_path("prompt.user_template", "{{TEXT}}")
    try:
        stub = OcrResult(text=user, confidence=100.0, engine="vision-ocr")
        answer = AIClient(cfg).ask(stub, task_key="answer", image=image)
    except Exception as exc:
        print(f"[warn] vision transcription failed: {exc}", file=sys.stderr)
        return None
    finally:
        cfg.set_path("prompt.system", saved_system)
        cfg.set_path("prompt.user_template", saved_template)
    text = (answer.text or "").strip()
    if not text or NO_TEXT_MARKER in text.upper():
        return OcrResult(text="", engine="vision", image_size=getattr(image, "size", (0, 0)),
                         note="Vision model found no text.")
    if verbose:
        print("[info] text read by vision model", file=sys.stderr)
    return OcrResult(text=text, confidence=95.0, engine="vision",
                     image_size=getattr(image, "size", (0, 0)), note="Read by vision model.")


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    cfg = load_config(force=True)
    apply_overrides(cfg, args)

    payload = {"ok": False, "stage": "init"}

    # ------------------------------------------------------------------ read text
    ocr: Optional[OcrResult] = None
    image = None
    source = resolve_source(cfg, args)
    if args.text:
        ocr = OcrResult(text=args.text, confidence=100.0, engine="cli", note="manual input")
        payload["stage"] = "input"
    else:
        try:
            image, source = get_input_image(args, source)
        except SystemExit as exc:
            print(str(exc), file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"Capture failed: {exc}", file=sys.stderr)
            return 2
        engine = OcrEngine(cfg)
        ocr = engine.read(image, source)
        payload["stage"] = "ocr"
        if not ocr.ok and ocr.engine in ("unavailable", "error"):
            # No Tesseract (or OCR blew up): let the vision model read the pixels.
            payload["note"] = ocr.note
            print(f"[warn] {ocr.note}", file=sys.stderr)
            vision_ocr = _vision_transcribe_cli(cfg, image, args.verbose)
            if vision_ocr is not None:
                ocr = vision_ocr
        if not ocr.ok:
            print(f"No readable text. ({ocr.engine}) {ocr.note}", file=sys.stderr)
            payload.update(ok=False, error="no_text", ocr_engine=ocr.engine, note=ocr.note)
            _emit(args, payload)
            return 3
    if args.ocr_only:
        payload.update(ok=True, stage="ocr_done", text=ocr.text, confidence=ocr.confidence,
                       engine=ocr.engine)
        if args.json:
            _emit(args, payload)
        else:
            print(ocr.text)
        return 0

    # ------------------------------------------------------------------- ask AI
    client = AIClient(cfg)
    try:
        answer = client.ask(ocr, task_key=args.task, image=image if cfg.get_path("ai.use_vision") else None,
                            region=source, extra_context=args.context)
    except Exception as exc:
        print(f"AI request failed: {exc}", file=sys.stderr)
        payload.update(ok=False, stage="ai", error=str(exc))
        _emit(args, payload)
        return 4
    payload.update(stage="ai_done", answer=answer.text, model=answer.model,
                   provider=answer.provider, latency_ms=answer.latency_ms,
                   ocr_text=ocr.text, ocr_confidence=ocr.confidence)

    if args.ask_only or args.dry_run:
        payload.update(ok=True, acted=False)
        if args.dry_run and not args.ask_only:
            planner = ActionPlanner(cfg)
            target = parse_region(args.target or "", "target")
            payload["planned_actions"] = [s.describe() for s in planner.build(
                answer.text, mode=args.mode, source_region=source, target_region=target,
                clear_field=not args.no_clear, press_enter=args.press_enter)]
        _emit(args, payload, show_answer=not args.json)
        return 0

    # ----------------------------------------------------------------- do actions
    controller = InputController(cfg)
    planner = ActionPlanner(cfg)
    target = parse_region(args.target or "", "target")
    steps = planner.build(
        answer.text,
        mode=args.mode,
        source_region=source,
        target_region=target,
        clear_field=not args.no_clear,
        press_enter=args.press_enter,
    )
    descriptions = [step.describe() for step in steps]
    payload["planned_actions"] = descriptions
    if cfg.get_path("actions.confirm_before_action", True) and not confirm(descriptions):
        ok = controller.copy_to_clipboard(answer.text)
        payload.update(ok=True, acted=False, copied=ok, reason="user declined actions")
        _emit(args, payload, show_answer=True)
        return 0
    result = controller.run_steps(steps)
    payload.update(ok=result.ok, acted=True, executed=result.executed, error=result.error)
    _emit(args, payload, show_answer=True)
    return 0 if result.ok else 5


def _emit(args: argparse.Namespace, payload: dict, show_answer: bool = False) -> None:
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif show_answer:
        print("\nANSWER:\n" + str(payload.get("answer", "")))
        status = "typed/pasted" if payload.get("acted") else "not applied"
        print(f"\n({status}; {payload.get('executed', 0)} action(s))")


if __name__ == "__main__":
    raise SystemExit(main())
