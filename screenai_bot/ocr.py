"""OCR layer: turn a screenshot of a screen region into text (+ confidence).

Primary engine is Tesseract via ``pytesseract``. If Tesseract is not installed we
transparently fall back to a *vision LLM* read (the AI provider looks at the
image and transcribes it), so the bot still works out of the box on a clean
machine as long as an API key is configured.
"""

from __future__ import annotations

import io
import logging
import re
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import Config, Region

log = logging.getLogger("screenai.ocr")

try:
    from PIL import Image, ImageFilter, ImageOps
except Exception:  # pragma: no cover
    Image = None  # type: ignore

try:
    import pytesseract
    from pytesseract import Output
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore
    Output = None  # type: ignore


@dataclass
class OcrWord:
    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float

    @property
    def center(self):
        return (self.left + self.width // 2, self.top + self.height // 2)


@dataclass
class OcrResult:
    text: str
    confidence: float = 0.0
    engine: str = "none"
    words: List[OcrWord] = field(default_factory=list)
    image_size: tuple = (0, 0)
    note: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.text.strip())

    def summary(self) -> str:
        preview = " ".join(self.text.split())
        if len(preview) > 90:
            preview = preview[:87] + "..."
        return f"{self.engine} {self.confidence:.0f}% \"{preview}\""


def tesseract_binary(cfg: Optional[Config] = None) -> Optional[str]:
    """Resolve the tesseract executable (config override -> PATH -> common paths)."""
    candidates: List[str] = []
    if cfg is not None:
        explicit = cfg.get_path("ocr.tesseract_cmd")
        if explicit:
            candidates.append(str(explicit))
    found = shutil.which("tesseract")
    if found:
        candidates.append(found)
    candidates += [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/opt/homebrew/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/usr/bin/tesseract",
    ]
    for path in candidates:
        try:
            import os

            if path and os.path.exists(path):
                return path
        except Exception:
            continue
    return None


def preprocess(image: "Image.Image", upscale_target_height: int = 34) -> "Image.Image":
    """Greyscale -> autocontrast -> upscale -> light sharpen.

    Tesseract likes ~30px tall black-on-white glyphs; small UI text is usually
    much smaller than that, hence the conditional upscale.
    """
    img = image.convert("L")
    if img.height < upscale_target_height:
        ratio = max(2, min(6, round(upscale_target_height / max(1, img.height))))
        img = img.resize((img.width * ratio, img.height * ratio), Image.LANCZOS)
    img = ImageOps.autocontrast(img, cutoff=1)
    img = img.filter(ImageFilter.SHARPEN)
    return img


class OcrEngine:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.binary = tesseract_binary(cfg)
        if self.binary and pytesseract is not None:
            pytesseract.pytesseract.tesseract_cmd = self.binary
        log.info("Tesseract binary: %s", self.binary or "NOT FOUND")

    @property
    def available(self) -> bool:
        return bool(self.binary) and pytesseract is not None

    # ------------------------------------------------------------------ reading
    def read(self, image: "Image.Image", region: Optional[Region] = None) -> OcrResult:
        if Image is None:
            raise RuntimeError("Pillow is required for OCR.")
        size = image.size
        if not self.available:
            return OcrResult(
                text="",
                engine="unavailable",
                image_size=size,
                note=(
                    "Tesseract not found. Install it (Windows: UB-Mannheim installer, "
                    "macOS: brew install tesseract, Debian/Ubuntu: apt install tesseract-ocr) "
                    "or enable ai.use_vision=true to let the model read the pixels."
                ),
            )
        work = preprocess(image) if self.cfg.get_path("ocr.preprocess", True) else image.convert("L")
        lang = str(self.cfg.get_path("ocr.language", "eng"))
        psm = int(self.cfg.get_path("ocr.psm", 6))
        config = f"--oem 3 --psm {psm}"
        try:
            data: Dict[str, Any] = pytesseract.image_to_data(
                work, lang=lang, config=config, output_type=Output.DICT
            )
            text = pytesseract.image_to_string(work, lang=lang, config=config)
        except Exception as exc:
            log.exception("OCR failed")
            return OcrResult(text="", engine="error", image_size=size, note=f"OCR error: {exc}")

        scale = work.height / max(1, image.height)
        words: List[OcrWord] = []
        confidences: List[float] = []
        for i, raw in enumerate(data.get("text", [])):
            token = (raw or "").strip()
            if not token:
                continue
            try:
                conf = float(data["conf"][i])
            except (KeyError, TypeError, ValueError):
                conf = -1.0
            if conf < 0:
                continue
            confidences.append(conf)
            words.append(
                OcrWord(
                    text=token,
                    left=int(int(data["left"][i]) / scale),
                    top=int(int(data["top"][i]) / scale),
                    width=int(int(data["width"][i]) / scale),
                    height=int(int(data["height"][i]) / scale),
                    confidence=conf,
                )
            )
        overall = sum(confidences) / len(confidences) if confidences else 0.0
        cleaned = "\n".join(line.rstrip() for line in text.splitlines()).strip()
        return OcrResult(
            text=cleaned,
            confidence=round(overall, 1),
            engine="tesseract",
            words=words,
            image_size=size,
            note="" if overall >= float(self.cfg.get_path("ocr.min_confidence", 45))
            else "Low OCR confidence - consider enlarging the region or using vision mode.",
        )

    # ------------------------------------------------------------------- helpers
    def word_at(self, result: OcrResult, x: int, y: int, padding: int = 4) -> Optional[OcrWord]:
        for word in result.words:
            if (word.left - padding <= x <= word.left + word.width + padding
                    and word.top - padding <= y <= word.top + word.height + padding):
                return word
        return None

    def lines(self, result: OcrResult, tolerance: int = 6) -> List[List[OcrWord]]:
        rows: List[List[OcrWord]] = []
        for word in sorted(result.words, key=lambda w: (w.top, w.left)):
            placed = False
            for row in rows:
                if abs(row[0].top - word.top) <= tolerance:
                    row.append(word)
                    placed = True
                    break
            if not placed:
                rows.append([word])
        return rows


def normalize_match_text(text: str) -> str:
    """Lowercase and strip punctuation/quotes so OCR lines compare loosely."""
    value = str(text or "").lower()
    value = value.replace("\u2019", "'").replace("\u2018", "'")
    value = value.replace("\u201c", '"').replace("\u201d", '"')
    value = re.sub(r"[^0-9a-z\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def clean_option_text(text: str) -> str:
    """Trim an AI answer down to a bare option phrase (drops labels/quotes)."""
    value = str(text or "").strip().strip('"').strip("'").strip()
    # drop a leading "Answer:" / "The correct option is" style label
    value = re.sub(
        r"^(?:the\s+)?(?:correct\s+)?(?:answer|option|choice)\s*(?:is|:)?\s*",
        "", value, flags=re.IGNORECASE,
    )
    value = value.strip().strip('"').strip("'").strip()
    # drop a leading option marker like "B) ", "(B) ", "B. " or "B: "
    value = re.sub(r"^[\(\[]?[A-Za-z][\)\].:\-]\s+", "", value).strip()
    return value


def _line_match_score(target_norm: str, line_norm: str) -> float:
    if not target_norm or not line_norm:
        return 0.0
    if target_norm == line_norm:
        return 100.0
    if target_norm in line_norm or line_norm in target_norm:
        return 80.0
    target_tokens = set(target_norm.split())
    line_tokens = set(line_norm.split())
    if not target_tokens:
        return 0.0
    overlap = len(target_tokens & line_tokens) / len(target_tokens)
    return 50.0 * overlap if overlap >= 0.6 else 0.0


def find_line_center(
    lines: List[List["OcrWord"]], target: str, region_left: int = 0, region_top: int = 0
) -> Optional[Tuple[int, int, str]]:
    """Locate the OCR line best matching ``target`` and return its global center.

    ``lines`` comes from :meth:`OcrEngine.lines`. Coordinates in ``lines`` are
    region-local, so ``region_left``/``region_top`` convert them to the global
    (virtual-desktop) space used for clicking. Returns ``(x, y, matched_text)``
    or ``None`` when nothing scores above zero.
    """
    norm_target = normalize_match_text(target)
    if not norm_target:
        return None
    best: Optional[Tuple[float, List["OcrWord"], str]] = None
    for line in lines:
        words = [w for w in line if w.text.strip()]
        if not words:
            continue
        words.sort(key=lambda w: w.left)
        line_text = " ".join(w.text for w in words)
        score = _line_match_score(norm_target, normalize_match_text(line_text))
        if score > 0 and (best is None or score > best[0]):
            best = (score, words, line_text)
    if best is None:
        return None
    _, words, line_text = best
    left = min(w.left for w in words)
    top = min(w.top for w in words)
    right = max(w.left + w.width for w in words)
    bottom = max(w.top + w.height for w in words)
    cx = int(region_left + (left + right) / 2)
    cy = int(region_top + (top + bottom) / 2)
    return cx, cy, line_text


def image_to_base64_png(image: "Image.Image", max_side: int = 1600) -> str:
    """PNG -> base64, downscaled so vision requests stay cheap and fast."""
    import base64

    img = image
    if max(img.size) > max_side:
        ratio = max_side / max(img.size)
        img = img.resize((max(1, int(img.width * ratio)), max(1, int(img.height * ratio))), Image.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
