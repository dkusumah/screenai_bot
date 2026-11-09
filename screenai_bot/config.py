"""Typed access to config.json with safe defaults and hot reloading.

Everything the bot does is driven by this file, so no code edits are needed to
change models, prompts, hotkeys or action behaviour.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

APP_NAME = "ScreenAI Bot"
APP_DIR = Path(os.environ.get("SCREENAI_HOME", Path.home() / ".screenai_bot"))
CONFIG_PATH = APP_DIR / "config.json"
BUNDLED_CONFIG = Path(__file__).resolve().with_name("config.json")
LOG_PATH = APP_DIR / "bot.log"

# Answer length limits, counted in words. 0 = no limit / no minimum.
# No limit by default: a hard cap silently chopped long explanations in half.
DEFAULT_MAX_ANSWER_WORDS = 0
DEFAULT_MIN_ANSWER_WORDS = 0
MAX_ANSWER_WORDS_LIMIT = 5000

# How often a request is retried when the model stops early (token limit) or
# answers with fewer words than ai.min_answer_words asks for.
DEFAULT_LENGTH_RETRIES = 1

# Formatting privileges for the answer. Off = plain text, which is what can be
# typed into small on-screen fields without looking wrong.
DEFAULT_FORMAT_SWITCHES: Dict[str, bool] = {
    "allow_markdown": False,
    "allow_html": False,
    "allow_code": False,
    "allow_bullet_lists": False,
}

# (feature key in prompt.format_features, config key in the ai section)
FORMAT_FEATURES: Tuple[Tuple[str, str], ...] = (
    ("markdown", "allow_markdown"),
    ("html", "allow_html"),
    ("code", "allow_code"),
    ("bullet_lists", "allow_bullet_lists"),
)

# Selectable strictness of the model, mapped to ``ai.hallucination_level``.
# Index == level value; 0 stays on the captured text, 4 may invent details.
HALLUCINATION_LEVELS: Tuple[str, ...] = (
    "0 - Strict (only what is on screen)",
    "1 - Careful (screen first, firm facts)",
    "2 - Balanced (default)",
    "3 - Creative (fills the gaps)",
    "4 - Unrestricted (may invent details)",
)
DEFAULT_HALLUCINATION_LEVEL = 2

# System-prompt directives appended for each hallucination level. Kept in the
# config file (``prompt.hallucination_levels``) so users can reword them.
HALLUCINATION_DIRECTIVES: Tuple[str, ...] = (
    "Grounding rule: answer using ONLY facts present in the captured text. If the "
    "text does not contain the answer, say plainly that the screen does not show it.",
    "Grounding rule: base the answer on the captured text. Add outside knowledge only "
    "when it is certain and clearly relevant, and say briefly when the text itself does "
    "not answer the question.",
    "Grounding rule: answer from the captured text and use general knowledge when that "
    "makes the answer clearly better; mention uncertainty in a few words when it exists.",
    "Grounding rule: answer even when the captured text is thin - fill small gaps with "
    "the most plausible explanation and name any assumption in a short phrase.",
    "Grounding rule: never refuse and never hedge. Give the most likely answer even if "
    "the captured text is incomplete, inventing plausible details when that produces a "
    "usable answer.",
)

# Added to the system prompt when ``ai.natural_answers`` is enabled.
NATURAL_STYLE_DIRECTIVE = (
    "Answer like a real person typing into a chat box: everyday words and contractions, "
    "no preamble such as 'Sure' or 'The answer is', and never restate the question."
)

DEFAULTS: Dict[str, Any] = {
    "ai": {
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
        "api_key": "",
        "temperature": 0.2,
        "max_tokens": 800,
        "use_vision": False,
        "timeout": 60,
        # Longest answer the bot may type/paste, in words (0 = no limit).
        "max_answer_words": DEFAULT_MAX_ANSWER_WORDS,
        # Shortest answer asked of the model, in words (0 = no minimum).
        "min_answer_words": DEFAULT_MIN_ANSWER_WORDS,
        # Retries when the model stops at the token limit or ignores the minimum.
        "length_retries": DEFAULT_LENGTH_RETRIES,
        # 0 = stay on the captured text ... 4 = free to invent details.
        "hallucination_level": DEFAULT_HALLUCINATION_LEVEL,
        # Rewrite answers so they read like a human typed them.
        "natural_answers": False,
        # Answer formatting privileges (all off = plain text).
        **DEFAULT_FORMAT_SWITCHES,
    },
    "ocr": {
        "tesseract_cmd": None,
        "language": "eng",
        "psm": 6,
        "preprocess": True,
        "min_confidence": 45,
    },
    "actions": {
        "type_interval": 0.012,
        "click_duration": 0.06,
        "move_duration": 0.25,
        "failsafe": True,
        "pause_after_answer_ms": 350,
        "confirm_before_action": True,
        "focus_click": True,
        "clear_field_shortcut": "ctrl+a",
        "debug_pointer": False,
        "debug_pointer_delay_ms": 120,
        # Hide the bot window while input events fire, so a click that lands
        # under our own window reaches the app underneath instead.
        "auto_hide_window": True,
        "auto_hide_delay_ms": 220,
        "auto_hide_max_ms": 8000,
        # Keep the panel hidden from the first click of the write action until the
        # answer has been typed completely (not just while the click happens).
        # Only the task/mode pairs listed below do this, so picky click modes and
        # copy-only runs keep their old behaviour.
        "hide_until_writing_done": True,
        "hide_until_writing_tasks": ["answer"],
        "hide_until_writing_modes": ["type_in_region", "click_and_type"],
        # Watchdog for the "stay hidden while writing" mode: typing a long answer
        # can take a while, but a hung run must still bring the panel back.
        "auto_hide_writing_max_ms": 60000,
        # "always" hides the panel for every capture (recommended: it keeps our
        # own UI out of the screenshot). "overlap" only hides when the read area
        # intersects the panel.
        "auto_hide_capture_mode": "always",
        # MCQ action: how many separate left clicks to send on the answer, and
        # the pause between them. The gap must stay above the OS double-click
        # interval (~0.5s on macOS) or the two clicks merge into a double-click.
        "mcq_click_repeat": 2,
        "mcq_click_gap_ms": 600,
        # Keep the panel hidden this long after the last input event so the
        # target app can finish reacting (page transition, checkbox repaint)
        # before our window pops back on top.
        "restore_delay_ms": 1000,
    },
    # Last GUI selections, persisted so restarting the app does not silently
    # fall back to the default "type_in_region" action (which clicks the
    # region center instead of the chosen MCQ line).
    "ui": {
        "action_mode": "type_in_region",
        "task_key": "answer",
        "show_log": True,
        "show_options": False,
        "show_ai": False,
    },
    "hotkeys": {
        "select_region": "ctrl+alt+r",
        "run": "ctrl+alt+enter",
        "stop": "ctrl+alt+s",
        "repeat_last": "ctrl+alt+l",
        "paste_answer": "ctrl+alt+v",
        "toggle_overlay": "ctrl+alt+h",
    },
    # Background behaviour: closing the panel drops to a menu-bar / tray icon
    # instead of quitting, so the hotkeys keep working with no window on screen.
    "app": {
        "run_in_background": True,
        "start_minimized": False,
        "hide_dock_icon": True,
        "tray_notifications": True,
        # Automatic hotkey-listener rebuilds allowed before the bot stops trying
        # and asks you to press "Restart Hotkeys" (0 = never auto-recover).
        "hotkey_max_restarts": 2,
    },
    # Audible debug feedback ("did the hotkey fire?", "did the click happen?").
    # debug.sounds is the master switch; each cue has its own toggle.
    "debug": {
        "sounds": False,
        "sound_volume": 0.5,
        "sound_hotkey": True,
        "sound_hotkey_miss": False,
        "sound_click": True,
        "sound_type": True,
        "sound_start": False,
        "sound_done": True,
        "sound_error": True,
        "sound_cancel": True,
    },
    "prompt": {
        "system": (
            "You are a helpful desktop assistant. You receive text read from a "
            "region of the user's screen. Answer concisely so the reply can be "
            "typed into an on-screen field."
        ),
        "user_template": 'Captured text:\n"""\n{{TEXT}}\n"""\n\nTask: {{TASK}}',
        "tasks": {
            "answer": "Answer the question in the captured text.",
            "choose_option": (
                "The captured text contains a multiple-choice question and its options. "
                "Reply with ONLY the exact text of the single correct option, copied "
                "verbatim from the captured text. No letter, no quotes, no explanation."
            ),
        },
        "target_lang": "English",
        "custom_instruction": "Explain the captured text.",
        # Length rules appended to the system prompt (placeholders are filled from
        # ai.min_answer_words / ai.max_answer_words). A rule is skipped when its
        # limit is 0.
        "length_rules": {
            "min": "Write at least {{MIN_WORDS}} words.",
            "max": "Keep the answer within {{MAX_WORDS}} words.",
        },
        # Formatting privileges: one line per feature, allowed or denied depending
        # on the ai.allow_* switches. "{feature}" is replaced with the label below.
        "format_features": {
            "markdown": "Markdown formatting",
            "html": "HTML tags",
            "code": "code blocks",
            "bullet_lists": "bullet or numbered lists",
        },
        "format_allow": "You may use {feature} when it makes the answer clearer.",
        "format_deny": "Do not use {feature} in the answer.",
        # One directive per hallucination level (index == level value).
        "hallucination_levels": list(HALLUCINATION_DIRECTIVES),
        # Appended to the system prompt when ai.natural_answers is on.
        "natural_style": NATURAL_STYLE_DIRECTIVE,
    },
    "output": {
        "mode": "type",
        "destination_region": None,
        "clipboard_only": False,
        "save_history": True,
        "history_file": "history.jsonl",
    },
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` onto a copy of ``base``."""
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def migrate_legacy(data: Dict[str, Any]) -> Dict[str, Any]:
    """Fold retired config keys into their current equivalents (in place).

    ``ai.max_answer_chars`` (a character budget) became ``ai.max_answer_words``;
    the stored value is converted at a rough 6 characters per word so an existing
    setting keeps roughly the same answer length.
    """
    ai = data.get("ai")
    if not isinstance(ai, dict) or "max_answer_chars" not in ai:
        return data
    legacy = ai.pop("max_answer_chars")
    if ai.get("max_answer_words") not in (None, ""):
        return data  # the user already set the word-based limit
    try:
        chars = int(legacy or 0)
    except (TypeError, ValueError):
        chars = 0
    ai["max_answer_words"] = 0 if chars <= 0 else max(1, round(chars / 6))
    return data


@dataclass
class Region:
    """A rectangle in virtual-desktop coordinates (spans all monitors)."""

    name: str
    left: int
    top: int
    width: int
    height: int
    monitor_hint: str = ""

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def center(self) -> Tuple[int, int]:
        return (self.left + self.width // 2, self.top + self.height // 2)

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
            "monitor_hint": self.monitor_hint,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], name: str = "") -> "Region":
        return cls(
            name=name or data.get("name", "region"),
            left=int(data["left"]),
            top=int(data["top"]),
            width=int(data["width"]),
            height=int(data["height"]),
            monitor_hint=data.get("monitor_hint", ""),
        )

    def normalized(self) -> "Region":
        """Return an equivalent region with positive width/height."""
        left, top, right, bottom = self.box
        return Region(
            name=self.name,
            left=min(left, right),
            top=min(top, bottom),
            width=abs(right - left),
            height=abs(bottom - top),
            monitor_hint=self.monitor_hint,
        )

    def inflated(self, px: int) -> "Region":
        return Region(
            self.name, self.left - px, self.top - px,
            self.width + 2 * px, self.height + 2 * px, self.monitor_hint,
        )

    def __str__(self) -> str:
        return f"{self.name}[{self.left},{self.top} {self.width}x{self.height}]"


@dataclass
class Config:
    """Dictionary-backed config with dotted-path access."""

    data: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULTS))
    path: Path = CONFIG_PATH

    # ------------------------------------------------------------------ access
    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)

    # ------------------------------------------------------------------- misc
    @property
    def regions(self) -> List[Region]:
        raw = self.data.get("regions") or []
        out: List[Region] = []
        for item in raw:
            try:
                out.append(Region.from_dict(item))
            except Exception:
                continue
        return out

    @regions.setter
    def regions(self, value: List[Region]) -> None:
        self.data["regions"] = [r.to_dict() for r in value]

    def add_region(self, region: Region, limit: int = 12) -> None:
        regions = self.regions
        regions = [r for r in regions if r.name != region.name]
        regions.insert(0, region)
        self.regions = regions[:limit]

    def region_named(self, name: Optional[str]) -> Optional[Region]:
        regions = self.regions
        if not regions:
            return None
        if not name:
            return regions[0]
        for region in regions:
            if region.name == name:
                return region
        return regions[0]

    @property
    def api_key(self) -> str:
        ai = self.data["ai"]
        return (ai.get("api_key") or "").strip() or os.environ.get(
            ai.get("api_key_env") or "OPENAI_API_KEY", ""
        ).strip()


_lock = threading.RLock()
_cached: Optional[Config] = None


def ensure_config_file() -> Path:
    """Copy the bundled template to the user directory on first run."""
    APP_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        if BUNDLED_CONFIG.exists():
            CONFIG_PATH.write_text(BUNDLED_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            Config(dict(DEFAULTS), CONFIG_PATH).save()
    return CONFIG_PATH


def load_config(force: bool = False) -> Config:
    global _cached
    with _lock:
        if _cached is not None and not force:
            return _cached
        path = ensure_config_file()
        try:
            user_data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            user_data = {}
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON in {path}: {exc}") from exc
        # Convert retired keys before merging so a stored setting still wins over
        # the default instead of being masked by it.
        user_data = migrate_legacy(user_data)
        merged = deep_merge(deep_merge(DEFAULTS, json.loads(BUNDLED_CONFIG.read_text(encoding="utf-8")))
                            if BUNDLED_CONFIG.exists() else DEFAULTS, user_data)
        _cached = Config(merged, path)
        return _cached


def reset_config() -> Config:
    global _cached
    with _lock:
        _cached = None
        return load_config(force=True)
