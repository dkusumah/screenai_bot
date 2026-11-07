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
    },
    "hotkeys": {
        "select_region": "ctrl+alt+r",
        "run": "ctrl+alt+enter",
        "stop": "ctrl+alt+s",
        "repeat_last": "ctrl+alt+l",
        "paste_answer": "ctrl+alt+v",
        "toggle_overlay": "ctrl+alt+h",
    },
    "prompt": {
        "system": (
            "You are a helpful desktop assistant. You receive text read from a "
            "region of the user's screen. Reply concisely in plain text so the "
            "reply can be typed into a small on-screen field."
        ),
        "user_template": 'Captured text:\n"""\n{{TEXT}}\n"""\n\nTask: {{TASK}}',
        "tasks": {"answer": "Answer the question in the captured text."},
        "target_lang": "English",
        "custom_instruction": "Explain the captured text.",
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
        merged = deep_merge(deep_merge(DEFAULTS, json.loads(BUNDLED_CONFIG.read_text(encoding="utf-8")))
                            if BUNDLED_CONFIG.exists() else DEFAULTS, user_data)
        _cached = Config(merged, path)
        return _cached


def reset_config() -> Config:
    global _cached
    with _lock:
        _cached = None
        return load_config(force=True)
