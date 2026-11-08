"""AI client: send the captured text (and optionally the image) to an LLM API.

Speaks plain REST with ``requests`` so it works against any OpenAI-compatible
endpoint (OpenAI, Azure-style gateways, Ollama, LM Studio, vLLM, OpenRouter,
Groq, ...), Anthropic, and Google Gemini, without vendor SDKs.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import requests

from .config import Config, Region
from .ocr import OcrResult, image_to_base64_png

log = logging.getLogger("screenai.ai")

MAX_ANSWER_CHARS_DEFAULT = 1200


class AIError(RuntimeError):
    """Raised for configuration/HTTP problems so the UI can show a clean message."""


@dataclass
class AiAnswer:
    text: str
    model: str = ""
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)

    def one_liner(self) -> str:
        flat = " ".join(self.text.split())
        return flat if len(flat) <= 120 else flat[:117] + "..."


TASK_KEYS = ("answer", "explain", "summarize", "translate", "suggest_input", "custom")


def build_prompt(cfg: Config, ocr: OcrResult, task_key: str, region: Optional[Region]) -> tuple:
    """Render (system, user) messages from config templates."""
    tasks: Dict[str, str] = dict(cfg.get_path("prompt.tasks", {}) or {})
    template_task = tasks.get(task_key) or tasks.get("answer") or "Answer the question in the text."
    max_chars = int(cfg.get_path("ai.max_answer_chars", MAX_ANSWER_CHARS_DEFAULT))

    task_text = (
        template_task.replace("{{TARGET_LANG}}", str(cfg.get_path("prompt.target_lang", "English")))
        .replace("{{CUSTOM}}", str(cfg.get_path("prompt.custom_instruction", "")))
    )

    system = str(cfg.get_path("prompt.system", "")).replace("{{MAX_CHARS}}", str(max_chars))
    user_template = str(
        cfg.get_path(
            "prompt.user_template",
            'Captured text:\n"""\n{{TEXT}}\n"""\n\nTask: {{TASK}}',
        )
    )
    user = (
        user_template.replace("{{TEXT}}", ocr.text or "(no text detected)")
        .replace("{{TASK}}", task_text)
        .replace("{{CONFIDENCE}}", f"{ocr.confidence:.0f}")
        .replace("{{REGION_NAME}}", region.name if region else "selection")
        .replace("{{OCR_NOTE}}", ocr.note or "")
    )
    return system.strip(), user.strip()


class AIClient:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ helpers
    @property
    def provider(self) -> str:
        return str(self.cfg.get_path("ai.provider", "openai")).lower()

    def _base_url(self) -> str:
        return str(self.cfg.get_path("ai.base_url", "https://api.openai.com/v1")).rstrip("/")

    def _headers(self, api_key: str) -> Dict[str, str]:
        if self.provider == "anthropic":
            return {
                "Authorization": f"Bearer {api_key}",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # --------------------------------------------------------------------- ask
    def ask(
        self,
        ocr: OcrResult,
        task_key: str = "answer",
        image: Any = None,
        region: Optional[Region] = None,
        extra_context: str = "",
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> AiAnswer:
        api_key = self.cfg.api_key
        if not api_key:
            raise AIError(
                "No API key configured. Set ai.api_key in ~/.screenai_bot/config.json "
                "or export the environment variable named by ai.api_key_env."
            )
        system, user = build_prompt(self.cfg, ocr, task_key, region)
        if extra_context:
            user = f"{user}\n\nAdditional context: {extra_context}"
        use_vision = bool(self.cfg.get_path("ai.use_vision", False)) and image is not None

        if on_progress:
            on_progress("Sending to AI...")
        payload_desc = f"{self.provider} vision" if use_vision else self.provider
        log.info("AI request via %s (%d chars of OCR text)", payload_desc, len(user))

        if self.provider == "anthropic":
            answer = self._ask_anthropic(api_key, system, user, ocr, image, use_vision)
        elif self.provider == "gemini":
            answer = self._ask_gemini(api_key, system, user, image, use_vision)
        else:
            answer = self._ask_openai_compat(api_key, system, user, ocr, image, use_vision)
        if on_progress:
            on_progress("AI answered.")
        return answer

    # -------------------------------------------------------- openai compatible
    def _ask_openai_compat(
        self, api_key: str, system: str, user: str, ocr: OcrResult, image: Any, vision: bool
    ) -> AiAnswer:
        content: Any = user
        if vision:
            content = [
                {"type": "text", "text": user},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_to_base64_png(image)}"},
                },
            ]
        body = {
            "model": self.cfg.get_path("ai.model", "gpt-4o-mini"),
            "temperature": float(self.cfg.get_path("ai.temperature", 0.2)),
            "max_tokens": int(self.cfg.get_path("ai.max_tokens", 800)),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
        }
        url = f"{self._base_url()}/chat/completions"
        data = self._post(url, self._headers(api_key), body)
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AIError(f"Unexpected response shape: {json.dumps(data)[:400]}") from exc
        usage = data.get("usage", {}) or {}
        return AiAnswer(
            text=_tidy(text),
            model=data.get("model", body["model"]),
            provider=self.provider,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            raw=data,
        )

    # ------------------------------------------------------------------- gemini
    def _ask_gemini(
        self, api_key: str, system: str, user: str, image: Any, vision: bool
    ) -> AiAnswer:
        """Call the native Gemini ``generateContent`` endpoint.

        Uses the ``x-goog-api-key`` header because recent authorisation keys
        (``AQ.`` prefix) are rejected on the OpenAI-compatible route when sent
        as ``Authorization: Bearer``.
        """
        parts: List[Dict[str, Any]] = []
        if vision:
            parts.append(
                {
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": image_to_base64_png(image),
                    }
                }
            )
        parts.append({"text": user})
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": float(self.cfg.get_path("ai.temperature", 0.2)),
                "maxOutputTokens": int(self.cfg.get_path("ai.max_tokens", 800)),
            },
        }
        model = str(self.cfg.get_path("ai.model", "gemini-3.8-flash"))
        url = f"{self._base_url()}/models/{model}:generateContent"
        data = self._post(
            url,
            {"x-goog-api-key": api_key, "Content-Type": "application/json"},
            body,
        )
        try:
            candidates = data.get("candidates") or []
            if not candidates:
                raise AIError(f"Gemini returned no candidates: {json.dumps(data)[:300]}")
            finish = candidates[0].get("finishReason", "")
            parts_out = candidates[0].get("content", {}).get("parts", [])
        except (AttributeError, TypeError) as exc:
            raise AIError(f"Unexpected Gemini response shape: {json.dumps(data)[:400]}") from exc
        if finish not in ("STOP", "MAX_TOKENS", ""):
            raise AIError(f"Gemini stopped with finishReason={finish!r}")
        text = "".join(str(part.get("text", "")) for part in parts_out)
        usage = data.get("usageMetadata") or {}
        return AiAnswer(
            text=_tidy(text),
            model=data.get("modelVersion", model),
            provider="gemini",
            prompt_tokens=int(usage.get("promptTokenCount", 0) or 0),
            completion_tokens=int(usage.get("candidatesTokenCount", 0) or 0),
            raw=data,
        )

    # --------------------------------------------------------------- anthropic
    def _ask_anthropic(
        self, api_key: str, system: str, user: str, ocr: OcrResult, image: Any, vision: bool
    ) -> AiAnswer:
        blocks: List[Dict[str, Any]] = []
        if vision:
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_to_base64_png(image),
                    },
                }
            )
        blocks.append({"type": "text", "text": user})
        body = {
            "model": self.cfg.get_path("ai.model", "claude-3-5-haiku-latest"),
            "max_tokens": int(self.cfg.get_path("ai.max_tokens", 800)),
            "temperature": float(self.cfg.get_path("ai.temperature", 0.2)),
            "system": system,
            "messages": [{"role": "user", "content": blocks}],
        }
        base = self._base_url().replace("/v1", "")
        data = self._post(f"{base}/v1/messages", self._headers(api_key), body)
        try:
            text = "".join(part.get("text", "") for part in data["content"])
        except (KeyError, TypeError) as exc:
            raise AIError(f"Unexpected response shape: {json.dumps(data)[:400]}") from exc
        usage = data.get("usage", {}) or {}
        return AiAnswer(
            text=_tidy(text),
            model=data.get("model", body["model"]),
            provider="anthropic",
            prompt_tokens=int(usage.get("input_tokens", 0) or 0),
            completion_tokens=int(usage.get("output_tokens", 0) or 0),
            raw=data,
        )

    # -------------------------------------------------------------------- http
    def _post(self, url: str, headers: Dict[str, str], body: Dict[str, Any]) -> Dict[str, Any]:
        timeout = float(self.cfg.get_path("ai.timeout", 60))
        attempts = int(self.cfg.get_path("ai.retries", 2))
        last_error: Optional[str] = None
        for attempt in range(attempts + 1):
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=timeout)
            except requests.RequestException as exc:
                last_error = f"Network error calling {url}: {exc}"
                log.warning("%s (attempt %d)", last_error, attempt + 1)
            else:
                if resp.status_code >= 400:
                    snippet = resp.text[:500]
                    if resp.status_code in (429, 500, 502, 503) and attempt < attempts:
                        delay = 1.5 * (attempt + 1)
                        log.warning("HTTP %d, retrying in %.1fs", resp.status_code, delay)
                        import time

                        time.sleep(delay)
                        continue
                    raise AIError(f"HTTP {resp.status_code} from AI API: {snippet}")
                try:
                    return resp.json()
                except ValueError as exc:
                    raise AIError(f"AI API returned non-JSON body: {resp.text[:300]}") from exc
        raise AIError(last_error or "AI request failed")


def _tidy(text: str) -> str:
    """Trim code fences/markdown noise that would look wrong when typed on screen."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
    cleaned = re.sub(r"\n```$", "", cleaned)
    cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", cleaned)
    return cleaned.strip()


def vision_transcribe_request(system_hint: str, image: Any) -> tuple:
    """Prompt pair used when the vision model does the OCR itself (no Tesseract)."""
    user = (
        "Transcribe exactly what is written in this screenshot region. "
        "Output only the text, preserving line breaks. If there is no readable "
        f"text, output {NO_TEXT_MARKER}."
    )
    return system_hint, user


NO_TEXT_MARKER = "<<NO_TEXT>>"
