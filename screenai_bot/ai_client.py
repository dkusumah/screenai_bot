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

from .config import FORMAT_FEATURES, Config, Region
from .ocr import OcrResult, image_to_base64_png

log = logging.getLogger("screenai.ai")

MAX_ANSWER_WORDS_DEFAULT = 150


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
    # Why the provider stopped ("stop", "length", "max_tokens", "STOP", ...).
    finish_reason: str = ""
    # True when the provider ran out of output tokens, so the text may stop
    # mid-sentence. ``short`` is set when fewer words came back than
    # ``ai.min_answer_words`` asked for.
    truncated: bool = False
    short: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)

    def one_liner(self) -> str:
        flat = " ".join(self.text.split())
        return flat if len(flat) <= 120 else flat[:117] + "..."


TASK_KEYS = ("answer", "explain", "summarize", "translate", "suggest_input", "custom")

# A "word" for length purposes: any whitespace-separated token that contains at
# least one letter/digit, so list bullets ("-", "*") are not counted or cut into.
_WORD_TOKEN = re.compile(r"\S*[^\W_]\S*", re.UNICODE)

# Sentence terminator followed by whitespace/end, used to trim on a full stop.
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


def build_prompt(
    cfg: Config,
    ocr: OcrResult,
    task_key: str,
    region: Optional[Region],
    apply_limits: bool = True,
) -> tuple:
    """Render (system, user) messages from config templates.

    The system prompt is built from ``prompt.system`` plus the directives that
    implement the answer-shaping options: word limits (``ai.min_answer_words`` /
    ``ai.max_answer_words``), ``ai.hallucination_level`` (how far the model may go
    beyond the captured text), ``ai.natural_answers`` (write like a human) and the
    formatting switches (markdown / html / code / bullet lists).

    ``apply_limits=False`` leaves the word-limit rules out; it is used by the
    vision transcription calls, where an answer-length rule makes no sense.
    """
    tasks: Dict[str, str] = dict(cfg.get_path("prompt.tasks", {}) or {})
    template_task = tasks.get(task_key) or tasks.get("answer") or "Answer the question in the text."

    task_text = (
        template_task.replace("{{TARGET_LANG}}", str(cfg.get_path("prompt.target_lang", "English")))
        .replace("{{CUSTOM}}", str(cfg.get_path("prompt.custom_instruction", "")))
    )

    system = _render_system(cfg, apply_limits=apply_limits)
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


def _render_system(cfg: Config, apply_limits: bool = True) -> str:
    """System prompt: template + length rules + style/format directives."""
    template = str(cfg.get_path("prompt.system", ""))
    max_words = max_answer_words(cfg)
    min_words = min_answer_words(cfg)
    if "{{MAX_CHARS}}" in template:
        # Older templates hard-coded a character budget. Drop that sentence: the
        # length rules below now come from the word-based options.
        template = re.sub(r"[^.\n]*\{\{MAX_CHARS\}\}[^.\n]*[.\n]?", "", template)
        template = template.replace("{{MAX_CHARS}}", "")

    own_length_rule = "{{MAX_WORDS}}" in template or "{{MIN_WORDS}}" in template
    system = template
    if "{{MAX_WORDS}}" in system:
        system = system.replace("{{MAX_WORDS}}", str(max_words) if max_words > 0 else "")
    if "{{MIN_WORDS}}" in system:
        system = system.replace("{{MIN_WORDS}}", str(min_words) if min_words > 0 else "")
    system = system.strip()

    parts = [system] if system else []
    if not own_length_rule and apply_limits:
        parts.extend(length_rules(cfg, max_words, min_words))
    parts.extend(style_directives(cfg))
    parts.extend(format_directives(cfg))
    return "\n\n".join(part for part in parts if part)


def length_rules(cfg: Config, max_words: int, min_words: int) -> List[str]:
    """'at least N words' / 'within N words' lines from ``prompt.length_rules``."""
    rules = dict(cfg.get_path("prompt.length_rules", {}) or {})
    out: List[str] = []
    if min_words > 0:
        rule = str(rules.get("min", "")).replace("{{MIN_WORDS}}", str(min_words)).strip()
        if rule:
            out.append(rule)
    if max_words > 0:
        rule = str(rules.get("max", "")).replace("{{MAX_WORDS}}", str(max_words)).strip()
        if rule:
            out.append(rule)
    return out


def format_directives(cfg: Config) -> List[str]:
    """One allow/deny line per formatting feature (markdown, html, code, lists)."""
    features = dict(cfg.get_path("prompt.format_features", {}) or {})
    allow = str(cfg.get_path(
        "prompt.format_allow", "You may use {feature} when it helps."
    ))
    deny = str(cfg.get_path("prompt.format_deny", "Do not use {feature}."))
    out: List[str] = []
    for feature, cfg_key in FORMAT_FEATURES:
        label = str(features.get(feature) or feature.replace("_", " ")).strip()
        template = allow if bool(cfg.get_path(f"ai.{cfg_key}", False)) else deny
        line = template.replace("{feature}", label).strip()
        if line:
            out.append(line)
    return out


def style_directives(cfg: Config) -> List[str]:
    """Prompt lines contributed by the hallucination level and human-style option."""
    out: List[str] = []
    levels = list(cfg.get_path("prompt.hallucination_levels", []) or [])
    if levels:
        level = int(cfg.get_path("ai.hallucination_level", 2) or 0)
        level = max(0, min(level, len(levels) - 1))
        directive = str(levels[level]).strip()
        if directive:
            out.append(directive)
    if bool(cfg.get_path("ai.natural_answers", False)):
        natural = str(cfg.get_path("prompt.natural_style", "")).strip()
        if natural:
            out.append(natural)
    return out


def _int_setting(cfg: Config, dotted: str, default: int) -> int:
    """Read an integer setting, tolerating blank/legacy values."""
    try:
        return int(cfg.get_path(dotted, default) or 0)
    except (TypeError, ValueError):
        return default


def max_answer_words(cfg: Config) -> int:
    """Configured answer-length cap in words; ``<= 0`` means no limit."""
    return _int_setting(cfg, "ai.max_answer_words", MAX_ANSWER_WORDS_DEFAULT)


def min_answer_words(cfg: Config) -> int:
    """Configured minimum answer length in words; ``<= 0`` means no minimum."""
    return _int_setting(cfg, "ai.min_answer_words", 0)


def count_words(text: str) -> int:
    """Number of real words in ``text`` (list markers like "-" or "*" do not count)."""
    return len(_WORD_TOKEN.findall(text))


def cap_answer(text: str, max_words: int, min_words: int = 0) -> str:
    """Trim ``text`` to ``max_words`` words; ``<= 0`` means no limit.

    The cut is made at a sentence end when one exists reasonably close to the
    limit (so the answer still reads as finished), otherwise at a word boundary.
    Newlines and list markers inside the kept part survive, and the result is
    never shorter than ``min_words``.
    """
    if max_words <= 0:
        return text
    limit = max(int(max_words), int(min_words or 0))
    matches = list(_WORD_TOKEN.finditer(text))
    if len(matches) <= limit:
        return text
    kept = text[: matches[limit - 1].end()].rstrip()
    # Prefer the last complete sentence, as long as it keeps most of the budget.
    ends = [match.end() for match in _SENTENCE_END.finditer(kept)]
    if ends:
        candidate = kept[: ends[-1]].rstrip()
        if len(_WORD_TOKEN.findall(candidate)) >= max(1, int(limit * 0.6)):
            kept = candidate
    kept = kept.rstrip()
    if kept.endswith("."):
        kept = kept[:-1].rstrip()   # avoid "sentence...." when the cut is clean
    return kept + "..."


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
    def _max_tokens(self, scale: float = 1.0) -> int:
        """Output-token budget, sized so the word limits are actually reachable.

        A fixed ``ai.max_tokens`` (default 800) used to cut long explanations off
        mid-sentence even when the word limit allowed more, so the configured
        value is treated as a floor and grown to fit ``ai.max_answer_words`` /
        ``ai.min_answer_words`` (~1.8 tokens per word plus slack).
        """
        budget = max(64, int(self.cfg.get_path("ai.max_tokens", 800) or 800))
        target_words = max(max_answer_words(self.cfg), min_answer_words(self.cfg))
        if target_words > 0:
            budget = max(budget, int(target_words * 1.8) + 160)
        return max(64, int(budget * max(0.1, float(scale))))

    def _dispatch(
        self,
        api_key: str,
        system: str,
        user: str,
        image: Any,
        vision: bool,
        max_tokens: Optional[int] = None,
    ) -> AiAnswer:
        """Send one request to the configured provider."""
        if self.provider == "anthropic":
            return self._ask_anthropic(api_key, system, user, image, vision, max_tokens)
        if self.provider == "gemini":
            return self._ask_gemini(api_key, system, user, image, vision, max_tokens)
        return self._ask_openai_compat(api_key, system, user, image, vision, max_tokens)

    def ask(
        self,
        ocr: OcrResult,
        task_key: str = "answer",
        image: Any = None,
        region: Optional[Region] = None,
        extra_context: str = "",
        on_progress: Optional[Callable[[str], None]] = None,
        apply_limits: bool = True,
    ) -> AiAnswer:
        api_key = self.cfg.api_key
        if not api_key:
            raise AIError(
                "No API key configured. Set ai.api_key in ~/.screenai_bot/config.json "
                "or export the environment variable named by ai.api_key_env."
            )
        system, user = build_prompt(self.cfg, ocr, task_key, region, apply_limits=apply_limits)
        if extra_context:
            user = f"{user}\n\nAdditional context: {extra_context}"
        use_vision = bool(self.cfg.get_path("ai.use_vision", False)) and image is not None

        if on_progress:
            on_progress("Sending to AI...")
        payload_desc = f"{self.provider} vision" if use_vision else self.provider
        log.info(
            "AI request via %s (%d chars of OCR text, max_tokens=%d)",
            payload_desc, len(user), self._max_tokens(),
        )

        answer = self._dispatch(api_key, system, user, image, use_vision)
        if apply_limits:
            answer = self._retry_for_length(
                answer, api_key, system, user, image, use_vision, on_progress
            )
            min_words = min_answer_words(self.cfg)
            answer.short = min_words > 0 and count_words(answer.text) < min_words
        if on_progress:
            on_progress("AI answered.")
        return answer

    def _retry_for_length(
        self,
        answer: AiAnswer,
        api_key: str,
        system: str,
        user: str,
        image: Any,
        vision: bool,
        on_progress: Optional[Callable[[str], None]],
    ) -> AiAnswer:
        """Retry when the model stopped early or ignored the word minimum.

        The provider's token limit is the usual reason an answer "stops in the
        middle", so a truncated reply is retried with a larger budget. A reply
        under ``ai.min_answer_words`` is retried with an explicit instruction.
        The longer of the two answers wins.
        """
        min_words = min_answer_words(self.cfg)
        attempts = max(0, int(self.cfg.get_path("ai.length_retries", 1) or 0))
        for attempt in range(1, attempts + 1):
            words = count_words(answer.text)
            too_short = min_words > 0 and words < min_words
            if not too_short and not answer.truncated:
                break
            notes: List[str] = []
            if too_short:
                notes.append(
                    f"Your previous reply was only {words} words long. Answer again and use "
                    f"at least {min_words} words, with complete sentences; do not stop early."
                )
            if answer.truncated:
                notes.append(
                    "Your previous reply was cut off before it ended (token limit). Give the "
                    "complete answer and finish every sentence."
                )
            budget = self._max_tokens(scale=2.0 ** attempt)
            reason = "too short" if too_short else "cut off"
            log.info("Retrying the AI answer (%s), max_tokens=%d", reason, budget)
            if on_progress:
                on_progress(
                    f"Answer was {reason} - retrying for a "
                    f"{'longer' if too_short else 'complete'} one..."
                )
            try:
                retry = self._dispatch(
                    api_key, "\n\n".join([system] + notes), user, image, vision, budget
                )
            except AIError as exc:
                log.warning("Length retry failed: %s", exc)
                break
            if count_words(retry.text) > words:
                answer = retry
            else:
                break
        return answer

    def _clean(self, text: str) -> str:
        """Apply :func:`_tidy` according to the enabled format switches."""
        return _tidy(
            text,
            keep_markdown=bool(self.cfg.get_path("ai.allow_markdown", False)),
            keep_code=bool(self.cfg.get_path("ai.allow_code", False)),
        )

    # -------------------------------------------------------- openai compatible
    def _ask_openai_compat(
        self,
        api_key: str,
        system: str,
        user: str,
        image: Any,
        vision: bool,
        max_tokens: Optional[int] = None,
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
            "max_tokens": int(max_tokens or self._max_tokens()),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
        }
        url = f"{self._base_url()}/chat/completions"
        data = self._post(url, self._headers(api_key), body)
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AIError(f"Unexpected response shape: {json.dumps(data)[:400]}") from exc
        finish = str(choice.get("finish_reason") or "")
        usage = data.get("usage", {}) or {}
        return AiAnswer(
            text=self._clean(text or ""),
            model=data.get("model", body["model"]),
            provider=self.provider,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            finish_reason=finish,
            truncated=finish == "length",
            raw=data,
        )

    # ------------------------------------------------------------------- gemini
    def _ask_gemini(
        self,
        api_key: str,
        system: str,
        user: str,
        image: Any,
        vision: bool,
        max_tokens: Optional[int] = None,
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
                "maxOutputTokens": int(max_tokens or self._max_tokens()),
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
            text=self._clean(text),
            model=data.get("modelVersion", model),
            provider="gemini",
            prompt_tokens=int(usage.get("promptTokenCount", 0) or 0),
            completion_tokens=int(usage.get("candidatesTokenCount", 0) or 0),
            finish_reason=str(finish or ""),
            truncated=finish == "MAX_TOKENS",
            raw=data,
        )

    # --------------------------------------------------------------- anthropic
    def _ask_anthropic(
        self,
        api_key: str,
        system: str,
        user: str,
        image: Any,
        vision: bool,
        max_tokens: Optional[int] = None,
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
            "max_tokens": int(max_tokens or self._max_tokens()),
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
        stop = str(data.get("stop_reason") or "")
        usage = data.get("usage", {}) or {}
        return AiAnswer(
            text=self._clean(text),
            model=data.get("model", body["model"]),
            provider="anthropic",
            prompt_tokens=int(usage.get("input_tokens", 0) or 0),
            completion_tokens=int(usage.get("output_tokens", 0) or 0),
            finish_reason=stop,
            truncated=stop == "max_tokens",
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


def _tidy(text: str, keep_markdown: bool = False, keep_code: bool = False) -> str:
    """Trim code fences/markdown noise that would look wrong when typed on screen.

    Fences and ``**bold**`` markers are only stripped when the matching format
    switch (``ai.allow_code`` / ``ai.allow_markdown``) is off; otherwise the
    formatting the user asked for is preserved verbatim.
    """
    cleaned = text.strip()
    if not keep_code:
        cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)
    if not keep_markdown:
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
