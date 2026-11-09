# Changelog

All notable changes to ScreenAI Bot are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-11-09

A big quality-of-life release: the bot can now live in the menu bar, and both the asking and
the answering side became much more transparent (word limits, grounding level, answer formats,
audible debug cues and a hotkey diagnostics panel).

### Added

* **Answer shaping options** in the *Options* row
  * **Max Answer Words** / **Min Answer Words** (`0` = no limit / no minimum). A cap is applied
    at the nearest full stop and never below the minimum; a reply that is too short is retried
    once (`ai.length_retries`).
  * **AI Hallucination** level `0` Strict → `4` Unrestricted (`ai.hallucination_level`), each
    level adding its own editable grounding rule (`prompt.hallucination_levels`).
  * **Answer Like A Human** (`ai.natural_answers`) appends a chat-style directive.
  * **Allow In Answer**: Markdown, HTML, Code, Bullet Lists (`ai.allow_*`). When Markdown or
    Code is allowed, the answer keeps its bold markers and code fences instead of being cleaned.
* **Background mode (menu bar / tray)**: closing the panel hides it to a status icon instead of
  quitting — *Open Panel · Run Now · Stop · Select Area To Read · Paste Last Answer ·
  Show/Hide Outlines · Test Hotkeys · Restart Hotkeys · Quit*. On macOS the Dock icon is hidden
  while the panel is closed. `python -m screenai_bot --background` starts hidden.
* **Hotkey diagnostics** (*Options → Test Hotkeys*, also in the tray menu): live key counters,
  macOS permission state, the bound shortcuts rendered the Mac way (`⌃⌥↩`) and one-click
  buttons for the Accessibility / Input Monitoring panes plus **Restart Hotkeys**.
* **Debug sounds** (off by default, `debug.sounds`): audible cues for *hotkey fired*, *key that
  matched no shortcut*, *mouse click*, *typing*, *run started/finished*, *error* and *cancelled*,
  with per-cue switches, a volume setting and a **Test Sounds** button.
* **Self-healing hotkeys**: a 3 s watchdog rebuilds the listener when macOS silently disables
  the event tap (bounded by `app.hotkey_max_restarts`, default 2).
* **Log file** at `~/.screenai_bot/screenai.log` (rotating), so a windowless app can be
  diagnosed afterwards.
* `pipeline.hold_panel_until_done()`: with the `answer` task and a type-into-a-field action mode
  the panel now stays hidden from the first click until the typing finished.

### Changed

* The answer length limit is counted in **words**, not characters, and the output-token budget
  is derived from it (`max_tokens` is a floor, no longer a hard 800 cap).
* Truncated model output is detected (`finish_reason=length`, `stop_reason=max_tokens`,
  `MAX_TOKENS`) and retried once with a doubled budget; if it is still incomplete you get an
  explicit warning instead of a silently chopped answer.
* Hotkeys fire on key **press** (not release) and are dispatched on the GUI thread; the in-app
  Qt shortcuts are always bound as well, de-duplicated against the global ones.
* `cli.py`: `--max-answer-words` / `--min-answer-words` replace `--max-answer-chars`, plus
  `--markdown`, `--html`, `--code`, `--bullets` and their `--no-*` forms. `--json` reports
  `answer_words`, truncation and the active format switches.
* `actions.hide_until_writing_done` (default on) with the task/mode lists and a longer writing
  watchdog (`auto_hide_writing_max_ms`).
* macOS-specific frameworks (Quartz/AppKit/ApplicationServices) are declared explicitly as an
  optional `macos` extra.

### Fixed

* **Hotkeys did nothing on macOS** when the panel was not focused:
  * Option composes characters (`Option+R` → `®`), so `ctrl+alt+r` could never match — the
    physical keycode is now used.
  * Return/Tab/Esc/Backspace arriving as control characters (`\r` was folded into the letter
    `m`), and keypad Enter (`\x03`, same code as Ctrl+C) now resolve correctly.
  * A stuck `cmd`/`tab` after `Cmd+Tab` blocked every shortcut: matching now uses
    "key just pressed + modifiers held", and the state resets on application activation.
  * Handlers ran on the `pynput` listener thread, touching Qt widgets off the GUI thread.
  * `pynput` starts even when macOS denies Input Monitoring, so the app claimed "hotkeys
    active" while receiving nothing — permission problems are now reported, and the in-app
    shortcuts are always available.
* **`zsh: trace trap` crash** when the watchdog rebuilt the listener: the rebuild now runs on a
  worker thread, joins the previous listener before creating a new one, is bounded and never
  retries a tap that never delivered a key.
* **Answers cut in half**: the old fixed `max_tokens=800` plus a 150-word default cap were both
  removed; trimming is sentence-aware and reported.
* Closing the window no longer kills the bot (and stopping it is now an explicit *Quit*).
* Pressing *Run* from the background with no area selected pops the panel forward instead of
  appearing to do nothing.
* `QSystemTrayIcon` clicks on macOS open the menu; only the menu item or a double click
  reopens the panel.


### Upgrade notes

* `ai.max_answer_chars` is converted automatically to `ai.max_answer_words` (÷6) on first load
  and the retired key is dropped; a `{{MAX_CHARS}}` sentence in `prompt.system` is replaced by
  the word-based rule. Nothing else needs to be edited by hand — missing keys fall back to the
  built-in defaults.
* Default `ai.max_answer_words` is `0` (no cap) so long explanations arrive complete.
* The default shortcut is still `ctrl+alt+enter`, i.e. **Control + Option + Return** on a Mac
  (Command is a different modifier). Use `cmd+alt+enter` in `hotkeys` if you prefer Command.

### Verification

Behaviour was checked against the real macOS window server and a mock HTTP endpoint: five
hotkey combinations fire from real key events (also while Safari/TextEdit are frontmost), the
Option-composition and keypad-Enter cases, stuck-key/Cmd+Tab immunity, async+bounded listener
restart (no `SIGTRAP`) and recovery of hotkeys afterwards, all eight system sounds, tray
close-to-tray/quit, prompt composition for the new options, sentence-aware trimming, and one
retry each for truncated or too-short answers.

## [1.0.0] - 2026-11-08

* First public version: select a screen region, OCR it, ask an OpenAI-compatible or Anthropic
  model, then click/type the answer back (or copy it), with global hotkeys, an outline overlay
  and a headless CLI.

[1.1.0]: https://github.com/dkusumah/screenai_bot/compare/9747794...HEAD
[1.0.0]: https://github.com/dkusumah/screenai_bot/releases/tag/v1.0.0

