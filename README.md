# ScreenAI Bot 🖥️🤖

A **desktop automation bot** that:

1. **Reads text from a screen area you select** (fast `mss` capture + Tesseract OCR, or a vision LLM),
2. **Sends that text to an AI API** (any OpenAI-compatible endpoint — OpenAI, Ollama, LM Studio, OpenRouter, Groq, vLLM — or Anthropic),
3. **Receives the answer / explanation**, and
4. **Clicks and types the answer back inside the chosen screen area** (or copies it to the clipboard).

Works on **Windows, macOS and Linux** (Python ≥ 3.10).

---

## Install

```bash
# 1) Python packages
pip install -r requirements.txt

# 2) Tesseract OCR binary (pick yours)
#    Windows : https://github.com/UB-Mannheim/tesseract/wiki  (installer)
#    macOS   : brew install tesseract
#    Debian  : sudo apt install tesseract-ocr
#    Fedora  : sudo dnf install tesseract-ocr

# 3) API key
export OPENAI_API_KEY=sk-...        # or set ai.api_key in the config file
```

Optional: install as a command (`screenai-bot`, `screenai-bot-cli`):

```bash
pip install .
```

## Run

```bash
python -m screenai_bot                # desktop GUI (control panel + region selector)
python -m screenai_bot --background   # start straight into the menu bar / tray (no window)
python -m screenai_bot.cli --help      # headless / scripted mode
```

### Background mode (like Docker Dashboard)

The bot is meant to live in the background, so **closing the panel does not quit it**:

* The window drops to a **menu-bar item** (macOS) / **tray icon** (Windows, Linux) with
  *Open Panel · Run Now · Stop · Select Area To Read · Paste Last Answer · Show/Hide Outlines ·
  Test Hotkeys · Restart Hotkeys · Quit ScreenAI Bot*.
* **Click the icon** (or press a hotkey) to bring the panel back; **Quit** in that menu is
  the only thing that really exits.
* On macOS the Dock icon is hidden while the panel is closed, so the bot behaves exactly
  like a menu-bar app.
* Everything keeps working while hidden: global hotkeys, capture/OCR/AI/typing, and the
  panel stays out of your way (it never pops back up on its own after you closed it).
* Logs go to `~/.screenai_bot/screenai.log` (rotating), since a tray app has no console.

Turn this off with the **Keep Running In Background** checkbox in *Options*
(`app.run_in_background`), or start hidden with `app.start_minimized` / `--background`.

### Global hotkeys (and the macOS permission they need)

`Ctrl+Alt+R` select region · `Ctrl+Alt+Enter` run · `Ctrl+Alt+S` stop ·
`Ctrl+Alt+V` paste last answer · `Ctrl+Alt+L` repeat · `Ctrl+Alt+H` toggle outlines.

**On a Mac these names mean different keys than on Windows** — that is the usual reason a
shortcut "doesn't work" on an Apple keyboard:

| Config name | Mac key | Windows / Linux key |
|---|---|---|
| `ctrl` | ⌃ **Control** | Ctrl |
| `alt` | ⌥ **Option** ← this is what "Alt" means on macOS | Alt |
| `cmd` | ⌘ **Command** (not a substitute for `ctrl`) | — (Win / Super) |
| `enter` | ↩ **Return** | Enter |
| `shift` | ⇧ Shift | Shift |

So the default run shortcut is **hold ⌃ Control + ⌥ Option, then press ↩ Return**.
Holding ⌘ Command instead of Option is a *different* shortcut (`cmd+ctrl+enter`) and will
not fire — `cmd` is its own modifier, it does not stand in for `alt`.

The app shows the bindings in that notation everywhere (`⌃⌥↩`), and *Options → Test
Hotkeys* lists every bound shortcut with the same legend, so there is nothing to guess.

Prefer the Command key? Edit `hotkeys` in `config.json` (all names in the table work):

```json
"hotkeys": {
  "select_region": "cmd+alt+r",
  "run": "cmd+alt+enter",
  "stop": "cmd+alt+s"
}
```

Global hotkeys work with **no window on screen**. If nothing happens on macOS:

1. Open **Options → Test Hotkeys** (also in the tray menu). It shows how many key presses
   actually reached the bot and whether the OS is blocking them.
2. Grant **Input Monitoring** (and **Accessibility**) to the app that launches the bot —
   Terminal, iTerm, VS Code — in *System Settings → Privacy & Security*. The dialog has
   buttons that jump straight to those panes.
3. Restart the bot. Keys pressed while the panel has focus always work through Qt
   shortcuts, even when the OS permission is missing.

Notes on how this is implemented (all three were real bugs):

* Hotkeys fire on key **press** (not release) and the handler hops onto the GUI thread —
  Qt widgets were previously touched from the listener thread, which could silently drop
  the action.
* On macOS, holding Option *composes* the character (`Option+R` = `®`, `Option+S` = `ß`),
  so `ctrl+alt+r` could never match. The physical key code is used instead.
* A shortcut now matches on **"the key just pressed + the modifiers currently held"**.
  Switching apps with `Cmd+Tab` used to lose the key-release events and leave `cmd`/`tab`
  stuck in the pressed set, which silently blocked *every* shortcut until a restart.

**Self-healing (and what the log warning means):** macOS disables an event tap without
telling anyone (after heavy input, sleep/wake, or a secure-input dialog) and `pynput` never
re-enables it, so the hotkeys used to die quietly. Every 3 s the bot compares the OS's own
last-key time with its event counter; when they disagree it logs

```
WARNING screenai.hotkeys: Restarting the global hotkey listener (no events while the OS saw key presses, attempt 1/2)
```

…and rebuilds the listener **on a worker thread**, stopping and *joining* the old one first —
tearing a CoreFoundation listener down inline is what caused `zsh: trace trap` crashes. The
rebuild is bounded by `app.hotkey_max_restarts` (default `2`); after that the bot stops
retrying, says *"Automatic recovery paused"* and waits for **Restart Hotkeys** (tray menu or
*Test Hotkeys*), which is unlimited. `0` disables automatic recovery completely.

A listener that has **never** delivered a single key is not rebuilt at all: that is a
permission or secure-input problem, where restarting cannot help and would only thrash.

**When a shortcut genuinely cannot work:** another app can take the keyboard away from
every monitor on the system. This happens with *secure keyboard entry* (password fields,
banking apps, 1Password) and with kiosk/lockdown browsers such as **Safe Exam Browser** -
by design, no automation tool can read keys there. The bot detects it (it will report
"Another application is blocking key monitoring") and the tray menu keeps working in the
meantime.

### GUI workflow

1. Click **“1. Select area to READ”** and drag a rectangle over the text.
2. Pick a **task**: *answer*, *explain*, *summarize*, *translate*, *suggest input*, or *custom*.
3. Optionally click **“2. Select area to WRITE”** and drag over the input box where the answer should go.
4. Choose the **action mode** (click-in-area + clear + type, double-click, type at cursor, Enter after typing, clipboard only…).
5. Shape the answer in the **Options** row: *Max/Min Answer Words*, *Allow In Answer* (Markdown, HTML, Code, Bullet Lists), *Answer Like A Human*, *AI Hallucination*, *Keep Hidden Until Typing Ends* and *Keep Running In Background* (see below).
6. Press **Run**. Preview of the exact steps is shown for confirmation first (safety).

**Emergency stop:** move the mouse to a screen corner (pyautogui failsafe) or press the Stop hotkey.

### CLI examples

```bash
# OCR a region, ask the AI, print the answer (never clicks/types)
python -m screenai_bot.cli --region 100,100,600,200 --task answer --dry-run

# Type the answer into another region (click -> clear -> type)
python -m screenai_bot.cli --region 100,100,600,200 \
    --target 800,400,300,40 --mode click_and_type --yes

# Use local Ollama instead of OpenAI
python -m screenai_bot.cli --text "What is 6*7?" --task answer \
    --base-url http://localhost:11434/v1 --model llama3

# Just see what OCR reads
python -m screenai_bot.cli --region 100,100,600,200 --ocr-only --json

# Answer like a person, only from what is on screen, 40 words max
python -m screenai_bot.cli --region 100,100,600,200 --task answer \
    --hallucination 0 --natural --max-words 40 --dry-run

# Ask for 60-120 words, allow Markdown and bullet lists, no code blocks
python -m screenai_bot.cli --text "Summarize the attached notes" \
    --min-words 60 --max-words 120 --markdown --bullets --no-code
```

## Configuration

The live config lives at `~/.screenai_bot/config.json` (override dir with `SCREENAI_HOME`).
Key sections:

| Section | What it controls |
|---|---|
| `ai` | provider (`openai`/`anthropic`/`gemini`), `base_url`, `model`, `api_key_env`, `use_vision`, `max_tokens` (raised automatically to fit the word limits), `length_retries`, plus the answer-shaping keys `max_answer_words`, `min_answer_words`, `hallucination_level`, `natural_answers` and the `allow_*` format switches (see below) |
| `ocr` | tesseract path, language, page-segmentation mode, preprocessing, min confidence |
| `actions` | typing speed, focus click, clear-field shortcut, `confirm_before_action`, failsafe, auto-hide (`auto_hide_window`, `hide_until_writing_done` + its task/mode lists) |
| `hotkeys` | global shortcuts (work with the panel closed) |
| `app` | `run_in_background` (close to the menu bar), `start_minimized` (start with no window), `hide_dock_icon` (macOS menu-bar-only), `tray_notifications`, `hotkey_max_restarts` (automatic listener rebuilds, `0` = never) |
| `debug` | audible feedback: `sounds` (master switch), `sound_volume`, and `sound_*` per-cue toggles |
| `prompt` | system prompt + task templates (`answer`, `choose_option`, `explain`, `summarize`, `translate`, `suggest_input`, `custom`) + `hallucination_levels`, `natural_style`, `length_rules` and the `format_*` wording |
| `output` | delivery mode, saved regions, history file (`history.jsonl`) |

### Answer shaping (Options row)

| Option | Config key | What it does |
|---|---|---|
| **Max Answer Words** | `ai.max_answer_words` | Longest answer the bot may type/paste, in words. **`0` = No limit (default)**, so long explanations arrive complete. A cap trims on the nearest full stop (never mid-sentence when avoidable) and appends `...`. |
| **Min Answer Words** | `ai.min_answer_words` | Shortest answer asked of the model (`0` = **No minimum**). Adds “Write at least N words.” to the prompt **and** retries once when the reply comes back shorter (`ai.length_retries`, default `1`); a still-short reply is used but flagged in the log. Never overrides the maximum — the two spinners keep each other consistent. |
| **Answer Like A Human** | `ai.natural_answers` | Appends `prompt.natural_style` to the system prompt: chat-style wording and contractions, no “The answer is” preamble. |
| **Allow In Answer** | `ai.allow_markdown`, `ai.allow_html`, `ai.allow_code`, `ai.allow_bullet_lists` | Each checkbox adds either “You may use …” or “Do not use …” to the prompt (`prompt.format_allow` / `prompt.format_deny` feeding `prompt.format_features`). All off = plain text, which is what usually fits an on-screen field. With **Markdown**/**Code** enabled the answer keeps its bold markers and code fences instead of being cleaned. |
| **AI Hallucination** | `ai.hallucination_level` | `0` Strict – answers only from the captured text and says so when the screen doesn't show it; `1` Careful; `2` Balanced (default); `3` Creative – fills small gaps; `4` Unrestricted – may invent plausible details. Each level adds its own grounding line from `prompt.hallucination_levels`. |

#### If an answer stops halfway

Two different things can shorten a reply, and both are now handled:

1. **The provider ran out of output tokens** (`finish_reason=length`,
   `stop_reason=max_tokens`, `MAX_TOKENS`). The request budget is no longer a fixed
   800: it is raised automatically to fit the word limits (~1.8 tokens per word), and
   a truncated reply is retried once with double the budget. If it is *still*
   truncated you get a clear warning in the Activity log / stderr.
   You can also raise `ai.max_tokens` yourself.
2. **Your own word cap** (`ai.max_answer_words`). It defaults to `0` (no cap); when
   you do set one, the cut lands on a sentence boundary and the log says
   “Answer trimmed to N words”.

Length limits are counted in real words: list markers such as `-` or `*` are ignored, so a
5-word cap means five actual words. All directives are plain strings in `config.json`,
so you can reword them without touching the code.
Use “Preview Prompt” in the GUI to see exactly what the model receives.

> Upgrading from the older options: `ai.max_answer_chars` is converted to
> `ai.max_answer_words` automatically (÷6) the first time the config is loaded, and any
> `{{MAX_CHARS}}` sentence in `prompt.system` is replaced by the word-based rule.

### Debug feedback (sounds)

Tick **Debug Sounds** in *Options* (`debug.sounds`) and the bot tells you out loud what it
is doing — handy to answer "did the hotkey actually fire?" and "did it really click?"
without watching the log:

| Cue | macOS sound | What it confirms |
|---|---|---|
| `hotkey` | Tink | a shortcut fired |
| `hotkey_miss` | Pop | a key press matched **no** shortcut (off by default — noisy) |
| `click` | Morse | the bot is clicking the mouse right now |
| `type` | Tink | the bot is about to type/paste |
| `start` / `done` | Ping / Glass | a run started / finished |
| `error` | Basso | something failed or was blocked (e.g. no area selected) |
| `cancel` | Bottle | you declined the confirmation dialog |

* **Test Sounds** plays hotkey → click → done → error so you can judge the volume (it
  switches audio on temporarily and restores your setting).
* Everything is opt-in and per-cue: `debug.sounds` is the master switch, and
  `debug.sound_hotkey`, `debug.sound_hotkey_miss`, `debug.sound_click`, `debug.sound_type`,
  `debug.sound_start`, `debug.sound_done`, `debug.sound_error`, `debug.sound_cancel` select
  the individual cues. `debug.sound_volume` (0–1, default `0.5`) sets the level.
* Playback is best-effort: no audio device, no PyObjC or no permission simply means silence,
  never an error in the pipeline.

### Staying out of the way while the answer is typed

With the `answer` task and either **Click Inside Area, Clear It, Type Answer** or
**Click Target Once, Then Type**, the control panel hides on the first click and only
comes back once the *whole* answer has been typed (plus `actions.restore_delay_ms`),
so it never covers the field the bot is writing into.

| Config key | Meaning |
|---|---|
| `actions.hide_until_writing_done` | `true` (default) keeps the panel hidden for the entire write action |
| `actions.hide_until_writing_tasks` | task keys it applies to — default `["answer"]` |
| `actions.hide_until_writing_modes` | action modes it applies to — default `["type_in_region", "click_and_type"]` |
| `actions.auto_hide_writing_max_ms` | watchdog for very long typings (default `60000`); the panel is forced back if a run hangs |

Toggle it with **Keep Hidden Until Typing Ends** in the Options row. It requires
`actions.auto_hide_window` to stay on; other action modes (MCQ click, type at cursor,
copy-only) keep the older “hide only while the click happens” behaviour.

## Project layout

```
screenai_bot/
├── __init__.py / __main__.py   entry points (GUI; forwards flags to CLI)
├── config.py / config.json     typed settings + defaults
├── capture.py                  screen capture (mss), monitors, coordinates
├── ocr.py                      Tesseract wrapper + image preprocessing
├── ai_client.py                REST client for OpenAI-compatible & Anthropic APIs
├── actions.py                  click / double-click / type / paste / hotkey simulation
├── hotkeys.py                  global hotkey manager (pynput, press-based matching)
├── native.py                   macOS permissions, privacy panes, Dock icon policy
├── sound.py                    audible debug cues (hotkey / click / error ...)
├── tray.py                     menu-bar / system-tray icon + menu (background mode)
├── overlay.py                  drag-to-select region + outline marker (Qt)
├── pipeline.py                 capture → OCR → AI → action worker (QThread)
├── ui.py                       main control window
└── cli.py                      headless runner
```

## Notes & safety

- Actions are **previewed and confirmed by default** (`actions.confirm_before_action`).
- Unicode answers are delivered via clipboard **paste** automatically (typing can't send them reliably).
- On Linux, run under an X11/Wayland session with permissions for input simulation; on Windows the `keyboard` backend needs admin rights (optional fallback).
- Low OCR confidence is surfaced in the UI and passed to the model so it can flag uncertainty.
