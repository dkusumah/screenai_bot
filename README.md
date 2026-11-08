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
python -m screenai_bot          # desktop GUI (control panel + region selector)
python -m screenai_bot.cli --help   # headless / scripted mode
```

### GUI workflow

1. Click **“1. Select area to READ”** and drag a rectangle over the text.
2. Pick a **task**: *answer*, *explain*, *summarize*, *translate*, *suggest input*, or *custom*.
3. Optionally click **“2. Select area to WRITE”** and drag over the input box where the answer should go.
4. Choose the **action mode** (click-in-area + clear + type, double-click, type at cursor, Enter after typing, clipboard only…).
5. Press **Run**. Preview of the exact steps is shown for confirmation first (safety).

Global hotkeys (configurable): `Ctrl+Alt+R` select region · `Ctrl+Alt+Enter` run ·
`Ctrl+Alt+S` stop · `Ctrl+Alt+V` paste last answer · `Ctrl+Alt+L` repeat · `Ctrl+Alt+H` toggle outlines.

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
```

## Configuration

The live config lives at `~/.screenai_bot/config.json` (override dir with `SCREENAI_HOME`).
Key sections:

| Section | What it controls |
|---|---|
| `ai` | provider (`openai`/`anthropic`), `base_url`, `model`, `api_key_env`, `use_vision` (send the screenshot to a vision model instead of/in addition to Tesseract) |
| `ocr` | tesseract path, language, page-segmentation mode, preprocessing, min confidence |
| `actions` | typing speed, focus click, clear-field shortcut, `confirm_before_action`, failsafe |
| `hotkeys` | global shortcuts |
| `prompt` | system prompt + task templates (`answer`, `explain`, `summarize`, `translate`, `suggest_input`, `custom`) |
| `output` | delivery mode, saved regions, history file (`history.jsonl`) |

## Project layout

```
screenai_bot/
├── __init__.py / __main__.py   entry points (GUI; forwards flags to CLI)
├── config.py / config.json     typed settings + defaults
├── capture.py                  screen capture (mss), monitors, coordinates
├── ocr.py                      Tesseract wrapper + image preprocessing
├── ai_client.py                REST client for OpenAI-compatible & Anthropic APIs
├── actions.py                  click / double-click / type / paste / hotkey simulation
├── hotkeys.py                  global hotkey manager (pynput)
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
