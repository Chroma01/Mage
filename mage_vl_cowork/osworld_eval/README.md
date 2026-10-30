# Mage-VL-Cowork — Full OSWorld-Verified Evaluation Suite

This directory provides the complete, drop-in **OSWorld-Verified** evaluation package for **Mage-VL-Cowork** (`MAGEHybridAgent`), aligned with the 66.0% evaluation pipeline (`test_nogdrive.json`, 361 tasks, 50-step budget).

## Directory Structure

```text
osworld_eval/
├── README.md
├── desktop_env/
│   └── pydeps_bundle.py             # Guest Python dependency bundle builder & installer (Evolved Harness)
├── evaluation_examples/
│   └── test_nogdrive.json           # 361 OSWorld-Verified benchmark tasks (excluding Google Drive)
├── mm_agents/
│   ├── mage_gui_agent.py            # Base GUI computer_use agent & action parser
│   └── mage_hybrid_agent.py         # Hybrid GUI (computer_use) + Bash agent & SGLang /generate client
├── run/
│   └── run_multienv_mage_cowork.py  # Multi-environment OSWorld evaluation runner (GUI + Bash routing)
└── scripts/
    ├── serve_sglang_docker.sh       # SGLang Docker inference server launcher
    └── run_osworld_eval.sh          # One-command overlay & evaluation launcher
```

## 1. Launch the SGLang Inference Runtime

We provide a pre-built, dependency-pinned GPU runtime container (`docker.io/xiaoyizhang/mage-vl-cowork:0.1.0`, SGLang `0.5.2` + PyTorch `2.8.0+cu128` + Transformers `5.7.0` + FlashInfer `0.3.1` with native `llava_onevision2` multimodal support):

```bash
export MODEL_DIR=/path/to/Mage-VL-Cowork   # local clone of microsoft/Mage-VL-Cowork
bash osworld_eval/scripts/serve_sglang_docker.sh
```

Verify that the server is healthy on `http://127.0.0.1:30001/health`:

```bash
curl --fail http://127.0.0.1:30001/health
```

> **Important (Multimodal Request Protocol):** `MAGEHybridAgent` uses Hugging Face `AutoProcessor.apply_chat_template(..., enable_thinking=True)` to render multimodal prompts with `<|vision_start|><|image_pad|><|vision_end|>` vision placeholders and sends JPEG (`quality=95`) `image_data` to SGLang's native `POST /generate` endpoint (`api_backend="sglang_generate"`).

## 2. Run OSWorld-Verified Evaluation

1. Clone the official [OSWorld](https://github.com/xlang-ai/OSWorld) repository and prepare the Ubuntu desktop VM environment (`docker` provider recommended).
2. *(Optional, for Evolved Harness guest Python dependency bundle)* To inject the 15 office/PDF/data/vision Python distributions (`numpy`, `openpyxl`, `XlsxWriter`, `python-docx`, `python-pptx`, `pypdf`, `pandas`, `opencv-python`, etc.) into the Ubuntu VM before task execution, run the evaluation host under **CPython 3.10** (matching the Ubuntu guest ABI) with those packages installed. If running under another Python version, `MAGEHybridAgent` automatically logs a warning and falls back to workspace-only initialization (or pass `--no_bootstrap_python_deps`).
3. Run the evaluation launcher:

```bash
export MODEL_DIR=/path/to/Mage-VL-Cowork
export OSWORLD_ROOT=/path/to/OSWorld
export SGLANG_BASE_URL=http://127.0.0.1:30001
export NUM_ENVS=4

bash osworld_eval/scripts/run_osworld_eval.sh
```

## 3. Standard Evaluation Hyperparameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `--api_backend` | `sglang_generate` | Uses `AutoProcessor` (`enable_thinking=True`) + SGLang native `POST /generate` with JPEG (`quality=95`) `image_data` |
| `--test_all_meta_path` | `evaluation_examples/test_nogdrive.json` | 361 OSWorld-Verified tasks excluding Google Drive |
| `--max_steps` | `50` | Maximum interaction turns per task |
| `--history_n` | `3` | Full multimodal/text dialogue turns retained in context |
| `--images_to_keep` | `3` | Recent desktop screenshots retained (`1920×1080`, JPEG `quality=95`) |
| `--temperature` / `--top_p` | `0.7` / `0.8` | Standard sampling parameters (reported score is averaged over 2 runs) |
| `--max_tokens` | `4096` | Max new tokens per step (`<think>` + `Action:` + `<tool_call>`) |
| `--sleep_after_execution` | `3.0` | Seconds to wait after each `pyautogui` GUI action |
| `--bash_timeout` | `300` | Timeout in seconds per `bash` tool execution |
| `--empty_action_retries` | `2` | Automatic resamples when a step parses 0 executable actions |
| `--bootstrap_guest` | `True` | Prepares `/tmp_workspace`, injects guest Python dependencies (`pydeps_bundle.py`), disables GNOME idle screen lock, and logs `bash` commands to `~/.bash_history` |
