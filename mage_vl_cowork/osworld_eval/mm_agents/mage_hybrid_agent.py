"""Mage-VL-Cowork hybrid computer-use + bash agent for OSWorld.

The model may emit either ``computer_use`` calls (mapped to OSWorld pyautogui code)
or ``bash`` calls (executed through ``PythonController.run_bash_script``).

One model response must use only one tool family. Computer calls receive one
post-action screenshot; bash calls receive their merged stdout/stderr text.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import shlex
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from PIL import Image

from mm_agents.mage_gui_agent import (
    InvalidScreenshotError,
    MAX_RETRY_TIMES,
    MAX_TOOL_CALLS,
    MAGEGUIAgent,
    _data_image_part,
    _extract_action_sentence,
    _replace_string_in_tool_call,
    _summarize_actions,
    encode_jpeg95,
)

_DEFAULT_LOGGER = logging.getLogger("desktopenv.mage_gui_agent")

DOUBLE_FOLDED_MAX_STEP = 49
DOUBLE_FOLD_TRIGGER_OFFSET = 1
BASH_OUTPUT_LIMIT = 5_000
IDLE_LOCK_SETTINGS = (
    ("org.gnome.desktop.session", "idle-delay", "uint32 0"),
    ("org.gnome.desktop.screensaver", "idle-activation-enabled", "false"),
    ("org.gnome.desktop.screensaver", "lock-enabled", "false"),
    (
        "org.gnome.settings-daemon.plugins.power",
        "sleep-inactive-ac-type",
        "'nothing'",
    ),
    (
        "org.gnome.settings-daemon.plugins.power",
        "sleep-inactive-battery-type",
        "'nothing'",
    ),
)

MAGE_HYBRID_SYSTEM_PROMPT = 'You are a multi-purpose intelligent assistant. Based on my requests, you can use tools to help me complete various tasks.\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer screen.", "parameters": {"properties": {"action": {"description": "* `click`: Click at (x,y). Use `button` for left/right/middle.\\n* `double_click`: Double click at (x,y).\\n* `triple_click`: Triple click at (x,y) to select line/paragraph.\\n* `move`: Move cursor to (x,y) without clicking.\\n* `drag`: Drag from `start` to `end` coordinates.\\n* `type`: Type text string.\\n* `keypress`: Press key(s). Single key or combo.\\n* `scroll`: Scroll at optional (x,y) position. scroll_x/scroll_y in [-999,999], |999|=full screen. Negative scroll_y=down, negative scroll_x=left.\\n* `wait`: Wait specified seconds.\\n* `screenshot`: Capture the current screen without changing anything.\\n* `terminate`: End the task.", "enum": ["click", "double_click", "triple_click", "move", "drag", "type", "keypress", "scroll", "wait", "screenshot", "terminate"], "type": "string"}, "coordinate": {"description": "(x,y) in [0,999].", "type": "array"}, "button": {"description": "Mouse button for click: left/right/middle.", "type": "string", "enum": ["left", "right", "middle"]}, "start": {"description": "(x,y) drag start in [0,999].", "type": "array"}, "end": {"description": "(x,y) drag end in [0,999].", "type": "array"}, "keys": {"description": "Key list for keypress.", "type": "array"}, "text": {"description": "Text for action=type.", "type": "string"}, "scroll_x": {"description": "Horizontal scroll [-999,999]. Negative=left.", "type": "number"}, "scroll_y": {"description": "Vertical scroll [-999,999]. Negative=down.", "type": "number"}, "time": {"description": "Seconds for wait.", "type": "number"}, "status": {"description": "success/failure for terminate.", "type": "string", "enum": ["success", "failure"]}, "msg_to_user": {"description": "Optional message for terminate.", "type": "string"}}, "required": ["action"], "type": "object"}}}\n{"type": "function", "function": {"name": "bash", "description": "Run a shell command on the machine and read its output. Use it for file inspection and edits, and for anything the shell does more reliably than the GUI; the result comes back as text, not as a screenshot.", "parameters": {"properties": {"command": {"description": "The shell command to run.", "type": "string"}}, "required": ["command"], "type": "object"}}}\n</tools>\n\nFor every step, reply in EXACTLY this format with NO suffix:\n\n<think>\nyour reasoning paragraph describing the current screen state, how it connects to the task goal, and the rationale for the chosen action\n</think>\n\nAction: one short imperative sentence describing the next move.\n\n<tool_call>\n{"name": "example_function_name", "arguments": {"example_parameter_1": "value_1"}}\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format\n- Required parameters MUST be specified\n- Reasoning goes inside <think>...</think> BEFORE the Action line; nothing after the last tool_call\n- You may output at most 15 <tool_call> blocks per response\n- Use `type` for text input, `keypress` for shortcuts and special keys\n- Every tool_call in one response must use the same function\n</IMPORTANT>\n\n# Response format\n\nFor every step:\n1) <think>...</think> block with your reasoning.\n2) Action: a short imperative sentence.\n3) One or more <tool_call>...</tool_call> blocks (max 15).\n\nRules:\n- The screen\'s resolution is 1000x1000. Coordinates are in [0, 999] range.\n- Scroll amounts are normalized: |999| = one full screen width/height.\n- For scroll, you may include coordinate to scroll at that position.\n- For drag, specify start and end coordinates.\n- Choose `bash` when the shell is the reliable way to do or check something, and `computer_use` when the work has to happen in the interface on screen.\n- A `bash` call is answered with the command\'s text output; a `computer_use` call is answered with a new screenshot.\n- Be brief: one sentence for Action.\n- Do not output anything after the last tool call.\n- If finishing, use action=terminate.'

_PROCESSOR_LOAD_LOCK = threading.RLock()
_PROCESSOR_CACHE: dict[str, Any] = {}
_DATA_URI_RE = re.compile(r"^data:(?P<mime>[^;,]+)?(?:;[^,]*)?;base64,(?P<data>.*)$", re.DOTALL)


def _load_processor(name_or_path: str) -> Any:
    from transformers import AutoProcessor

    key = str(name_or_path)
    with _PROCESSOR_LOAD_LOCK:
        processor = _PROCESSOR_CACHE.get(key)
        if processor is None:
            processor = AutoProcessor.from_pretrained(key, trust_remote_code=True)
            _PROCESSOR_CACHE[key] = processor
        return processor


def _to_hf_processor_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            new_content: Any = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            new_content = []
            for part in content:
                if isinstance(part, str):
                    new_content.append({"type": "text", "text": part})
                    continue
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "image_url":
                    url = (part.get("image_url") or {}).get("url")
                    if url:
                        new_content.append({"type": "image", "image": url})
                else:
                    new_content.append(dict(part))
        else:
            new_content = []
        converted.append({"role": message.get("role", "user"), "content": new_content})
    return converted


def _extract_image_base64_list(messages: list[dict[str, Any]]) -> list[str]:
    images: list[str] = []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            url = None
            if part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url")
            elif part.get("type") == "image":
                url = part.get("image")
            if not isinstance(url, str) or not url:
                continue
            match = _DATA_URI_RE.match(url)
            if match:
                raw_bytes = base64.b64decode(match.group("data"), validate=False)
                img = Image.open(BytesIO(raw_bytes)).convert("RGB")
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=95)
                images.append(base64.b64encode(buf.getvalue()).decode("ascii"))
            else:
                images.append(url)
    return images


def _render_prompt_with_processor(
    messages: list[dict[str, Any]],
    *,
    processor_model_name: str,
    enable_thinking: bool = True,
) -> str:
    processor = _load_processor(processor_model_name)
    hf_messages = _to_hf_processor_messages(messages)
    attempts = [
        {
            "tokenize": False,
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        },
        {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
        },
        {"tokenize": False, "add_generation_prompt": True},
    ]
    last_exc: Exception | None = None
    for kwargs in attempts:
        try:
            text = processor.apply_chat_template(hf_messages, **kwargs)
            if isinstance(text, list):
                text = text[0] if text else ""
            if isinstance(text, str) and text:
                return text
        except TypeError as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Failed to render prompt with Hugging Face AutoProcessor")


class MAGEHybridAgent(MAGEGUIAgent):
    """Mage-VL-Cowork policy matching the hybrid computer_use + bash contract."""

    def __init__(
        self,
        model: str = "Mage-VL-Cowork",
        processor_model_name: str | None = None,
        max_tokens: int = 4096,
        top_p: float = 0.8,
        temperature: float = 0.7,
        action_space: str = "pyautogui",
        observation_type: str = "screenshot",
        history_n: int = 3,
        images_to_keep: int = 3,
        coordinate_type: str = "relative",
        screen_width: int = 1920,
        screen_height: int = 1080,
        api_backend: str = "sglang_generate",
        bash_timeout: int = 300,
        max_steps: int | None = 50,
        bootstrap_guest: bool = True,
        bootstrap_python_deps: bool = True,
        disable_idle_lock: bool = True,
        request_timeout: float = 120.0,
    ):
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            action_space=action_space,
            observation_type=observation_type,
            history_n=history_n,
            images_to_keep=images_to_keep,
            coordinate_type=coordinate_type,
            screen_width=screen_width,
            screen_height=screen_height,
            api_backend=api_backend,
        )
        self.processor_model_name = processor_model_name or os.environ.get(
            "MAGE_PROCESSOR_PATH", "microsoft/Mage-VL-Cowork"
        )
        self.bash_timeout = int(bash_timeout)
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.bootstrap_guest_enabled = bool(bootstrap_guest)
        self.bootstrap_python_deps = bool(bootstrap_python_deps)
        self.disable_idle_lock = bool(disable_idle_lock)
        self.request_timeout = float(request_timeout)
        self.enable_thinking = True
        self.thinking_budget = self.max_tokens

        self.step_inputs: list[dict[str, str]] = []
        self.tool_names: list[str | None] = []
        self.last_parsed_tool_name: str | None = None
        self._pending_bash_output = ""
        self._has_pending_bash_output = False

    def bootstrap_guest(
        self,
        env: Any,
        *,
        task_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Prepare the OSWorld guest for reliable Hybrid Bash execution.

        When ``bootstrap_python_deps=True`` (and ``desktop_env.pydeps_bundle`` is
        available on a CPython 3.10 host), the shared document/PDF/data/vision
        dependency bundle is copied into the guest VM. Workspace directories
        (``/tmp_workspace``) and idle-lock hardening are always configured.
        """
        evidence: dict[str, Any] = {
            "enabled": self.bootstrap_guest_enabled,
            "workspace": {"ok": False},
            "python_deps": {"enabled": self.bootstrap_python_deps},
            "idle_lock": {"enabled": self.disable_idle_lock},
        }
        if not self.bootstrap_guest_enabled:
            return evidence

        self.logger.info("Bootstrapping Mage-VL-Cowork guest prerequisites")
        controller = env.controller
        password = str(getattr(env, "client_password", None) or "password")

        pydeps_extracted = False
        if self.bootstrap_python_deps:
            try:
                from desktop_env.pydeps_bundle import (
                    REMOTE_PYDEPS_DIR,
                    REMOTE_PYDEPS_TARBALL,
                    ensure_python_deps_tarball as _ensure_python_deps_tarball,
                    guest_extract_script,
                    guest_verify_command,
                )

                tarball = _ensure_python_deps_tarball()
                _upload_guest_file(env, tarball, REMOTE_PYDEPS_TARBALL)
                setup_body = (
                    "set -euo pipefail; "
                    f"{guest_extract_script(REMOTE_PYDEPS_TARBALL)}; "
                    "mkdir -p /tmp_workspace/results /tmp_workspace/tmp; "
                    f"chmod -R a+rX {shlex.quote(REMOTE_PYDEPS_DIR)}; "
                    "chmod -R 0777 /tmp_workspace"
                )
                setup_command = (
                    f"printf '%s\\n' {shlex.quote(password)} | "
                    f"sudo -S -p '' /bin/bash -lc {shlex.quote(setup_body)}"
                )
                setup_result = _run_guest_bash(
                    controller,
                    setup_command,
                    timeout=max(120, self.bash_timeout),
                )
                if int(setup_result.get("returncode", -1)) != 0:
                    raise RuntimeError(
                        f"Mage-VL-Cowork Python dependency setup failed: {setup_result}"
                    )
                verify_command = " ".join(
                    shlex.quote(argument) for argument in guest_verify_command()
                )
                verify_result = _run_guest_bash(
                    controller,
                    verify_command,
                    timeout=60,
                    working_dir="/tmp_workspace",
                )
                if int(verify_result.get("returncode", -1)) != 0:
                    raise RuntimeError(
                        f"Mage-VL-Cowork Python dependency verification failed: {verify_result}"
                    )
                evidence["python_deps"].update(
                    {
                        "ok": True,
                        "bundle": tarball.name,
                        "versions": str(verify_result.get("output") or "").strip(),
                    }
                )
                evidence["workspace"] = {"ok": True, "returncode": 0}
                pydeps_extracted = True
            except Exception as exc:
                self.logger.warning(
                    "Guest Python dependency bundle skipped (%s: %s); falling back to workspace-only setup.",
                    type(exc).__name__,
                    exc,
                )
                evidence["python_deps"].update(
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                )

        if not pydeps_extracted:
            setup_body = (
                "set -euo pipefail; "
                "mkdir -p /tmp_workspace/results /tmp_workspace/tmp; "
                "chmod -R 0777 /tmp_workspace"
            )
            setup_command = (
                f"printf '%s\\n' {shlex.quote(password)} | "
                f"sudo -S -p '' /bin/bash -lc {shlex.quote(setup_body)}"
            )
            setup_result = _run_guest_bash(
                controller,
                setup_command,
                timeout=max(60, self.bash_timeout),
            )
            evidence["workspace"] = {
                "ok": int(setup_result.get("returncode", -1)) == 0,
                "returncode": setup_result.get("returncode"),
            }

        if self.disable_idle_lock:
            try:
                evidence["idle_lock"] = _disable_guest_idle_lock(
                    controller,
                    task_config=task_config,
                )
            except Exception as exc:
                evidence["idle_lock"] = {
                    "enabled": True,
                    "ok": False,
                    "keys": [],
                    "xset": [],
                    "errors": [f"{type(exc).__name__}: {exc}"],
                }
            if not evidence["idle_lock"].get("ok"):
                self.logger.warning(
                    "Mage-VL-Cowork idle-lock hardening was incomplete: %s",
                    evidence["idle_lock"].get("errors"),
                )
        self.logger.info("Mage-VL-Cowork guest bootstrap complete: %s", evidence)
        return evidence

    def reset(self, _logger=None, vm_ip=None):
        super().reset(_logger=_logger, vm_ip=vm_ip)
        self.step_inputs = []
        self.tool_names = []
        self.last_parsed_tool_name = None
        self._pending_bash_output = ""
        self._has_pending_bash_output = False

    def call_llm(self, payload: dict[str, Any], model: str) -> str:
        messages = payload["messages"]
        if self.api_backend != "sglang_generate":
            raise ValueError(
                f"Unsupported api_backend={self.api_backend!r}; only 'sglang_generate' "
                "(AutoProcessor + SGLang POST /generate) is supported."
            )
        return self._call_llm_sglang_generate(messages, model)

    def _call_llm_sglang_generate(self, messages: list[dict[str, Any]], model: str) -> str:
        base_url = os.environ.get(
            "SGLANG_BASE_URL",
            os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:30001"),
        ).rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3].rstrip("/")
        generate_url = f"{base_url}/generate"
        api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")

        image_data = _extract_image_base64_list(messages)
        prompt_text = _render_prompt_with_processor(
            messages,
            processor_model_name=self.processor_model_name,
            enable_thinking=self.enable_thinking,
        )

        request_payload = {
            "model": model,
            "text": prompt_text,
            "image_data": image_data,
            "sampling_params": {
                "max_new_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
            "return_logprob": True,
            "logprob_start_len": -1,
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRY_TIMES + 1):
            try:
                resp = requests.post(
                    generate_url,
                    json=request_payload,
                    headers=headers,
                    timeout=self.request_timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, dict):
                    raise RuntimeError("SGLang /generate returned non-object JSON")
                return str(data.get("text") or "")
            except Exception as exc:
                last_error = exc
                self.logger.error(
                    "[SGLang /generate] call failed attempt %s/%s: %s",
                    attempt,
                    MAX_RETRY_TIMES,
                    exc,
                )
                if attempt < MAX_RETRY_TIMES:
                    time.sleep(min(5.0, 1.5 * attempt))
        raise RuntimeError(f"SGLang /generate call failed: {last_error}")

    def predict(self, instruction: str, obs: Dict[str, Any]) -> Tuple[str, List[Any]]:
        current_input = self._make_current_step_input(obs)
        messages = self._build_messages(instruction, current_input)
        self.last_messages = messages

        response = self.call_llm(
            {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "temperature": self.temperature,
            },
            self.model,
        )
        self.logger.info("MAGE output: %s", response)

        action_sentence, executable_actions = self.parse_response(
            response,
            original_width=self.screen_width,
            original_height=self.screen_height,
        )
        self.logger.info("MAGE action sentence: %s", action_sentence)
        self.logger.info("MAGE executable actions: %s", executable_actions)

        self.step_inputs.append(current_input)
        self.responses.append(response)
        self.action_summaries.append(
            action_sentence or _summarize_hybrid_actions(executable_actions)
        )
        self.actions.append(executable_actions)
        self.tool_names.append(self.last_parsed_tool_name)
        return response, executable_actions

    def _make_current_step_input(self, obs: Dict[str, Any]) -> dict[str, str]:
        if self.responses and self.tool_names[-1] == "bash":
            if not self._has_pending_bash_output:
                raise RuntimeError(
                    "bash actions were generated but their tool output was not recorded"
                )
            output = self._pending_bash_output
            self._pending_bash_output = ""
            self._has_pending_bash_output = False
            return {"kind": "bash", "data": output}

        try:
            screenshot_b64, (width, height) = encode_jpeg95(
                obs.get("screenshot") if isinstance(obs, dict) else None
            )
        except InvalidScreenshotError as exc:
            raise InvalidScreenshotError(
                f"[gui_step {len(self.responses) + 1}] {exc}"
            ) from exc
        self.screen_width = width
        self.screen_height = height
        return {"kind": "image", "data": screenshot_b64}

    def _build_messages(
        self,
        instruction: str,
        current_input: dict[str, str] | str,
    ) -> list[dict[str, Any]]:
        if isinstance(current_input, str):
            current_input = {"kind": "image", "data": current_input}

        current_step = len(self.responses) + 1
        all_inputs = [*self.step_inputs, current_input]
        if len(all_inputs) != current_step:
            raise RuntimeError("MAGE hybrid step/input history is out of sync")

        start_step = max(1, current_step - self.history_n)
        previous_actions = _build_previous_action_lines(
            self.action_summaries,
            n_folded=start_step - 1,
            history_n=self.history_n,
            double_folded_max_step=DOUBLE_FOLDED_MAX_STEP,
            cap_bonus=int(
                self.max_steps is not None
                and current_step == self.max_steps
                and DOUBLE_FOLDED_MAX_STEP > 0
            ),
        )
        previous_actions_str = "\n".join(previous_actions) if previous_actions else "None"

        image_steps = [
            step
            for step in range(start_step, current_step + 1)
            if all_inputs[step - 1].get("kind") == "image"
        ]
        keep_count = max(0, self.images_to_keep)
        kept_image_steps = set(image_steps[-keep_count:] if keep_count else [])

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": MAGE_HYBRID_SYSTEM_PROMPT}
        ]
        messages.append(
            {
                "role": "user",
                "content": self._first_user_content(
                    all_inputs[start_step - 1],
                    step_number=start_step,
                    keep_image=start_step in kept_image_steps,
                    instruction=instruction,
                    previous_actions=previous_actions_str,
                ),
            }
        )

        for step in range(start_step, current_step):
            messages.append({"role": "assistant", "content": self.responses[step - 1]})
            next_step = step + 1
            messages.append(
                {
                    "role": "user",
                    "content": self._tool_response_content(
                        all_inputs[next_step - 1],
                        keep_image=next_step in kept_image_steps,
                    ),
                }
            )
        return messages

    @staticmethod
    def _instruction_context(instruction: str, previous_actions: str, *, tool: str) -> str:
        if tool == "bash":
            lead = "Please generate the next move according to the tool output, instruction and previous actions."
        else:
            lead = "Please generate the next move according to the UI screenshot, instruction and previous actions."
        return (
            f"{lead}\n\n"
            f"Instruction: {instruction}\n\n"
            f"Previous actions:\n{previous_actions}"
        )

    def _first_user_content(
        self,
        step_input: dict[str, str],
        *,
        step_number: int,
        keep_image: bool,
        instruction: str,
        previous_actions: str,
    ) -> Any:
        kind = step_input.get("kind")
        data = step_input.get("data", "")
        if kind == "bash":
            context = self._instruction_context(
                instruction, previous_actions, tool="bash"
            )
            return f"{context}\n\n<tool_response>\n{data}\n</tool_response>"

        context = self._instruction_context(
            instruction, previous_actions, tool="computer_use"
        )
        text = "\n" + context
        if keep_image:
            return [_data_image_part(data), {"type": "text", "text": text}]
        return "[This screenshot has been collapsed.]" + text

    @staticmethod
    def _tool_response_content(step_input: dict[str, str], *, keep_image: bool) -> Any:
        kind = step_input.get("kind")
        data = step_input.get("data", "")
        if kind == "bash":
            return f"<tool_response>\n{data}\n</tool_response>"
        if keep_image:
            return [
                {"type": "text", "text": "<tool_response>\n"},
                _data_image_part(data),
                {"type": "text", "text": "\n</tool_response>"},
            ]
        return "<tool_response>\n[This screenshot has been collapsed.]\n</tool_response>"

    def parse_response(
        self,
        response: str,
        original_width: int | None = None,
        original_height: int | None = None,
    ) -> Tuple[str, List[Any]]:
        self.last_parsed_tool_name = None
        if not response or not response.strip():
            return "", []

        action_sentence = _extract_action_sentence(response)
        calls = _extract_hybrid_tool_calls(response, logger=self.logger)
        if not calls:
            return action_sentence, []

        tool_names = {
            str(call.get("name") or "") for call in calls[:MAX_TOOL_CALLS]
        }
        if len(tool_names) != 1 or not tool_names.issubset({"computer_use", "bash"}):
            self.logger.warning(
                "MAGE requires one tool family per response; got %s",
                sorted(tool_names),
            )
            return action_sentence, []

        tool_name = next(iter(tool_names))
        if tool_name == "computer_use":
            action_sentence, executable_actions = self._parse_computer_response(
                response,
                original_width=original_width,
                original_height=original_height,
                max_tool_calls=MAX_TOOL_CALLS,
                calls=calls,
            )
            if executable_actions:
                self.last_parsed_tool_name = tool_name
            return action_sentence, executable_actions

        if len(calls) > MAX_TOOL_CALLS:
            self.logger.warning(
                "MAGE response produced %d tool_calls; truncating to first %d",
                len(calls),
                MAX_TOOL_CALLS,
            )
            calls = calls[:MAX_TOOL_CALLS]

        executable_actions: list[Any] = []
        for raw_call in calls:
            call = _replace_string_in_tool_call(
                raw_call, "osworld-public-evaluation", "password"
            )
            if tool_name == "bash":
                args = call.get("arguments")
                command = args.get("command") if isinstance(args, dict) else None
                if not isinstance(command, str) or not command:
                    self.logger.warning("MAGE bash call has no command: %s", call)
                    continue
                executable_actions.append(
                    {"action_type": "bash", "command": command}
                )

        if executable_actions:
            self.last_parsed_tool_name = tool_name
        if not action_sentence:
            action_sentence = _summarize_hybrid_actions(executable_actions)
        return action_sentence, executable_actions

    def execute_bash_actions(
        self,
        controller: Any,
        actions: list[dict[str, Any]],
        *,
        timeout: int | None = None,
    ) -> list[dict[str, Any]]:
        """Execute parsed bash actions and record the next textual tool response."""
        execution_results: list[dict[str, Any]] = []
        outputs: list[str] = []
        effective_timeout = self.bash_timeout if timeout is None else int(timeout)
        for action in actions:
            if not isinstance(action, dict) or action.get("action_type") != "bash":
                raise ValueError(f"not a MAGE hybrid bash action: {action!r}")
            command = str(action.get("command") or "")
            execution_script = (
                "if [ ! -d /tmp_workspace ] || [ ! -w /tmp_workspace ]; then "
                "uid=$(id -u) && gid=$(id -g) && "
                "printf 'password\\n' | sudo -S -p '' install -d "
                "-o \"$uid\" -g \"$gid\" -m 0755 /tmp_workspace; "
                "fi && cd /tmp_workspace && "
                f"{_bash_history_append_script(command)}"
                f"exec /bin/bash -lc {shlex.quote(command)}"
            )
            result = controller.run_bash_script(
                execution_script,
                timeout=effective_timeout,
                working_dir=None,
            )
            if not isinstance(result, dict):
                result = {
                    "status": "error",
                    "output": "Failed to execute bash command.",
                    "error": "controller returned no result",
                    "returncode": -1,
                }
            raw_stdout = str(result.get("output") or "")
            error = str(result.get("error") or "")
            raw_output = f"{raw_stdout}{error}" if error else raw_stdout
            output = _bash_output_for_model(raw_stdout)
            if error:
                output = f"{output}{error}" if output else error
            outputs.append(output)
            execution_results.append(
                {
                    **result,
                    "raw_output": raw_output,
                    "output": output,
                }
            )

        self.record_bash_output(_merge_bash_outputs(outputs))
        return execution_results

    def record_bash_output(self, output: str) -> None:
        self._pending_bash_output = str(output or "")
        self._has_pending_bash_output = True


def _bash_history_append_script(command: str) -> str:
    """Return a guest shell prefix that records ``command`` in ~/.bash_history."""
    entry = command.rstrip("\r\n")
    if not entry.strip():
        return ""
    return (
        "{ (umask 077 && printf '%s\\n' "
        f"{shlex.quote(entry)} >> ~/.bash_history) 2>/dev/null || true; }} && "
    )


def _run_guest_bash(
    controller: Any,
    command: str,
    *,
    timeout: int,
    working_dir: str | None = None,
) -> dict[str, Any]:
    result = controller.run_bash_script(
        command,
        timeout=int(timeout),
        working_dir=working_dir,
    )
    if isinstance(result, dict):
        return result
    return {
        "status": "error",
        "output": "",
        "error": "controller returned no result",
        "returncode": -1,
    }


def _upload_guest_file(env: Any, local_path: Path, remote_path: str) -> None:
    """Upload one bootstrap asset through OSWorld's guest setup endpoint."""
    server_port = int(getattr(env, "server_port", 5000))
    url = f"http://{env.vm_ip}:{server_port}/setup/upload"
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with local_path.open("rb") as stream:
                response = requests.post(
                    url,
                    files={
                        "file_data": (
                            local_path.name,
                            stream,
                            "application/gzip",
                        )
                    },
                    data={"file_path": remote_path},
                    timeout=(10, 600),
                )
            response.raise_for_status()
            return
        except requests.RequestException as exc:
            last_error = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if attempt >= 3 or (status is not None and status < 500):
                break
            time.sleep(0.5 * attempt)
    raise RuntimeError(f"Mage-VL-Cowork guest upload failed: {last_error}")


def _idle_lock_settings_for(
    task_config: dict[str, Any] | None,
) -> tuple[list[list[str]], list[dict[str, str]]]:
    if not isinstance(task_config, dict):
        return [list(item) for item in IDLE_LOCK_SETTINGS], []
    task_text = json.dumps(task_config, ensure_ascii=False, default=str)
    allowed: list[list[str]] = []
    skipped: list[dict[str, str]] = []
    for schema, key, value in IDLE_LOCK_SETTINGS:
        if schema in task_text and key in task_text:
            skipped.append(
                {
                    "schema": schema,
                    "key": key,
                    "reason": "task references this key; leaving it to the task",
                }
            )
        else:
            allowed.append([schema, key, value])
    return allowed, skipped


def _disable_guest_idle_lock(
    controller: Any,
    *,
    task_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Disable guest idle lock best-effort and return read-back evidence."""
    allowed, skipped = _idle_lock_settings_for(task_config)
    script = (
        "import json, os, subprocess\n"
        f"settings = json.loads({json.dumps(allowed)!r})\n"
        "environ = dict(os.environ)\n"
        "environ['DISPLAY'] = ':0'\n"
        "environ['XDG_RUNTIME_DIR'] = '/run/user/1000'\n"
        "environ['DBUS_SESSION_BUS_ADDRESS'] = 'unix:path=/run/user/1000/bus'\n"
        "result = {'ok': False, 'keys': [], 'xset': [], 'errors': []}\n"
        "for schema, key, value in settings:\n"
        "    entry = {'schema': schema, 'key': key, 'wanted': value, "
        "'read_back': '', 'ok': False}\n"
        "    try:\n"
        "        applied = subprocess.run(['gsettings', 'set', schema, key, value], "
        "capture_output=True, text=True, env=environ, timeout=10)\n"
        "        read = subprocess.run(['gsettings', 'get', schema, key], "
        "capture_output=True, text=True, env=environ, timeout=10)\n"
        "        entry['read_back'] = read.stdout.strip()\n"
        "        entry['ok'] = (applied.returncode == 0 and read.returncode == 0 "
        "and entry['read_back'] == value)\n"
        "        if not entry['ok']:\n"
        "            entry['error'] = (applied.stderr.strip() or read.stderr.strip() "
        "or 'gsettings read-back mismatch')\n"
        "    except Exception as exc:\n"
        "        entry['error'] = type(exc).__name__ + ': ' + str(exc)\n"
        "    if not entry['ok']:\n"
        "        result['errors'].append(schema + ' ' + key + ': ' + "
        "entry.get('error', entry['read_back']))\n"
        "    result['keys'].append(entry)\n"
        "for command in (['xset', 's', 'off'], ['xset', '-dpms'], "
        "['xset', 's', 'noblank']):\n"
        "    entry = {'command': command, 'ok': False}\n"
        "    try:\n"
        "        proc = subprocess.run(command, capture_output=True, text=True, "
        "env=environ, timeout=10)\n"
        "        entry['ok'] = proc.returncode == 0\n"
        "        if not entry['ok']:\n"
        "            entry['error'] = proc.stderr.strip() or str(proc.returncode)\n"
        "    except Exception as exc:\n"
        "        entry['error'] = type(exc).__name__ + ': ' + str(exc)\n"
        "    if not entry['ok']:\n"
        "        result['errors'].append(' '.join(command) + ': ' + "
        "entry.get('error', 'failed'))\n"
        "    result['xset'].append(entry)\n"
        "result['ok'] = bool(result['keys']) and all(x['ok'] for x in result['keys']) "
        "and all(x['ok'] for x in result['xset'])\n"
        "print('MAGE_HYBRID_IDLE_LOCK=' + json.dumps(result, sort_keys=True))\n"
    )
    command = "/usr/bin/python3 -c " + shlex.quote(script)
    response = _run_guest_bash(
        controller,
        command,
        timeout=90,
        working_dir="/tmp_workspace",
    )
    evidence: dict[str, Any] = {
        "enabled": True,
        "ok": False,
        "keys": [],
        "xset": [],
        "errors": [],
        "skipped": skipped,
        "returncode": response.get("returncode"),
    }
    for line in str(response.get("output") or "").splitlines():
        if line.startswith("MAGE_HYBRID_IDLE_LOCK="):
            try:
                parsed = json.loads(line.split("=", 1)[1])
                if isinstance(parsed, dict):
                    evidence.update(parsed)
            except json.JSONDecodeError as exc:
                evidence["errors"].append(f"malformed idle-lock evidence: {exc}")
    if not allowed:
        evidence["ok"] = False
        evidence["errors"].append(
            "all idle-lock keys are owned by this task; none were changed"
        )
    if not evidence.get("keys") and allowed:
        evidence["errors"].append(
            str(response.get("error") or "idle-lock command returned no evidence")
        )
    return evidence


def _extract_hybrid_tool_calls(
    response: str,
    *,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
) -> list[dict[str, Any]]:
    active_logger = logger or _DEFAULT_LOGGER
    calls: list[dict[str, Any]] = []
    for match in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", response, flags=re.DOTALL):
        raw = match.group(1).strip()
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            active_logger.error(
                "Failed to parse MAGE tool_call JSON: %s text=%r",
                exc,
                raw[:300],
            )
            continue
        if isinstance(obj, dict):
            calls.append(obj)

    if not calls:
        for line in response.splitlines():
            line = line.strip()
            if not (line.startswith("{") and line.endswith("}")):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("name") in {"computer_use", "bash"}:
                calls.append(obj)
    return calls


def _merge_bash_outputs(outputs: list[str]) -> str:
    if not outputs:
        return ""
    merged = outputs[0]
    for output in outputs[1:]:
        if merged and output and not merged.endswith("\n"):
            merged += "\n"
        merged += output
    return merged


def _bash_output_for_model(text: str, limit: int = BASH_OUTPUT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return (
        text[:limit]
        + "\n\n[Output truncated: the content above is the first "
        + f"{limit:,} characters; the actual output contains {len(text):,} characters.]"
    )


_TRAILING_DOTS_RE = re.compile(r"\.+\s*$")
_WHITESPACE_RE = re.compile(r"\s+")


def _merge_two_actions(first: str, second: str) -> str:
    first = _WHITESPACE_RE.sub(" ", first or "").strip()
    second = _WHITESPACE_RE.sub(" ", second or "").strip()
    if not first and not second:
        return "Performing action"
    head = _TRAILING_DOTS_RE.sub("", first)
    merged = (head + " and " + second) if (head and second) else (head or second)
    merged = merged.strip()
    if not merged:
        return "Performing action"
    if merged[-1] not in ".!?":
        merged += "."
    return merged


def _build_previous_action_lines(
    action_summaries: list[str],
    *,
    n_folded: int,
    history_n: int,
    double_folded_max_step: int,
    cap_bonus: int = 0,
) -> list[str]:
    if n_folded <= 0:
        return []
    if n_folded > len(action_summaries):
        raise RuntimeError("MAGE hybrid action-summary history is out of sync")
    if double_folded_max_step <= 0:
        return [
            f"Step {index + 1}: {action_summaries[index]}"
            for index in range(n_folded)
        ]

    trigger_offset = DOUBLE_FOLD_TRIGGER_OFFSET - int(cap_bonus)
    cap = double_folded_max_step - history_n - trigger_offset
    if cap < 1:
        raise ValueError(
            "double-folded max step must exceed history_n plus trigger offset"
        )
    merges = max(0, n_folded - cap)
    if 2 * merges > n_folded:
        merges = n_folded // 2

    descriptions: list[str] = []
    for pair_index in range(merges):
        descriptions.append(
            _merge_two_actions(
                action_summaries[2 * pair_index],
                action_summaries[2 * pair_index + 1],
            )
        )
    descriptions.extend(action_summaries[2 * merges : n_folded])
    return [
        f"Step {index + 1}: {description}"
        for index, description in enumerate(descriptions)
    ]


def _summarize_hybrid_actions(actions: list[Any]) -> str:
    if not actions:
        return "No executable action"
    first = actions[0]
    if isinstance(first, dict) and first.get("action_type") == "bash":
        return "Run the requested shell command."
    return _summarize_actions([str(action) for action in actions])


__all__ = [
    "MAGEHybridAgent",
    "MAGE_HYBRID_SYSTEM_PROMPT",
    "MAX_TOOL_CALLS",
    "DOUBLE_FOLDED_MAX_STEP",
    "_extract_hybrid_tool_calls",
]
