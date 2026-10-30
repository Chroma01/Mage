"""Mage-VL-Cowork GUI tool-call base agent for OSWorld.

This agent implements the Mage GUI action contract:
``<think>`` + ``Action:`` + one or more ``<tool_call>`` JSON blocks using the
``computer_use`` action space, mapping normalized [0, 999] coordinates back to
OSWorld pyautogui commands.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

_DEFAULT_LOGGER = logging.getLogger("desktopenv.mage_gui_agent")

MAX_RETRY_TIMES = 5
SCROLL_NORM_PER_CLICK = 100
MAX_TOOL_CALLS = 15

MAGE_GUI_SYSTEM_PROMPT = 'You are a multi-purpose intelligent assistant. Based on my requests, you can use tools to help me complete various tasks.\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer screen.", "parameters": {"properties": {"action": {"description": "* `click`: Click at (x,y). Use `button` for left/right/middle.\\n* `double_click`: Double click at (x,y).\\n* `triple_click`: Triple click at (x,y) to select line/paragraph.\\n* `move`: Move cursor to (x,y) without clicking.\\n* `drag`: Drag from `start` to `end` coordinates.\\n* `type`: Type text string.\\n* `keypress`: Press key(s). Single key or combo.\\n* `scroll`: Scroll at optional (x,y) position. scroll_x/scroll_y in [-999,999], |999|=full screen. Negative scroll_y=down, negative scroll_x=left.\\n* `wait`: Wait specified seconds.\\n* `terminate`: End the task.", "enum": ["click", "double_click", "triple_click", "move", "drag", "type", "keypress", "scroll", "wait", "terminate"], "type": "string"}, "coordinate": {"description": "(x,y) in [0,999].", "type": "array"}, "button": {"description": "Mouse button for click: left/right/middle.", "type": "string", "enum": ["left", "right", "middle"]}, "start": {"description": "(x,y) drag start in [0,999].", "type": "array"}, "end": {"description": "(x,y) drag end in [0,999].", "type": "array"}, "keys": {"description": "Key list for keypress.", "type": "array"}, "text": {"description": "Text for action=type.", "type": "string"}, "scroll_x": {"description": "Horizontal scroll [-999,999]. Negative=left.", "type": "number"}, "scroll_y": {"description": "Vertical scroll [-999,999]. Negative=down.", "type": "number"}, "time": {"description": "Seconds for wait.", "type": "number"}, "status": {"description": "success/failure for terminate.", "type": "string", "enum": ["success", "failure"]}, "msg_to_user": {"description": "Optional message for terminate.", "type": "string"}}, "required": ["action"], "type": "object"}}}\n</tools>\n\nFor every step, reply in EXACTLY this format with NO suffix:\n\n<think>\nyour reasoning paragraph describing the current screen state, how it connects to the task goal, and the rationale for the chosen action\n</think>\n\nAction: one short imperative sentence describing the next move.\n\n<tool_call>\n{"name": "example_function_name", "arguments": {"example_parameter_1": "value_1"}}\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format\n- Required parameters MUST be specified\n- Reasoning goes inside <think>...</think> BEFORE the Action line; nothing after the last tool_call\n- You may output at most 5 <tool_call> blocks per response\n- Use `type` for text input, `keypress` for shortcuts and special keys\n</IMPORTANT>\n\n# Response format\n\nFor every step:\n1) <think>...</think> block with your reasoning.\n2) Action: a short imperative sentence.\n3) One or more <tool_call>...</tool_call> blocks (max 5).\n\nRules:\n- The screen\'s resolution is 1000x1000. Coordinates are in [0, 999] range.\n- Scroll amounts are normalized: |999| = one full screen width/height.\n- For scroll, you may include coordinate to scroll at that position.\n- For drag, specify start and end coordinates.\n- Be brief: one sentence for Action.\n- Do not output anything after the last tool call.\n- If finishing, use action=terminate.'

KEY_MAPPING = {
    "alt": "alt",
    "arrowdown": "down",
    "arrowleft": "left",
    "arrowright": "right",
    "arrowup": "up",
    "backspace": "backspace",
    "capslock": "capslock",
    "cmd": "command",
    "command": "command",
    "control": "ctrl",
    "ctrl": "ctrl",
    "delete": "delete",
    "down": "down",
    "end": "end",
    "enter": "enter",
    "esc": "esc",
    "escape": "esc",
    "home": "home",
    "insert": "insert",
    "left": "left",
    "meta": "super",
    "option": "option",
    "pagedown": "pagedown",
    "pageup": "pageup",
    "right": "right",
    "shift": "shift",
    "space": "space",
    "super": "super",
    "tab": "tab",
    "up": "up",
    "win": "win",
}


class InvalidScreenshotError(ValueError):
    """Raised when an OSWorld observation screenshot is missing, empty, or not a
    decodable image (e.g. the VM screenshot capture returned None / corrupt bytes).
    """


_IMAGE_MAGIC = (
    b"\x89PNG\r\n\x1a\n",  # PNG (OSWorld default)
    b"\xff\xd8\xff",        # JPEG
    b"GIF87a",
    b"GIF89a",
    b"BM",                  # BMP
)


def _looks_like_image(image_bytes: Any) -> bool:
    """Cheap signature check so we can reject a bad screenshot before PIL does."""
    if not isinstance(image_bytes, (bytes, bytearray)) or len(image_bytes) < 8:
        return False
    head = bytes(image_bytes[:16])
    if any(head.startswith(sig) for sig in _IMAGE_MAGIC):
        return True
    return head[:4] == b"RIFF" and head[8:12] == b"WEBP"


def encode_jpeg95(image_bytes: bytes) -> tuple[str, tuple[int, int]]:
    """Return ``data:image/jpeg`` base64 and original size."""
    if not image_bytes:
        raise InvalidScreenshotError(
            f"empty screenshot bytes (type={type(image_bytes).__name__}, "
            f"len={0 if image_bytes is None else len(image_bytes)})"
        )
    if not _looks_like_image(image_bytes):
        raise InvalidScreenshotError(
            f"screenshot bytes are not a recognized image "
            f"(len={len(image_bytes)}, head={bytes(image_bytes[:16])!r})"
        )
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        size = image.size
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=95)
    except Exception as exc:
        raise InvalidScreenshotError(
            f"failed to decode screenshot ({len(image_bytes)} bytes): "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return base64.b64encode(buf.getvalue()).decode("ascii"), size


def _data_image_part(b64: str) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
    }


class MAGEGUIAgent:
    def __init__(
        self,
        model: str = "Mage-VL-Cowork",
        max_tokens: int = 4096,
        top_p: float = 0.8,
        temperature: float = 0.7,
        action_space: str = "pyautogui",
        observation_type: str = "screenshot",
        history_n: int = 10,
        images_to_keep: int = 1,
        coordinate_type: str = "relative",
        screen_width: int = 1920,
        screen_height: int = 1080,
        api_backend: str = "sglang_generate",
    ):
        if action_space != "pyautogui":
            raise ValueError("MAGEGUIAgent only supports pyautogui action space")
        if observation_type != "screenshot":
            raise ValueError("MAGEGUIAgent only supports screenshot observation")
        if coordinate_type != "relative":
            raise ValueError("Mage uses normalized [0,999] coordinates")
        if api_backend != "sglang_generate":
            raise ValueError("MAGEGUIAgent only supports the 'sglang_generate' backend (AutoProcessor + SGLang POST /generate)")

        self.model = model
        self.max_tokens = int(max_tokens)
        self.top_p = float(top_p)
        self.temperature = float(temperature)
        self.action_space = action_space
        self.observation_type = observation_type
        self.history_n = int(history_n)
        self.images_to_keep = int(images_to_keep)
        self.coordinate_type = coordinate_type
        self.screen_width = int(screen_width)
        self.screen_height = int(screen_height)
        self.api_backend = api_backend
        self.logger = _DEFAULT_LOGGER

        self.responses: list[str] = []
        self.action_summaries: list[str] = []
        self.actions: list[list[str]] = []
        self.screenshot_b64_history: list[str] = []
        self.last_messages: list[dict[str, Any]] = []

    def reset(self, _logger=None, vm_ip=None):
        if _logger is not None:
            self.logger = _logger
        self.responses = []
        self.action_summaries = []
        self.actions = []
        self.screenshot_b64_history = []
        self.last_messages = []

    def predict(self, instruction: str, obs: Dict[str, Any]) -> Tuple[str, List[str]]:
        try:
            screenshot_b64, (width, height) = encode_jpeg95(
                obs.get("screenshot") if isinstance(obs, dict) else None
            )
        except InvalidScreenshotError as exc:
            raise InvalidScreenshotError(f"[gui_step {len(self.responses) + 1}] {exc}") from exc
        self.screen_width = width
        self.screen_height = height
        messages = self._build_messages(instruction, screenshot_b64)
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

        action_sentence, pyautogui_code = self.parse_response(
            response,
            original_width=width,
            original_height=height,
        )
        self.logger.info("MAGE action sentence: %s", action_sentence)
        self.logger.info("MAGE pyautogui code: %s", pyautogui_code)

        self.responses.append(response)
        self.action_summaries.append(action_sentence or _summarize_actions(pyautogui_code))
        self.actions.append(pyautogui_code)
        self.screenshot_b64_history.append(screenshot_b64)
        return response, pyautogui_code

    def _build_messages(self, instruction: str, current_screenshot_b64: str) -> list[dict[str, Any]]:
        current_step = len(self.responses) + 1
        start_step = max(1, current_step - self.history_n)
        previous_actions = []
        for step in range(1, start_step):
            summary = self.action_summaries[step - 1] if step - 1 < len(self.action_summaries) else ""
            previous_actions.append(f"Step {step}: {summary}")
        previous_actions_str = "\n".join(previous_actions) if previous_actions else "None"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": MAGE_GUI_SYSTEM_PROMPT}
        ]

        def screenshot_for_step(step_number: int) -> str | None:
            if step_number == current_step:
                return current_screenshot_b64
            history_index = step_number - 1
            if 0 <= history_index < len(self.screenshot_b64_history):
                return self.screenshot_b64_history[history_index]
            return None

        def should_keep_image(step_number: int) -> bool:
            return step_number > current_step - max(1, self.images_to_keep)

        def initial_user_content(step_number: int) -> Any:
            text = (
                "\nPlease generate the next move according to the UI screenshot, "
                "instruction and previous actions.\n\n"
                f"Instruction: {instruction}\n\n"
                f"Previous actions:\n{previous_actions_str}"
            )
            image_b64 = screenshot_for_step(step_number) if should_keep_image(step_number) else None
            if image_b64:
                return [_data_image_part(image_b64), {"type": "text", "text": text}]
            return "[This screenshot has been collapsed.]" + text

        def tool_response_content(step_number: int) -> Any:
            image_b64 = screenshot_for_step(step_number) if should_keep_image(step_number) else None
            if image_b64:
                return [
                    {"type": "text", "text": "<tool_response>\n"},
                    _data_image_part(image_b64),
                    {"type": "text", "text": "\n</tool_response>"},
                ]
            return "<tool_response>\n[This screenshot has been collapsed.]\n</tool_response>"

        if start_step == current_step:
            messages.append({"role": "user", "content": initial_user_content(current_step)})
            return messages

        messages.append({"role": "user", "content": initial_user_content(start_step)})
        for step in range(start_step, current_step):
            response = self.responses[step - 1]
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": tool_response_content(step + 1)})
        return messages

    def parse_response(
        self,
        response: str,
        original_width: int | None = None,
        original_height: int | None = None,
    ) -> Tuple[str, List[str]]:
        return self._parse_computer_response(
            response,
            original_width=original_width,
            original_height=original_height,
            max_tool_calls=MAX_TOOL_CALLS,
        )

    def _parse_computer_response(
        self,
        response: str,
        original_width: int | None = None,
        original_height: int | None = None,
        *,
        max_tool_calls: int,
        calls: list[dict[str, Any]] | None = None,
    ) -> Tuple[str, List[str]]:
        """Shared ``computer_use`` parser used by MAGE and MAGE Hybrid."""
        if not response or not response.strip():
            return "", []
        action_sentence = _extract_action_sentence(response)
        calls = calls if calls is not None else _extract_tool_calls(response, logger=self.logger)
        if len(calls) > max_tool_calls:
            self.logger.warning(
                "MAGE response produced %d tool_calls; truncating to first %d",
                len(calls),
                max_tool_calls,
            )
        pyautogui_code: list[str] = []
        for call in calls[:max_tool_calls]:
            code = self._convert_tool_call_to_pyautogui(
                call,
                original_width=original_width or self.screen_width,
                original_height=original_height or self.screen_height,
            )
            if code:
                pyautogui_code.append(code)
        if not action_sentence:
            action_sentence = _summarize_actions(pyautogui_code)
        return action_sentence, pyautogui_code

    def _norm_to_px(self, coord: Any, width: int, height: int) -> Optional[tuple[int, int]]:
        if not isinstance(coord, (list, tuple)) or len(coord) != 2:
            return None
        try:
            x = max(0.0, min(999.0, float(coord[0])))
            y = max(0.0, min(999.0, float(coord[1])))
        except Exception:
            return None
        return int(round(x / 999.0 * (width - 1))), int(round(y / 999.0 * (height - 1)))

    def _convert_tool_call_to_pyautogui(
        self,
        call: dict[str, Any],
        *,
        original_width: int,
        original_height: int,
    ) -> Optional[str]:
        call = _replace_string_in_tool_call(call, "osworld-public-evaluation", "password")
        if call.get("name") != "computer_use":
            self.logger.warning("Unsupported MAGE tool name: %s", call.get("name"))
            return None
        args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        action = str(args.get("action") or "").strip()
        try:
            if action == "click":
                xy = self._norm_to_px(args.get("coordinate"), original_width, original_height)
                button = str(args.get("button") or "left").lower()
                if button not in {"left", "right", "middle"}:
                    button = "left"
                if xy is None:
                    return None
                x, y = xy
                return f"import pyautogui\npyautogui.moveTo({x}, {y})\npyautogui.click(button={button!r})"

            if action == "double_click":
                xy = self._norm_to_px(args.get("coordinate"), original_width, original_height)
                if xy is None:
                    return None
                x, y = xy
                return f"import pyautogui\npyautogui.moveTo({x}, {y})\npyautogui.doubleClick()"

            if action == "triple_click":
                xy = self._norm_to_px(args.get("coordinate"), original_width, original_height)
                if xy is None:
                    return None
                x, y = xy
                return f"import pyautogui\npyautogui.moveTo({x}, {y})\npyautogui.click(clicks=3, interval=0.05)"

            if action == "move":
                xy = self._norm_to_px(args.get("coordinate"), original_width, original_height)
                if xy is None:
                    return None
                x, y = xy
                return f"import pyautogui\npyautogui.moveTo({x}, {y})"

            if action == "drag":
                start = self._norm_to_px(args.get("start"), original_width, original_height)
                end = self._norm_to_px(args.get("end"), original_width, original_height)
                if start is None or end is None:
                    return None
                sx, sy = start
                ex, ey = end
                return (
                    f"import pyautogui\npyautogui.moveTo({sx}, {sy})\n"
                    f"pyautogui.dragTo({ex}, {ey}, duration=0.5, button='left')"
                )

            if action == "type":
                return _type_command(str(args.get("text") or ""))

            if action == "keypress":
                keys = args.get("keys") or []
                if not keys and args.get("key"):
                    keys = [args.get("key")]
                if not isinstance(keys, (list, tuple)):
                    if isinstance(keys, str) and "+" in keys:
                        keys = [part.strip() for part in keys.split("+")]
                    else:
                        keys = [keys]
                mapped = [KEY_MAPPING.get(str(k).lower(), str(k).lower()) for k in keys if str(k)]
                if not mapped:
                    return None
                keys_str = ", ".join(repr(k) for k in mapped)
                if len(mapped) > 1:
                    return f"import pyautogui\npyautogui.hotkey({keys_str})"
                return f"import pyautogui\npyautogui.press({keys_str})"

            if action == "scroll":
                xy = self._norm_to_px(args.get("coordinate"), original_width, original_height)
                pos = f", x={xy[0]}, y={xy[1]}" if xy is not None else ""
                scroll_y = _to_int(args.get("scroll_y"), 0)
                scroll_x = _to_int(args.get("scroll_x"), 0)
                commands = ["import pyautogui"]
                if scroll_y:
                    clicks = _norm_scroll_to_clicks(scroll_y)
                    commands.append(f"pyautogui.scroll({clicks}{pos})")
                if scroll_x:
                    clicks = _norm_scroll_to_clicks(scroll_x)
                    commands.append(f"pyautogui.hscroll({clicks}{pos})")
                return "\n".join(commands) if len(commands) > 1 else None

            if action == "wait":
                secs = max(0.1, float(args.get("time") or 1.0))
                return f"import time\ntime.sleep({secs})"

            if action == "screenshot":
                return "import time"

            if action == "terminate":
                status = str(args.get("status") or "success").lower()
                return "FAIL" if status == "failure" else "DONE"
        except Exception:
            self.logger.exception("Failed to convert MAGE action: %s args=%s", action, args)
            return None

        self.logger.warning("Unsupported MAGE action: %s", action)
        return None

    def call_llm(self, payload: dict[str, Any], model: str) -> str:
        raise NotImplementedError("Use MAGEHybridAgent with api_backend='sglang_generate' (AutoProcessor + SGLang POST /generate).")


def _extract_action_sentence(response: str) -> str:
    m = re.search(r"(?im)^\s*Action\s*:\s*(.+?)\s*$", response)
    return m.group(1).strip() if m else ""


def _extract_tool_calls(
    response: str,
    *,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
) -> list[dict[str, Any]]:
    active_logger = logger or _DEFAULT_LOGGER
    calls: list[dict[str, Any]] = []
    for match in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", response, flags=re.DOTALL):
        text = match.group(1).strip()
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            active_logger.error("Failed to parse MAGE tool_call JSON: %s text=%r", exc, text[:300])
            continue
        if isinstance(obj, dict):
            calls.append(obj)
    if not calls:
        for line in response.splitlines():
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and obj.get("name") == "computer_use":
                    calls.append(obj)
    return calls


def _replace_string_in_tool_call(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, dict):
        return {key: _replace_string_in_tool_call(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_string_in_tool_call(item, old, new) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_string_in_tool_call(item, old, new) for item in value)
    return value


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return default


def _norm_scroll_to_clicks(value: int) -> int:
    clicks = int(round(value / SCROLL_NORM_PER_CLICK))
    if clicks == 0:
        clicks = 1 if value > 0 else -1
    return clicks


def _type_command(text: str) -> str:
    if not text:
        return "import time\ntime.sleep(0.1)"
    if text.isascii() and "\n" not in text:
        return f"import pyautogui\npyautogui.typewrite({text!r}, interval=0.03)"
    if text.isascii():
        commands = ["import pyautogui"]
        lines = text.split("\n")
        for idx, line in enumerate(lines):
            if line:
                commands.append(f"pyautogui.typewrite({line!r}, interval=0.03)")
            if idx < len(lines) - 1:
                commands.append("pyautogui.press('enter')")
        return "\n".join(commands)
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return (
        "import base64, time, pyautogui, pyperclip\n"
        f"_text = base64.b64decode('{encoded}').decode('utf-8')\n"
        "pyperclip.copy(_text)\n"
        "time.sleep(0.1)\n"
        "pyautogui.hotkey('ctrl', 'v')\n"
        "time.sleep(0.1)"
    )


def _summarize_actions(actions: list[str]) -> str:
    if not actions:
        return "No executable action"
    first = actions[0]
    if first in {"DONE", "FAIL", "WAIT"}:
        return first
    if "click" in first:
        return "Click the target element."
    if "typewrite" in first or "pyperclip" in first:
        return "Type the requested text."
    if "hotkey" in first or "press" in first:
        return "Press the requested key(s)."
    if "scroll" in first:
        return "Scroll the page."
    if "dragTo" in first:
        return "Drag between the selected points."
    return "Execute the next GUI action."
