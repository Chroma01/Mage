"""Standalone OSWorld-Verified runner for Mage-VL-Cowork (MAGEHybridAgent).

Drop this script into an OSWorld-Verified checkout (under ``run/`` or repository root)
alongside ``mm_agents/mage_gui_agent.py``, ``mm_agents/mage_hybrid_agent.py``, and
``desktop_env/pydeps_bundle.py``.

Example usage:
    python run/run_multienv_mage_cowork.py \
        --headless \
        --provider_name docker \
        --model Mage-VL-Cowork \
        --processor_model_name /path/to/Mage-VL-Cowork \
        --base_url http://127.0.0.1:30001 \
        --test_all_meta_path evaluation_examples/test_nogdrive.json \
        --max_steps 50 \
        --history_n 3 \
        --images_to_keep 3 \
        --temperature 0.7 \
        --top_p 0.8 \
        --max_tokens 4096 \
        --sleep_after_execution 3.0 \
        --num_envs 4 \
        --result_dir ./results
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import signal
import sys
import time
from multiprocessing import Manager, Process, current_process
from typing import Any, Dict, List

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists(os.path.join(_proj_root, "desktop_env")):
    _proj_root = os.path.abspath(os.getcwd())
sys.path.insert(0, _proj_root)
sys.path.insert(0, os.path.join(_proj_root, "run"))

from desktop_env.desktop_env import DesktopEnv
from mm_agents.mage_hybrid_agent import MAGEHybridAgent

try:
    from lib_multienv_common import distribute_tasks, get_result, get_unfinished
except ImportError:
    def distribute_tasks(test_all_meta: dict) -> List[tuple]:
        all_tasks = []
        for domain, examples in test_all_meta.items():
            for example_id in examples:
                all_tasks.append((domain, example_id))
        return all_tasks

    def get_unfinished(
        action_space: str,
        use_model: str,
        observation_type: str,
        result_dir: str,
        total_file_json: dict,
    ) -> dict:
        target_dir = os.path.join(result_dir, action_space, observation_type, use_model)
        if not os.path.exists(target_dir):
            return total_file_json
        finished: Dict[str, List[str]] = {}
        for domain in os.listdir(target_dir):
            domain_path = os.path.join(target_dir, domain)
            if not os.path.isdir(domain_path):
                continue
            finished[domain] = []
            for example_id in os.listdir(domain_path):
                example_path = os.path.join(domain_path, example_id)
                result_file = os.path.join(example_path, "result.txt")
                if os.path.isdir(example_path) and os.path.exists(result_file) and os.path.getsize(result_file) > 0:
                    finished[domain].append(example_id)
        for domain, examples in finished.items():
            if domain in total_file_json:
                total_file_json[domain] = [x for x in total_file_json[domain] if x not in examples]
        return total_file_json

    def get_result(
        action_space: str,
        use_model: str,
        observation_type: str,
        result_dir: str,
        total_file_json: dict,
    ):
        target_dir = os.path.join(result_dir, action_space, observation_type, use_model)
        if not os.path.exists(target_dir):
            return None
        all_result = []
        for domain in os.listdir(target_dir):
            domain_path = os.path.join(target_dir, domain)
            if os.path.isdir(domain_path):
                for example_id in os.listdir(domain_path):
                    result_file = os.path.join(domain_path, example_id, "result.txt")
                    if os.path.exists(result_file):
                        try:
                            with open(result_file, "r", encoding="utf-8") as f:
                                all_result.append(float(f.read().strip()))
                        except Exception:
                            all_result.append(0.0)
        if all_result:
            print(f"Current Success Rate: {sum(all_result) / len(all_result) * 100:.2f}% ({len(all_result)} tasks)")
        return all_result

active_environments: list = []
processes: list = []
is_terminating = False

logger = logging.getLogger("desktopenv.experiment")


def _snapshot_agent_step_state(agent: MAGEHybridAgent) -> dict[str, Any]:
    return {
        "responses": list(agent.responses),
        "action_summaries": list(agent.action_summaries),
        "actions": list(agent.actions),
        "screenshot_b64_history": list(agent.screenshot_b64_history),
        "step_inputs": list(agent.step_inputs),
        "tool_names": list(agent.tool_names),
        "last_parsed_tool_name": agent.last_parsed_tool_name,
        "_pending_bash_output": agent._pending_bash_output,
        "_has_pending_bash_output": agent._has_pending_bash_output,
    }


def _restore_agent_step_state(agent: MAGEHybridAgent, state: dict[str, Any]) -> None:
    agent.responses = list(state["responses"])
    agent.action_summaries = list(state["action_summaries"])
    agent.actions = list(state["actions"])
    agent.screenshot_b64_history = list(state["screenshot_b64_history"])
    agent.step_inputs = list(state["step_inputs"])
    agent.tool_names = list(state["tool_names"])
    agent.last_parsed_tool_name = state["last_parsed_tool_name"]
    agent._pending_bash_output = state["_pending_bash_output"]
    agent._has_pending_bash_output = state["_has_pending_bash_output"]


def _write_screenshot(obs: dict[str, Any] | None, path: str) -> str | None:
    screenshot = obs.get("screenshot") if isinstance(obs, dict) else None
    if not screenshot:
        return None
    with open(path, "wb") as f:
        f.write(screenshot)
    return os.path.basename(path)


def run_single_example_mage_hybrid(
    agent: MAGEHybridAgent,
    env: DesktopEnv,
    example: dict[str, Any],
    max_steps: int,
    instruction: str,
    args: argparse.Namespace,
    example_result_dir: str,
    scores: list,
) -> None:
    runtime_logger = logging.getLogger(f"desktopenv.example.{example['id']}")
    runtime_logger.setLevel(logging.DEBUG)
    runtime_logger.addHandler(logging.FileHandler(os.path.join(example_result_dir, "runtime.log"), encoding="utf-8"))

    env.reset(task_config=example)
    agent.reset(runtime_logger, vm_ip=getattr(env, "vm_ip", None))

    if args.initial_wait_seconds > 0:
        time.sleep(args.initial_wait_seconds)

    if args.bootstrap_guest:
        bootstrap_evidence = agent.bootstrap_guest(env, task_config=example)
        with open(os.path.join(example_result_dir, "mage_hybrid_bootstrap.json"), "w", encoding="utf-8") as f:
            json.dump(bootstrap_evidence, f, indent=2, ensure_ascii=False, default=str)

    obs = env._get_obs()
    _write_screenshot(obs, os.path.join(example_result_dir, "step_0.png"))

    done = False
    try:
        env.controller.start_recording()
    except Exception:
        pass

    for step_idx in range(max_steps):
        if done:
            break

        state_snapshot = _snapshot_agent_step_state(agent)
        resamples = 0
        while True:
            response, actions = agent.predict(instruction, obs)
            if actions or resamples >= args.empty_action_retries:
                break
            resamples += 1
            runtime_logger.warning(
                "Step %d parsed 0 actions; resampling %d/%d",
                step_idx + 1,
                resamples,
                args.empty_action_retries,
            )
            _restore_agent_step_state(agent, state_snapshot)

        response_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")
        action_results: list[dict[str, Any]] = []
        reward = 0.0
        info: Any = None

        if not actions:
            runtime_logger.warning("No actions returned at step %d; ending episode.", step_idx + 1)
            entry = {
                "step_num": step_idx + 1,
                "response_timestamp": response_timestamp,
                "response": response,
                "actions": [],
                "action_results": [],
                "reward": 0.0,
                "done": done,
                "info": {"reason": "empty_actions", "empty_action_resamples": resamples},
                "screenshot_file": None,
            }
            with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            break

        if agent.last_parsed_tool_name == "bash":
            for action in actions:
                if hasattr(env, "_step_no"):
                    env._step_no += 1
                if hasattr(env, "action_history"):
                    env.action_history.append(action)
                if hasattr(env, "is_environment_used"):
                    env.is_environment_used = True
            bash_results = agent.execute_bash_actions(
                env.controller,
                actions,
                timeout=args.bash_timeout,
            )
            for idx, (action, bash_res) in enumerate(zip(actions, bash_results)):
                action_results.append(
                    {
                        "action_index": idx,
                        "action": action,
                        "reward": 0.0,
                        "done": False,
                        "info": {
                            "tool_name": "bash",
                            "status": bash_res.get("status"),
                            "returncode": bash_res.get("returncode"),
                            "output": bash_res.get("output"),
                            "error": bash_res.get("error"),
                        },
                    }
                )
            obs = env._get_obs()
            info = {"tool_name": "bash", "call_count": len(actions)}
        else:
            for idx, action in enumerate(actions):
                is_last = idx == len(actions) - 1
                try:
                    obs, reward, done, info = env.step(
                        action,
                        args.sleep_after_execution,
                        skip_screenshot=not is_last,
                    )
                except TypeError:
                    obs, reward, done, info = env.step(
                        action,
                        args.sleep_after_execution,
                    )
                action_results.append(
                    {
                        "action_index": idx,
                        "action": action,
                        "reward": reward,
                        "done": done,
                        "info": info,
                    }
                )
                if done:
                    break

        screenshot_file = f"step_{step_idx + 1}_{response_timestamp}.png"
        saved_screenshot = _write_screenshot(obs, os.path.join(example_result_dir, screenshot_file))
        entry = {
            "step_num": step_idx + 1,
            "response_timestamp": response_timestamp,
            "response": response,
            "actions": actions,
            "action_results": action_results,
            "reward": reward,
            "done": done,
            "info": info,
            "screenshot_file": saved_screenshot,
        }
        with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    time.sleep(20)
    result = float(env.evaluate())
    logger.info("Result for %s: %.2f", example["id"], result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")
    try:
        env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))
    except Exception:
        pass


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mage-VL-Cowork evaluation on OSWorld-Verified")
    parser.add_argument("--path_to_vm", type=str, default=None)
    parser.add_argument("--headless", action="store_true", help="Run VM in headless mode")
    parser.add_argument("--action_space", type=str, default="pyautogui")
    parser.add_argument("--observation_type", choices=["screenshot"], default="screenshot")
    parser.add_argument("--sleep_after_execution", type=float, default=3.0)
    parser.add_argument("--initial_wait_seconds", type=float, default=60.0)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--history_n", type=int, default=3)
    parser.add_argument("--images_to_keep", type=int, default=3)
    parser.add_argument("--bash_timeout", type=int, default=300)
    parser.add_argument("--empty_action_retries", type=int, default=2)
    parser.add_argument("--bootstrap_guest", action="store_true", default=True)
    parser.add_argument("--no_bootstrap_guest", action="store_false", dest="bootstrap_guest")
    parser.add_argument("--bootstrap_python_deps", action="store_true", default=True)
    parser.add_argument("--no_bootstrap_python_deps", action="store_false", dest="bootstrap_python_deps")
    parser.add_argument("--disable_idle_lock", action="store_true", default=True)
    parser.add_argument("--no_disable_idle_lock", action="store_false", dest="disable_idle_lock")
    parser.add_argument("--model", type=str, default="Mage-VL-Cowork")
    parser.add_argument("--processor_model_name", type=str, default="microsoft/Mage-VL-Cowork")
    parser.add_argument("--api_backend", choices=["sglang_generate"], default="sglang_generate")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--base_url", type=str, default="http://127.0.0.1:30001")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--test_config_base_dir", type=str, default="evaluation_examples")
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument("--test_all_meta_path", type=str, default="evaluation_examples/test_nogdrive.json")
    parser.add_argument("--result_dir", type=str, default="./results")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO")
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--provider_name", type=str, default="docker", choices=["aws", "virtualbox", "vmware", "docker", "azure", "aliyun"])
    parser.add_argument("--client_password", type=str, default="")
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    return parser.parse_args()


def run_env_tasks(task_queue, args: argparse.Namespace, shared_scores: list) -> None:
    env = None
    try:
        screen_size = (args.screen_width, args.screen_height)
        snapshot_name = "init_state"
        if args.provider_name == "aws":
            from desktop_env.providers.aws.manager import IMAGE_ID_MAP
            snapshot_name = IMAGE_ID_MAP[args.region].get(screen_size, IMAGE_ID_MAP[args.region][(1920, 1080)])

        env = DesktopEnv(
            path_to_vm=args.path_to_vm,
            action_space=args.action_space,
            provider_name=args.provider_name,
            region=args.region,
            snapshot_name=snapshot_name,
            screen_size=screen_size,
            headless=args.headless,
            os_type="Ubuntu",
            require_a11y_tree=False,
            enable_proxy=False,
            client_password=args.client_password,
        )
        active_environments.append(env)
        agent = MAGEHybridAgent(
            model=args.model,
            processor_model_name=args.processor_model_name,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            temperature=args.temperature,
            action_space=args.action_space,
            observation_type=args.observation_type,
            history_n=args.history_n,
            images_to_keep=args.images_to_keep,
            screen_width=args.screen_width,
            screen_height=args.screen_height,
            api_backend=args.api_backend,
            bash_timeout=args.bash_timeout,
            max_steps=args.max_steps,
            bootstrap_guest=args.bootstrap_guest,
            bootstrap_python_deps=args.bootstrap_python_deps,
            disable_idle_lock=args.disable_idle_lock,
        )
        logger.info("Process %s started with MAGEHybridAgent.", current_process().name)
        while True:
            try:
                item = task_queue.get(timeout=5)
            except Exception:
                break
            domain, example_id = item
            try:
                config_file = os.path.join(args.test_config_base_dir, f"examples/{domain}/{example_id}.json")
                with open(config_file, "r", encoding="utf-8") as f:
                    example = json.load(f)
                example_result_dir = os.path.join(
                    args.result_dir, args.action_space, args.observation_type, args.model, domain, example_id
                )
                os.makedirs(example_result_dir, exist_ok=True)
                run_single_example_mage_hybrid(
                    agent, env, example, args.max_steps, example["instruction"], args, example_result_dir, shared_scores
                )
            except Exception as exc:
                logger.error("Task error in %s (%s/%s): %s", current_process().name, domain, example_id, exc, exc_info=True)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def signal_handler(signum, frame) -> None:
    global is_terminating
    if is_terminating:
        return
    is_terminating = True
    for env in active_environments:
        try:
            env.close()
        except Exception:
            pass
    for p in processes:
        if p.is_alive():
            try:
                p.terminate()
            except Exception:
                pass
    sys.exit(0)


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = config()
    if args.base_url:
        os.environ["SGLANG_BASE_URL"] = args.base_url
        os.environ["OPENAI_BASE_URL"] = (
            args.base_url if args.base_url.rstrip("/").endswith("/v1") else f"{args.base_url.rstrip('/')}/v1"
        )
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if args.processor_model_name:
        os.environ["MAGE_PROCESSOR_PATH"] = args.processor_model_name

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
    )
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
        test_all_meta = json.load(f)
    if args.domain != "all":
        test_all_meta = {args.domain: test_all_meta[args.domain]}

    test_file_list = get_unfinished(
        args.action_space, args.model, args.observation_type, args.result_dir, test_all_meta
    )
    get_result(args.action_space, args.model, args.observation_type, args.result_dir, test_all_meta)

    all_tasks = distribute_tasks(test_file_list)
    logger.info("Total remaining tasks to run: %d", len(all_tasks))
    with Manager() as manager:
        shared_scores = manager.list()
        task_queue = manager.Queue()
        for item in all_tasks:
            task_queue.put(item)
        for i in range(args.num_envs):
            p = Process(target=run_env_tasks, args=(task_queue, args, shared_scores), name=f"EnvProcess-{i+1}")
            p.daemon = True
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        scores = list(shared_scores)
    if scores:
        logger.info("Finished batch. Average score: %.2f%% (%d tasks)", sum(scores) / len(scores) * 100.0, len(scores))


if __name__ == "__main__":
    main()
