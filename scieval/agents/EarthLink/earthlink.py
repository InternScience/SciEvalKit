import os
import json
import logging
import asyncio
from typing import Any, Dict, List, Optional, Union, Literal
from ..base import AgentBase, EvalSample
from ..records import EvalResult, StepResult, ToolCalling

from . import config as CFG
from .agent.plan import chat_plan_agent


def get_run_logger(name: str, log_path: str):

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def normalize(obj):
    if isinstance(obj, dict):
        return {k: normalize(obj[k]) for k in sorted(obj)}
    elif isinstance(obj, list):
        return [normalize(x) for x in obj]
    else:
        return obj


def deep_equal(a, b):
    return normalize(a) == normalize(b)


def process_agent_logs(log_dir) -> List[StepResult]:

    all_steps = []

    def get_log_by_prefix(prefix: str):
        log_files = [f for f in os.listdir(log_dir) if f.startswith(prefix)]
        log_files = sorted(log_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        logs = []
        for file in log_files:
            with open(os.path.join(log_dir, file), "r", encoding='utf-8') as f:
                logs.append(json.load(f))
        return logs
    
    def process_one_log(messages, role_prefix: str = "") -> List[StepResult]:
        steps = []
        idx = 0
        while idx < len(messages):
            msg = messages[idx]
            content = msg["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]

            if msg["type"] == "system":
                steps.append(StepResult(role=role_prefix + "system", content=content))
                idx += 1
            elif msg["type"] == "human":
                steps.append(StepResult(role=role_prefix + "user", content=content))
                idx += 1
            elif msg["type"] == "ai":
                tool_calls = msg.get("tool_calls", [])
                step = StepResult(role=role_prefix + "assistant", content=content)
                idx += 1
                while idx < len(messages) and messages[idx]["type"] == "tool":
                    tool_msg = messages[idx]
                    tool_call_id = tool_msg["tool_call_id"]
                    for call in tool_calls:
                        if call["id"] == tool_call_id:
                            tool_name = call["name"]
                            tool_input = call["args"]
                            tool_output = tool_msg["content"]
                            tool_calling = ToolCalling(
                                tool_name=tool_name,
                                tool_input=tool_input,
                                tool_output=tool_output,
                            )
                            step.add_tool_calling(tool_calling)
                            break
                    idx += 1
                steps.append(step)
            else:
                assert False, f"Unknown message type: {msg['type']}"

        return steps
    
    def remove_last_round_msg(last_round, cur_round):
        last_msg = last_round[-1]
        for i in range(len(cur_round)):
            if deep_equal(last_msg, cur_round[i]):
                break

        return cur_round[i+1:]

    plan_logs = get_log_by_prefix("plan_agent_")
    for i, plan_log in enumerate(plan_logs):
        steps = process_one_log(plan_log, f"plan_agent_{i}_")
        all_steps.extend(steps)

    plan_aggr_logs = get_log_by_prefix("plan_aggregation_agent_round_")
    data_check_logs = get_log_by_prefix("data_check_agent_round_")

    for round in range(len(plan_aggr_logs)):
        if round > 0:
            cur_round_msgs = remove_last_round_msg(plan_aggr_logs[round-1], plan_aggr_logs[round])
        else:
            cur_round_msgs = plan_aggr_logs[round]
        steps = process_one_log(cur_round_msgs, f"plan_aggregation_agent_round_{round}_")
        all_steps.extend(steps)

        check_steps = process_one_log(data_check_logs[round], f"data_check_agent_round_{round}_")
        all_steps.extend(check_steps)

    return all_steps


class EarthLinkAgent(AgentBase):
    name = "EarthLink"

    def __init__(
        self,
        output_dir: str,
        mode: Literal["plan", "full"] = "plan",
        model_version: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        embedding_model: Optional[str] = "Qwen/Qwen3-Embedding-8B", # only support Qwen3-Embedding-8B for now
        embedding_api_key: Optional[str] = None,
        embedding_api_base: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(self.name, model_version, **kwargs)
        self.output_dir = output_dir
        self.mode = mode
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.api_base = api_base or os.environ.get("OPENAI_API_BASE", "")
        self.model_version = model_version or os.environ.get("MODEL_ID", "gpt-5-mini")

        self.embedding_model = embedding_model
        self.embedding_api_key = embedding_api_key or api_key
        self.embedding_api_base = embedding_api_base or api_base

        CFG.API_KEY = self.api_key
        CFG.BASE_URL = self.api_base
        CFG.EMBEDDING_MODEL = self.embedding_model
        CFG.EMBEDDING_API_KEY = self.embedding_api_key
        CFG.EMBEDDING_BASE_URL = self.embedding_api_base
        CFG.DEFAULT_MODEL = self.model_version
        CFG.TAVILY_API_KEY = tavily_api_key or os.environ.get("TAVILY_API_KEY", None)

        for key, value in kwargs.items():
            if hasattr(CFG, key.upper()):
                setattr(CFG, key.upper(), value)

        if self.mode not in ["plan", "full"]:
            raise ValueError("mode must be either 'plan' or 'full'")
        elif self.mode == "full":
            raise NotImplementedError("Full mode is not implemented yet.")

    def run(self, sample: EvalSample) -> EvalResult:
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "agent_logs"), exist_ok=True)

        run_info = {
            "root": self.output_dir,
            "user_request": sample.prompt,
            "logger": get_run_logger("EarthLink_run", os.path.join(self.output_dir, "run.log")),
        }

        try:
            if self.mode == "plan":
                run_info = asyncio.run(chat_plan_agent(run_info))

                result = EvalResult(
                    success=True,
                    final_answer=run_info["experiment_plan"]
                )
                result.steps = process_agent_logs(os.path.join(self.output_dir, "agent_logs"))
                return result
            else:
                raise NotImplementedError("Full mode is not implemented yet.")
        except Exception as e:
            return EvalResult(success=False, final_answer=f"Agent failed with error: \n{e}")
