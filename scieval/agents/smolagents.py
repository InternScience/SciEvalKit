import base64
import io
import os
from typing import Any, Dict, List, Optional, Union

from PIL import Image
from smolagents import (
    ActionStep,
    ChatMessage,
    OpenAIServerModel,
    PlanningStep,
    TaskStep,
    ToolCallingAgent,
)
from smolagents import PythonInterpreterTool, VisitWebpageTool, WikipediaSearchTool

from .base import AgentBase, EvalSample
from .records import EvalResult, StepResult, ToolCalling


def _decode_data_image(uri: str) -> Image.Image:
    _, b64data = uri.split(",", 1)
    raw = base64.b64decode(b64data)
    return Image.open(io.BytesIO(raw))


def _load_images(items: List[Any]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for item in items:
        if isinstance(item, Image.Image):
            images.append(item)
        elif isinstance(item, str):
            if item.startswith("data:image"):
                images.append(_decode_data_image(item))
            elif os.path.exists(item):
                images.append(Image.open(item))
    return images


def extract_step_from_messages(
    messages: Union[ChatMessage, List[ChatMessage]],
    previous_steps: Union[List[StepResult], None],
) -> Union[List[StepResult], None]:
    if messages is None:
        return None

    if not isinstance(messages, list):
        messages = [messages]

    step_results = previous_steps if previous_steps is not None else []

    for message in messages:
        if message.role == "tool-call":
            continue

        if message.role == "tool-response":
            info = message.content[0]["text"]
            step_result = StepResult(role="assistant", content="")
            tool_calling = ToolCalling(tool_name="unknown", tool_input="unknown")
            tool_calling.add_response(info)
            step_result.add_tool_calling(tool_calling)
            step_results.append(step_result)
            continue

        step_result = StepResult(role=message.role, content=message.content)

        if message.tool_calls is not None:
            for tool_call in message.tool_calls:
                tool_calling = ToolCalling(tool_call.function.name, tool_call.function.arguments)
                step_result.add_tool_calling(tool_calling)

        step_results.append(step_result)

    return step_results


def get_function_calls(messages: Union[ChatMessage, List[ChatMessage]]) -> List[List[dict]]:
    if not isinstance(messages, list):
        messages = [messages]

    function_calls = []
    for message in messages:
        if message is None or message.tool_calls is None:
            continue
        sub_function_calls = []
        for tool_call in message.tool_calls:
            sub_function_calls.append(
                {"name": tool_call.function.name, "arguments": tool_call.function.arguments}
            )
        function_calls.append(sub_function_calls)

    return function_calls


class SmolAgentsAgent(AgentBase):
    name = "smolagents"

    def __init__(
        self,
        model_version: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=self.name, model_version=model_version)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.api_base = api_base or os.environ.get("OPENAI_API_BASE", "")
        self.model_version = model_version or os.environ.get("MODEL_ID", "o3")

    def run(self, sample: EvalSample) -> EvalResult:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for smolagents.")

        model = OpenAIServerModel(
            model_id=self.model_version, api_key=self.api_key, api_base=self.api_base
        )
        agent = ToolCallingAgent(
            tools=[VisitWebpageTool(), WikipediaSearchTool(), PythonInterpreterTool()],
            model=model,
            verbosity_level=-1,
        )

        images = _load_images(sample.images)
        result = agent.run(task=sample.prompt, images=images, stream=False, return_full_result=True)
        eval_result = EvalResult(success=result.state == "success", final_answer=result.output)

        last_step = None
        for i in range(len(agent.memory.steps) - 1, -1, -1):
            if agent.memory.steps[i].error is None:
                last_step = agent.memory.steps[i]
                break

        steps_skeleton = None
        if last_step is not None:
            steps_skeleton = extract_step_from_messages(last_step.model_input_messages, None)
            steps_skeleton = extract_step_from_messages(last_step.model_output_message, steps_skeleton)

        function_calls = []
        for step in agent.memory.steps:
            if isinstance(step, TaskStep):
                continue
            if isinstance(step, (ActionStep, PlanningStep)):
                function_calls.extend(get_function_calls(step.model_output_message))

        unknown_function_idx = 0
        if steps_skeleton:
            for step in steps_skeleton:
                for tc in step.tool_calling:
                    if tc.tool_name == "unknown" and tc.tool_input == "unknown":
                        if unknown_function_idx < len(function_calls):
                            tc.tool_name = [f["name"] for f in function_calls[unknown_function_idx]]
                            tc.tool_input = [
                                f["arguments"] for f in function_calls[unknown_function_idx]
                            ]
                        unknown_function_idx += 1
                eval_result.add_step(step)

        return eval_result
