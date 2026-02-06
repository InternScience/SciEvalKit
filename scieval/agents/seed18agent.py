import base64
import io
import os
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from .base import AgentBase, EvalSample
from .records import EvalResult, StepResult, ToolCalling

import json
import json5
import re
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union
from qwen_agent.llm.schema import Message
from qwen_agent.utils.utils import build_text_completion_prompt
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
from transformers import AutoTokenizer 
import datetime
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, Message
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools import BaseTool
from qwen_agent.utils.utils import format_as_text_message, merge_generate_cfgs
import time
import asyncio

from scieval.agents.DRAgent.tool_search_mcp_r1 import *
from scieval.agents.DRAgent.tool_visit_mcp_r1 import *
from scieval.agents.DRAgent.prompt import *

TOOL_CLASS = [
    Visit(),
    Search()
]
TOOL_MAP = {tool.name: tool for tool in TOOL_CLASS}


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

def images_to_base64(images: List[Image.Image], format="JPEG") -> List[str]:
    """
    Convert a list of PIL Images to base64-encoded strings.
    """
    encoded_images = []
    for img in images:
        buffer = io.BytesIO()
        img.save(buffer, format=format)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        encoded_images.append(encoded)
    return encoded_images


def extract_step_from_messages(
    messages: Union[List, None],
    previous_steps: Union[List[StepResult], None],
) -> Union[List[StepResult], None]:

    if messages is None:
        return previous_steps

    step_results = previous_steps if previous_steps is not None else []

    # cache for pairing tool call -> tool response
    pending_tool_calls = {}  # tool_call_id -> ToolCalling

    for message in messages:
        role = message.get("role")

        # ---------- system or user message ----------
        if role == "system" or role == "user":
            content = message.get("content", None)
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            step_result = StepResult(
                role=role,
                content=content
            )
            step_results.append(step_result)

        # ---------- assistant message ----------
        elif role == "assistant":
            content = message.get("content", None)
            if isinstance(content, str):
                content = [{"text": content}]
            step_result = StepResult(
                role="assistant",
                content=content
            )

            # record tool calls (intent)
            tool_calls = message.get("tool_calls")
            if tool_calls is not None:
                for tool_call in tool_calls:
                    tool_calling = ToolCalling(
                        tool_call["function"]["name"],
                        tool_call["function"]["arguments"]
                    )
                    pending_tool_calls[tool_call["id"]] = tool_calling
                    step_result.add_tool_calling(tool_calling)

            step_results.append(step_result)

        # ---------- tool response ----------
        elif role == "tool":
            tool_call_id = message.get("tool_call_id")
            tool_response = message.get("content")

            tool_calling = pending_tool_calls.get(tool_call_id)
            if tool_calling is not None:
                tool_calling.add_response(tool_response)

        # ---------- ignore other roles ----------
        else:
            continue

    return step_results


def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")

class Seed18Agent(AgentBase):
    name = "seed18agent"

    def __init__(
        self,
        model_version: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        # llm: Optional[Union[Dict, BaseChatModel]] = None,
        **kwargs,
    ):
        super().__init__(name=self.name, model_version=model_version)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.api_base = api_base or os.environ.get("OPENAI_API_BASE", "")
        self.model_version = model_version or os.environ.get("MODEL_ID", "o3")
        # self.llm_generate_cfg = llm["generate_cfg"]
        # self.llm_local_path = llm["model"]
    
    def sanity_check_output(self, content):
        return "<think>" in content and "</think>" in content
    
    def call_server(self, msgs, max_tries=10):
        
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=600.0,
        )

        base_sleep_time = 1 
        for attempt in range(max_tries):
            try:
                print(f"--- Attempting to call the service, try {attempt + 1}/{max_tries} ---")
                chat_response = client.chat.completions.create(
                    model=self.model_version,
                    messages=msgs,
                    temperature=0.6,
                    top_p=0.95,
                    max_tokens=32768,
                    presence_penalty=1.1,
                    tools=TOOLS_DEEPSEEK_V32,
                )
                content = chat_response.choices[0].message.content   
                return chat_response.choices[0].message

            except (APIError, APIConnectionError, APITimeoutError) as e:
                print(f"Error: Attempt {attempt + 1} failed with an API or network error: {e}")
            except Exception as e:
                print(f"Error: Attempt {attempt + 1} failed with an unexpected error: {e}")

            if attempt < max_tries - 1:
                sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, 30) 
                
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print("Error: All retry attempts have been exhausted. The call has failed.")
        
        return f"vllm server error!!!"
    
    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        if tool_name in TOOL_MAP:
            tool_args["params"] = tool_args
            if "python" in tool_name.lower():
                result = TOOL_MAP['PythonInterpreter'].call(tool_args)
            if tool_name == "parse_file":
                params = {"files": tool_args["files"]}
                
                raw_result = asyncio.run(TOOL_MAP[tool_name].call(params))
                result = raw_result

                if not isinstance(raw_result, str):
                    result = str(raw_result)
            else:
                raw_result = TOOL_MAP[tool_name].call(tool_args, **kwargs)
                result = raw_result
            return result

        else:
            return f"Error: Tool {tool_name} not found"
        
    def count_tokens(self, messages):
        return 0
    
    def run_sample(self, question, images = None):
        start_time = time.time()
        # MODIFIED: DEEPSEEK system prompt
        system_prompt = SYSTEM_PROMPT_DEEPSEEK_V32
        cur_date = today_date()
        system_prompt = system_prompt + str(cur_date)
        if images is not None:
            content_input = [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{images}"
                    }
                }
            ]
        else:
            content_input = question
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content_input}]
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round = 0
        while num_llm_calls_available > 0:
            # Check whether time is reached
            if time.time() - start_time > 150 * 60:  # 150 minutes in seconds
                prediction = 'No answer found after 2h30mins'
                termination = 'No answer found after 2h30mins'
                result = {
                    "question": question,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination
                }
                return result
            round += 1
            num_llm_calls_available -= 1
            response = self.call_server(messages)
            if isinstance(response, str) and response == "vllm server error!!!":
                result = {
                    "question": question,
                    "messages": messages,
                    "prediction": "LLM server error",
                    "termination": "LLM server error"
                }
                return result
            
            reasoning_content = response.reasoning_content.strip()
            # content = response.content.strip()
            content = response.content.strip() if response.content else None
            tool_calls = response.tool_calls
            print(f"Round {round}\n{reasoning_content=}\n{content=}\n{tool_calls=}")

            messages.append({
                "role": "assistant",
                "reasoning_content": reasoning_content,
                "content": content,
                "tool_calls": [{"type": tool_call.type, "id": tool_call.id, "function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments}} for tool_call in tool_calls] if tool_calls is not None else None,
            })

            if tool_calls is not None:
                for tool in tool_calls:
                    try:
                        tool_name = tool.function.name
                        tool_args = json5.loads(tool.function.arguments)
                        tool_response = self.custom_call_tool(tool_name, tool_args)
                    except Exception as e:
                        tool_response = 'Error: ' + str(e)
                    messages.append({"role": "tool", "tool_call_id": tool.id, "content": tool_response})
            else:
                break
            
            if content is not None:
                if ('<answer>' in content and '</answer>' in content):
                    termination = 'answer'
                    break

            # if num_llm_calls_available <= 0 and not ('<answer>' in content and '</answer>' in content):
            if num_llm_calls_available <= 0:
                messages[-1]['content'] = 'Sorry, the number of llm calls exceeds the limit.'

            max_tokens = 128 * 1024
            token_count = self.count_tokens(messages)
            print(f"round: {round}, token count: {token_count}")

            if token_count > max_tokens:
                print(f"Token quantity exceeds the limit: {token_count} > {max_tokens}")
                
                messages[-1]['content'] = "You have now reached the maximum context length you can handle. You should stop making tool calls and, based on all the information above, think again and provide what you consider the most likely answer in the following format:<think>your final thinking</think>\n<answer>your answer</answer>"
                response = self.call_server(messages)
                content = response.content.strip() if response.content else None
                messages.append({"role": "assistant", "content": content})
                if content is not None:
                    if '<answer>' in content and '</answer>' in content:
                        prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
                        termination = 'generate an answer as token limit reached'
                else:
                    prediction = messages[-1]['content']
                    termination = 'format error: generate an answer as token limit reached'
                result = {
                    "question": question,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination
                }
                return result

        extracted_answer = messages[-1]['content'].strip()
        if '<answer>' in extracted_answer and '</answer>' in extracted_answer:
            prediction = extracted_answer.split('<answer>')[-1].split('</answer>')[0]
            termination = 'answer'
        else:
            prediction = 'No answer found.'
            termination = 'answer not found'
            if num_llm_calls_available == 0:
                termination = 'exceed available llm calls'

        result = {
            "question": question,
            "messages": messages,
            "prediction": prediction,
            "termination": termination
        }
        return result


    def run(self, sample: EvalSample) -> EvalResult:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for Seed18Agent.")
        if sample.images:
            images = _load_images(sample.images)
            base64_images = images_to_base64(images)
        else:
            base64_images = None
        result = self.run_sample(question=sample.prompt, images=base64_images)
        eval_result = EvalResult(success=result['termination'] == "answer", final_answer=result['prediction'])

        steps_skeleton = extract_step_from_messages(result['messages'], None)

        if steps_skeleton:
            for step in steps_skeleton:
                eval_result.add_step(step)

        return eval_result
