import os
import json
import httpx
import asyncio
import logging
from langchain.chat_models import init_chat_model, BaseChatModel
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from .common import color_text, encode_file_base64, FlushableLogger
from .. import config as CFG


class Agent:
    def __init__(
        self,
        name: str = "DefaultAgent",
        model_settings: dict = {
            "model": "gpt-5",
            "model_provider": "openai",
        },
        system_prompt: str = "",
        tools: list = [],
        tool_runtime_args: dict = {},
        http_proxy: str = None,
        logger: logging.Logger = None,
        verbose: bool = False,
        log_input: bool = False,
        max_retries: int = 3,
        max_agent_iterations: int = 15,
        handle_tool_calls: bool = True,
    ):
        self.name = name
        self.model_settings = model_settings
        self.system_prompt = system_prompt
        self.tools = tools
        self.http_proxy = http_proxy
        self.logger = logger

        if logger is not None:
            verbose = True
        self.verbose = verbose
        self.log_input = log_input

        self.max_retries = max_retries
        self.max_agent_iterations = max_agent_iterations
        self.handle_tool_calls = handle_tool_calls

        self.llm = self.initialize_llm()
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.tool_runtime_args = tool_runtime_args

        self.messages = []
        if self.system_prompt:
            self.messages.append(SystemMessage(content=self.system_prompt))

        self.event_counter = 0
        self.msg_mark_idx = None

    def initialize_llm(self):
        params = self.model_settings.copy()
        params["model"] = self.model_settings.get("model", CFG.DEFAULT_MODEL)
        params["model_provider"] = self.model_settings.get("model_provider", CFG.DEFAULT_MODEL_PROVIDER)

        params["api_key"] = self.model_settings.get("api_key", CFG.API_KEY)
        params["base_url"] = self.model_settings.get("base_url", CFG.BASE_URL)

        if self.http_proxy:
            params["http_client"] = httpx.Client(proxy=self.http_proxy)
            params["http_async_client"] = httpx.AsyncClient(proxy=self.http_proxy)
        
        llm: BaseChatModel = init_chat_model(**params)

        if self.tools:
            llm = llm.bind_tools(self.tools)
        return llm

    def log(self, message: str, color: str = None, style: str = None, flush: bool = False, verbose: bool = False, count: bool = True):

        if count:
            self.event_counter += 1

        verbose = verbose or self.verbose
        if not verbose:
            return
        
        if color or style:
            message = color_text(message, color=color or "default", style=style or "normal")
        
        if self.logger:
            self.logger.info(message)
            if flush:
                for handler in self.logger.handlers:
                    handler.flush()
        else:
            print(message, flush=flush)
    
    def clear_messages(self):
        self.messages = []
        if self.system_prompt:
            self.messages.append(SystemMessage(content=self.system_prompt))

    def save_messages(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([msg.model_dump() for msg in self.messages], f, ensure_ascii=False, indent=4)

    def remove_messages_after_mark(self, keep_last_n_response: int = 1):
        if self.msg_mark_idx is not None:
            remove_start_idx = self.msg_mark_idx + 1
            remove_end_idx = len(self.messages)
            for idx in range(len(self.messages) - 1, self.msg_mark_idx, -1):
                if isinstance(self.messages[idx], AIMessage):
                    keep_last_n_response -= 1
                if keep_last_n_response == 0:
                    remove_end_idx = idx
                    break
            del self.messages[remove_start_idx:remove_end_idx]

    async def _get_response(self) -> AIMessage:
        cur_try = 0

        while True:
            try:
                response = await self.llm.ainvoke(self.messages)
                return response

            except Exception as e:
                cur_try += 1
                self.log(f"{self.name}: Error when getting response [{cur_try}/{self.max_retries}]: {e}",
                         color='r', flush=True, verbose=True, count=False)

                if cur_try >= self.max_retries:
                    raise RuntimeError(f"{self.name}: Max retries exceeded when getting response. Error: {e}")
                
                await asyncio.sleep(3)

    async def _invoke_tool(self, tool_call):
        name = tool_call['name']
        args = tool_call['args']
        tool_log = f"Invoking `{name}` with `{args}`"
        self.log(
            f"({color_text(self.name, 'g')}) {self.event_counter}: Agent took action:\n{color_text(tool_log, 'y')}", flush=True
        )
        args = args.copy()
        args.update(self.tool_runtime_args.get(name, {}))
        
        tool = self.tools_by_name.get(name)
        tool_response = await tool.ainvoke(args)

        self.log(
            f"({color_text(self.name, 'g')}) {self.event_counter}: Tool `{name}` returned:\n{color_text(tool_response, 'y')}\n", flush=True
        )

        return tool_call['id'], tool_response

    async def chat(self,
                   input_msg: str | HumanMessage,
                   handle_tool_calls: bool = None,
                   input_images: str | list[str] = None,
                   input_files: str | list[str] = None,
                   log_input: bool = False,
                   mark_input: bool = False) -> AIMessage:
    
        if log_input or self.log_input:
            input_text = input_msg if isinstance(input_msg, str) else input_msg.content
            if input_images is not None:
                input_text += f"\n[Input Images: {input_images}]"
            if input_files is not None:
                input_text += f"\n[Input Files: {input_files}]"
            self.log(f"({color_text(self.name, 'g')}) {self.event_counter}: Agent started with input:\n{color_text(input_text, 'b')}\n", flush=True)
        else:
            self.log(f"({color_text(self.name, 'g')}) {self.event_counter}: Agent started", flush=True)

        input_images_files = []
        if input_images is not None:
            img_type_map = {'jpg': 'jpeg', 'jpeg': 'jpeg', 'png': 'png'}
            
            input_images = [input_images] if isinstance(input_images, str) else input_images
            input_images_files.extend([
                {
                    "type": "image",
                    "source_type": "base64",
                    "mime_type": f"image/{img_type_map[img_path.split('.')[-1].lower()]}",
                    "data": encode_file_base64(img_path),
                }
                for img_path in input_images
            ])
        if input_files is not None:
            input_files = [input_files] if isinstance(input_files, str) else input_files

            # only support pdf files
            all_pdf = all([file_path.split('.')[-1].lower() == 'pdf' for file_path in input_files])
            if not all_pdf:
                raise ValueError(f"Only PDF files are supported for input_files.")

            input_images_files.extend([
                {
                    "type": "file",
                    "source_type": "base64",
                    "mime_type": "application/pdf",
                    "data": encode_file_base64(file_path),
                    "filename": os.path.basename(file_path),
                }
                for file_path in input_files
            ])
            
        if isinstance(input_msg, str):
            if input_images_files:
                input_msg = HumanMessage(
                    input_images_files + [{
                        "type": "text",
                        "text": input_msg
                    }]
                )
            else:
                input_msg = HumanMessage(content=input_msg)
        elif isinstance(input_msg, HumanMessage):
            if input_images_files:
                if not isinstance(input_msg.content, list):
                    raise ValueError(f"When input_images or input_files are provided, input_msg must be a string or a HumanMessage with content as a list.")
                input_msg.content = input_msg.content + input_images_files
        else:
            raise ValueError(f"input_msg must be a string or a HumanMessage.")

        self.messages.append(input_msg)

        if mark_input:
            self.msg_mark_idx = len(self.messages) - 1

        cur_iteration = 0
        if handle_tool_calls is None:
            handle_tool_calls = self.handle_tool_calls

        while True:
            response = await self._get_response()
            self.messages.append(response)

            if response.content:
                self.log(
                    f"({color_text(self.name, 'g')}) {self.event_counter}: Agent response:\n{color_text(response.content, 'm')}\n", flush=True
                )

            if not response.tool_calls or not handle_tool_calls:
                break
                
            tasks = []
            for tool_call in response.tool_calls:
                tasks.append(
                    asyncio.create_task(self._invoke_tool(tool_call))
                )                
            tool_responses = await asyncio.gather(*tasks)
            for tool_call_id, tool_response in tool_responses:
                self.messages.append(
                    ToolMessage(content=tool_response, tool_call_id=tool_call_id)
                )

            cur_iteration += 1
            if cur_iteration >= self.max_agent_iterations:
                raise RuntimeError(f"{self.name}: Max agent iterations ({self.max_agent_iterations}) reached.")

        return response