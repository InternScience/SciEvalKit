import io
import os
import re
import json
import uuid
import base64
import asyncio
import inspect
import logging
import cairosvg
import tiktoken
import threading
import contextlib
import numpy as np
from json import JSONEncoder
from cachetools import LRUCache
from functools import wraps, partial
from contextvars import copy_context
from concurrent.futures import Executor
from typing import Optional, Callable, ParamSpec, TypeVar, cast



TIKTOKEN_ENCODING = tiktoken.get_encoding("o200k_base") # used for token counting


def get_run_logger(name: str, log_path: str = ""):

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if log_path:
        if logger.hasHandlers():
            logger.handlers.clear()

        fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def color_text(text, color="default", style="normal"):
    """
    color:  red, green, yellow, blue, magenta, cyan, white, default
    style:  normal, bold, underline
    """
    colors = {
        "default": 39,
        "black": 30, "k": 30,
        "red": 31, "r": 31,
        "green": 32, "g": 32,
        "yellow": 33, "y": 33,
        "blue": 34, "b": 34,
        "magenta": 35, "m": 35,
        "cyan": 36, "c": 36,
        "white": 97, "w": 97,
    }

    styles = {
        "normal": 0,
        "bold": 1,
        "underline": 4,
    }

    color_code = colors.get(color, 39)
    style_code = styles.get(style, 0)
    return f"\033[{style_code};{color_code}m{text}\033[0m"


def encode_file_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")
        

def time_string(seconds: int) -> str:
    """
    Convert seconds to a human-readable string format.
    """
    if seconds < 60:
        return f"{seconds:.3f}s"
    elif seconds < 3600:
        minutes, seconds = divmod(seconds, 60)
        return f"{minutes}m {seconds:.3f}s"
    else:
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h {minutes}m {seconds:.3f}s"


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def make_hashable(obj):
    """Convert any Python object to a hashable form"""
    if isinstance(obj, (list, tuple, set)):
        return tuple(sorted([make_hashable(i) for i in obj]))
    elif isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    else:
        return obj


def hashable_cache(func):
    """Custom cache decorator that supports list/dict parameters and LRU functionality"""
    _cache = LRUCache(maxsize=512)
    _lock = threading.Lock()
    
    if inspect.iscoroutinefunction(func):
        # async function version
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = make_hashable((args, kwargs))
            with _lock:
                if key in _cache:
                    return _cache[key]
            result = await func(*args, **kwargs)
            with _lock:
                _cache[key] = result
            return result
    else:
        # sync function version
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = make_hashable((args, kwargs))
            with _lock:
                if key in _cache:
                    return _cache[key]
            result = func(*args, **kwargs)
            with _lock:
                _cache[key] = result
            return result
        
    wrapper.__signature__ = inspect.signature(func)

    return wrapper


def extract_code_blocks(
    text: str, 
    language: str = "python", 
    extract_all: bool = False
) -> Optional[list | str]:
    """
    Extract code blocks of a specified programming language from text (e.g., code starting with ```python)
    
    Args:
        text: Input text
        language: Target language (e.g., 'python', default), case-insensitive
        extract_all: Whether to extract all matching code blocks
    
    Returns:
        List of matched code blocks or a single code block string (returns None if no match)
    """
    # Build regular expression: match code blocks starting with ```language (ignoring case and surrounding whitespace)
    pattern = rf"```\s*{language}\s*\n(.*?)\n```"
    matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    
    if not matches:
        return None if not extract_all else []
    
    cleaned = [m.strip() for m in matches]
    return cleaned if extract_all else cleaned[0]


def truncate_string(string, max_length, by='token', method='center', ellipsis=None):
    """
    Truncate a string to a maximum length, ensuring it does not exceed the limit.
    
    Args:
        string (str): The string to truncate.
        max_length (int): The maximum length of the truncated string.
        by (str): The unit to truncate by ('token' or 'char').
        method (str): The truncation method ('start', 'end', 'center').
        ellipsis (str): The string to indicate truncation.
        
    Returns:
        str: The truncated log string.
    """
    if by == 'char':
        tokens = list(string)
    else:
        tokens = TIKTOKEN_ENCODING.encode(string)

    if len(tokens) <= max_length:
        return string
    
    def decode(_tokens):
        if by == 'char':
            return ''.join(_tokens)
        else:
            return TIKTOKEN_ENCODING.decode(_tokens)
    
    if ellipsis is None:
        ellipsis = "\n... (truncated) ...\n"
    
    if method == 'start':
        return ellipsis + "\n" + decode(tokens[-max_length:])
    elif method == 'end':
        return decode(tokens[:max_length]) + "\n" + ellipsis
    elif method == 'center':
        half_len = max_length // 2
        return decode(tokens[:half_len]) + f"\n{ellipsis}\n" + decode(tokens[-half_len:])
    
    raise ValueError(f"Invalid truncation method: {method}")


def get_container_bind_path_map(container_cmd: list):
    bind_map = {}
    for i in range(len(container_cmd)-1):
        if container_cmd[i] == "--bind":
            splits = container_cmd[i+1].split(":")
            if not (len(splits) in [2, 3]):
                assert False, f"Invalid bind mount format: {container_cmd[i+1]}"
            if len(splits) == 3:
                splits = splits[:2]
            host_path, container_path = splits
            bind_map[container_path] = host_path
    return bind_map


def get_host_path_from_container_path(container_cmd: list, container_path: str):
    bind_map = get_container_bind_path_map(container_cmd)
    keys = sorted(bind_map.keys(), key=lambda x: -len(x))
    for key in keys:
        if container_path.startswith(key):
            relative_path = container_path[len(key):]
            return bind_map[key] + relative_path

    raise ValueError(f"Container path '{container_path}' not found in bind mounts.")


def svg2png(svg_path):
    out_path = os.path.join(
        os.path.dirname(svg_path),
        uuid.uuid4().hex + ".png"
    )
    cairosvg.svg2png(url=svg_path, write_to=out_path)
    return out_path


P = ParamSpec("P")
T = TypeVar("T")

def _run_func_in_executor(
    func: Callable[P, T],
    args: tuple = (),
    kwargs: dict = {},
) -> T:
    try:
        return func(*args, **kwargs)
    except StopIteration as exc:
        # StopIteration can't be set on an asyncio.Future
        # it raises a TypeError and leaves the Future pending forever
        # so we need to convert it to a RuntimeError
        raise RuntimeError from exc

def run_in_executor(
    executor_or_config: Executor | None,
    func: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
):
    """Run a function in an executor.

    Args:
        executor_or_config: The executor or config to run in.
        func: The function.
        *args: The positional arguments to the function.
        **kwargs: The keyword arguments to the function.

    Returns:
        The output of the function.
    """

    if executor_or_config is None or isinstance(executor_or_config, dict):
        # Use default executor with context copied from current context
        return asyncio.get_running_loop().run_in_executor(
            None,
            cast("Callable[..., T]", partial(copy_context().run, _run_func_in_executor, func, args, kwargs)),
        )

    return asyncio.get_running_loop().run_in_executor(executor_or_config, _run_func_in_executor, func, args, kwargs)


class CompactListEncoder(JSONEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_indent = ""
        
    def encode(self, o):
        if isinstance(o, list):
            # Lists don't break lines
            return "[" + ", ".join(self.encode(item) for item in o) + "]"
        elif isinstance(o, dict):
            # Dictionaries maintain indentation
            self.current_indent += " " * self.indent
            output = "{\n"
            for i, (key, value) in enumerate(o.items()):
                output += self.current_indent + json.dumps(key) + ": "
                output += self.encode(value)
                if i < len(o) - 1:
                    output += ","
                output += "\n"
            self.current_indent = self.current_indent[:-self.indent]
            output += self.current_indent + "}"
            return output
        else:
            return json.dumps(o)
        

class FlushableLogger(logging.Logger):
    def info(self, msg, *args, flush=False, **kwargs):
        super().info(msg, *args, **kwargs)
        if flush:
            for handler in self.handlers:
                handler.flush()

    def debug(self, msg, *args, flush=False, **kwargs):
        super().debug(msg, *args, **kwargs)
        if flush:
            for handler in self.handlers:
                handler.flush()

    def warning(self, msg, *args, flush=False, **kwargs):
        super().warning(msg, *args, **kwargs)
        if flush:
            for handler in self.handlers:
                handler.flush()

    def error(self, msg, *args, flush=False, **kwargs):
        super().error(msg, *args, **kwargs)
        if flush:
            for handler in self.handlers:
                handler.flush()

    def critical(self, msg, *args, flush=False, **kwargs):
        super().critical(msg, *args, **kwargs)
        if flush:
            for handler in self.handlers:
                handler.flush()


class LoggerPrefixFilter(logging.Filter):
    def __init__(self, prefix: str):
        super().__init__()
        self.prefix = prefix

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith(self.prefix)
    

@contextlib.asynccontextmanager
async def capture_logger_prefix(prefix: str, level=logging.DEBUG):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.addFilter(LoggerPrefixFilter(prefix))

    formatter = logging.Formatter(
        "[%(levelname)s]:%(name)s:%(filename)s:line %(lineno)4d: %(message)s"
    )
    handler.setFormatter(formatter)
    
    root = logging.getLogger()
    old_level = root.level
    root.setLevel(level)
    root.addHandler(handler)

    try:
        yield stream
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
