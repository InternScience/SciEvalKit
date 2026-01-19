from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class EvalSample:
    def __init__(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        files: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, str]] = None,
    ):
        self.prompt = prompt
        self.images = images or []
        self.files = files or {}
        self.metadata = metadata or {}


class AgentBase(ABC):
    name = "agent"

    def __init__(self, name: Optional[str] = None, model_version: Optional[str] = None, **kwargs):
        self.name = name or getattr(self, "name", self.__class__.__name__.lower())
        self.model_version = model_version or "default"

    @abstractmethod
    def run(self, sample: EvalSample):
        pass
