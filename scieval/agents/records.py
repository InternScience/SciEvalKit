import base64
import io
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PIL import Image

from ..smp import dump, load


def _image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return "data:image/png;base64," + img_str


@dataclass
class ToolCalling:
    tool_name: Any
    tool_input: Any
    tool_output: Optional[str] = None

    def add_response(self, response: str) -> None:
        self.tool_output = response

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "tool_output": self.tool_output,
        }


class StepResult:
    def __init__(self, role: str, content: Optional[List[Dict[str, Any]]]):
        self.role = role
        self.content = content if content is not None else []
        self.tool_calling: List[ToolCalling] = []

    def add_tool_calling(self, tool_result: ToolCalling) -> None:
        self.tool_calling.append(tool_result)

    def to_dict(self) -> Dict[str, Any]:
        serialized_content: List[Dict[str, Any]] = []
        for item in self.content:
            if item.get("type") == "image" and "image" in item:
                image = item["image"]
                if isinstance(image, Image.Image):
                    item = dict(item)
                    item["image"] = _image_to_base64(image)
            serialized_content.append(item)

        return {
            "role": self.role,
            "content": serialized_content,
            "tool_calling": [tc.to_dict() for tc in self.tool_calling],
        }


class EvalResult:
    def __init__(self, success: bool, final_answer: str):
        self.success = success
        self.final_answer = final_answer
        self.steps: List[StepResult] = []

    def add_step(self, step: StepResult) -> None:
        self.steps.append(step)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "final_answer": self.final_answer,
            "steps": [step.to_dict() for step in self.steps],
        }


class TrajectoryStore:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)

    def traj_path(self, idx: int) -> str:
        return os.path.join(self.root_dir, f"sample_{idx}_traj.json")

    def eval_path(self, idx: int) -> str:
        return os.path.join(self.root_dir, f"sample_{idx}_eval.json")

    def load_traj(self, idx: int) -> Optional[Dict[str, Any]]:
        path = self.traj_path(idx)
        if not os.path.exists(path):
            return None
        return load(path)

    def load_eval(self, idx: int) -> Optional[Dict[str, Any]]:
        path = self.eval_path(idx)
        if not os.path.exists(path):
            return None
        return load(path)

    def save_traj(self, idx: int, result: EvalResult) -> None:
        dump(result.to_dict(), self.traj_path(idx))

    def save_eval(self, idx: int, record: Any) -> None:
        if hasattr(record, "to_dict"):
            record = record.to_dict()
        dump(record, self.eval_path(idx))


class EvalRecord:
    def __init__(
        self,
        index: int,
        final_answer: str,
        score: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.index = index
        self.final_answer = final_answer
        self.score = score
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "final_answer": self.final_answer,
            "score": self.score,
            "metadata": self.metadata,
        }
