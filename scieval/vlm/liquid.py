import warnings

import torch
from PIL import Image

from .base import BaseModel


class LFM2VL(BaseModel):
    """Hugging Face Transformers adapter for the LiquidAI LFM2-VL family."""

    INSTALL_REQ = True
    INTERLEAVE = True

    _NO_BRIEF_INSTRUCTION = {"MathVista_MINI", "MM-IFEval", "MMVet"}
    _BRIEF_INSTRUCTION = (
        "\nPlease answer directly with only the final answer, "
        "do not give any explanation."
    )

    def __init__(
        self,
        model_path="LiquidAI/LFM2-VL-450M",
        device=None,
        model_kwargs=None,
        use_default_instruction=True,
        **kwargs,
    ):
        super().__init__()

        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "LFM2-VL requires a recent Transformers release "
                "(version 4.57 or newer)."
            ) from exc

        self.device = device or self._default_device()
        self.use_default_instruction = use_default_instruction

        load_kwargs = dict(model_kwargs or {})
        load_kwargs.setdefault(
            "dtype", torch.bfloat16 if self.device == "cuda" else torch.float32
        )

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, **load_kwargs
        )
        if "device_map" not in load_kwargs:
            self.model = self.model.to(self.device)
        self.model = self.model.eval()

        generation_kwargs = {"max_new_tokens": 1024, "use_cache": True}
        generation_kwargs.update(kwargs)
        self.kwargs = generation_kwargs
        warnings.warn(
            f"Following kwargs received: {self.kwargs}, will use as generation config."
        )

    @staticmethod
    def _default_device():
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def custom_instruction_prompt_by_dataset(self, dataset):
        if not self.use_default_instruction or dataset in self._NO_BRIEF_INSTRUCTION:
            return ""
        return self._BRIEF_INSTRUCTION

    @staticmethod
    def _load_image(path):
        with Image.open(path) as image:
            return image.convert("RGB")

    def message_to_chat_messages(self, message, instruction_prompt, dataset=None):
        content = []
        for item in message:
            if item["type"] == "image":
                content.append(
                    {"type": "image", "image": self._load_image(item["value"])}
                )
            elif item["type"] == "text":
                content.append({"type": "text", "text": item["value"]})

        if instruction_prompt:
            content.append({"type": "text", "text": instruction_prompt})

        if dataset == "MM-IFEval":
            images = [item for item in content if item["type"] == "image"]
            texts = [item for item in content if item["type"] != "image"]
            content = images + texts

        return [{"role": "user", "content": content}]

    def generate_inner(self, message, dataset=None):
        instruction = self.custom_instruction_prompt_by_dataset(dataset)
        conversation = self.message_to_chat_messages(message, instruction, dataset)
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(self.model.device)
        input_length = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **self.kwargs)

        generated_ids = outputs[:, input_length:]
        return self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

    def chat_inner(self, message, dataset=None):
        return self.generate_inner(message, dataset)
