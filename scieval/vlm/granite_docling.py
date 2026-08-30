import warnings

import torch
from PIL import Image

from .base import BaseModel


class GraniteDocling(BaseModel):
    """Transformers adapter for IBM Granite Docling document conversion."""

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(
        self,
        model_path="ibm-granite/granite-docling-258M",
        device=None,
        model_kwargs=None,
        skip_special_tokens=False,
        **kwargs,
    ):
        super().__init__()
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Granite Docling requires a recent Transformers release."
            ) from exc

        self.device = device or self._default_device()
        self.skip_special_tokens = skip_special_tokens

        load_kwargs = dict(model_kwargs or {})
        load_kwargs.setdefault(
            "dtype", torch.bfloat16 if self.device == "cuda" else torch.float32
        )
        load_kwargs.setdefault("_attn_implementation", "sdpa")

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, **load_kwargs
        )
        if "device_map" not in load_kwargs:
            self.model = self.model.to(self.device)
        self.model = self.model.eval()

        generation_kwargs = {"max_new_tokens": 1024, "use_cache": False}
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

    @staticmethod
    def _open_image(path):
        with Image.open(path) as image:
            return image.convert("RGB")

    def message_to_chat_messages(self, message):
        content = []
        images = []
        for item in message:
            if item["type"] == "image":
                images.append(self._open_image(item["value"]))
                content.append({"type": "image"})
            elif item["type"] == "text":
                content.append({"type": "text", "text": item["value"].strip()})
        return [{"role": "user", "content": content}], images

    def generate_inner(self, message, dataset=None):
        conversation, images = self.message_to_chat_messages(message)
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(
            text=prompt,
            images=images or None,
            return_tensors="pt",
        ).to(self.model.device)
        input_length = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **self.kwargs)

        return self.processor.batch_decode(
            generated_ids[:, input_length:],
            skip_special_tokens=self.skip_special_tokens,
        )[0].strip()
