import os
import sys
import warnings

import torch
from PIL import Image

from .base import BaseModel


_NANOVLM_INSTALL_MESSAGE = (
    "nanoVLM is not distributed as a Python package. Clone "
    "https://github.com/huggingface/nanoVLM and set NANOVLM_PATH to the "
    "checkout directory before running SciEvalKit."
)


def _ensure_nanovlm_importable():
    nanovlm_path = os.environ.get("NANOVLM_PATH", "")
    if nanovlm_path and nanovlm_path not in sys.path:
        sys.path.insert(0, nanovlm_path)


class NanoVLM(BaseModel):
    """Adapter for the pure-PyTorch Hugging Face nanoVLM implementation."""

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(
        self,
        model_path="lusxvr/nanoVLM-230M-8k",
        device=None,
        **kwargs,
    ):
        super().__init__()
        _ensure_nanovlm_importable()
        try:
            from data.processors import get_image_processor, get_tokenizer
            from models.vision_language_model import VisionLanguageModel
        except ImportError as exc:
            raise ImportError(_NANOVLM_INSTALL_MESSAGE) from exc

        self.device = torch.device(device or self._default_device())
        self.model = VisionLanguageModel.from_pretrained(model_path)
        self.model = self.model.to(self.device).eval()
        self.config = self.model.cfg

        self.tokenizer = get_tokenizer(
            self.config.lm_tokenizer,
            self.config.vlm_extra_tokens,
            self.config.lm_chat_template,
        )
        self.image_processor = get_image_processor(
            self.config.max_img_size,
            self.config.vit_img_size,
            getattr(self.config, "resize_to_max_side_len", False),
        )

        generation_kwargs = {"max_new_tokens": 2048, "greedy": True}
        generation_kwargs.update(kwargs)
        self.kwargs = generation_kwargs
        warnings.warn(f"NanoVLM kwargs: {self.kwargs}")

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

    def _prepare_images(self, message):
        processed_images = []
        image_ratios = []
        for item in message:
            if item["type"] != "image":
                continue
            image = self._open_image(item["value"])
            processed, ratio = self.image_processor(image)
            if (
                not hasattr(self.tokenizer, "global_image_token")
                and ratio[0] * ratio[1] == len(processed) - 1
            ):
                processed = processed[1:]
            processed_images.append(processed.to(self.device))
            image_ratios.append(ratio)
        return processed_images, image_ratios

    @staticmethod
    def _message_text(message):
        return "\n".join(
            item["value"].strip() for item in message if item["type"] == "text"
        )

    def generate_inner(self, message, dataset=None):
        _ensure_nanovlm_importable()
        try:
            from data.processors import get_image_string
        except ImportError as exc:
            raise ImportError(_NANOVLM_INSTALL_MESSAGE) from exc

        images, image_ratios = self._prepare_images(message)
        image_string = get_image_string(
            self.tokenizer, image_ratios, self.config.mp_image_token_length
        )
        prompt = image_string + self._message_text(message)
        conversation = [{"role": "user", "content": prompt}]
        encoded_prompt = self.tokenizer.apply_chat_template(
            [conversation], tokenize=True, add_generation_prompt=True
        )
        input_ids = torch.as_tensor(encoded_prompt, device=self.device)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        generated_ids = self.model.generate(
            input_ids,
            images or None,
            **self.kwargs,
        )
        return self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
