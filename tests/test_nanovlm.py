import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from PIL import Image


class FakeConfig:
    lm_tokenizer = "fake-tokenizer"
    vlm_extra_tokens = {"image_token": "<image>"}
    lm_chat_template = "fake-template"
    max_img_size = 2048
    vit_img_size = 512
    resize_to_max_side_len = True
    mp_image_token_length = 64


class FakeModel:
    def __init__(self):
        self.cfg = FakeConfig()
        self.to_device = None
        self.eval_called = False
        self.input_ids = None
        self.images = None
        self.generation_kwargs = None

    def to(self, device):
        self.to_device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def generate(self, input_ids, images, **kwargs):
        self.input_ids = input_ids
        self.images = images
        self.generation_kwargs = kwargs
        return torch.tensor([[7, 8]])


class FakeTokenizer:
    global_image_token = "<global-image>"

    def __init__(self):
        self.conversation = None
        self.decoded_ids = None

    def apply_chat_template(self, conversation, **_kwargs):
        self.conversation = conversation
        return [[1, 2, 3]]

    def batch_decode(self, generated_ids, **_kwargs):
        self.decoded_ids = generated_ids
        return ["  Cat.  "]


@pytest.fixture()
def nanovlm_module(monkeypatch):
    model = FakeModel()
    tokenizer = FakeTokenizer()
    image_processor_calls = []
    image_string_calls = []

    vision_language_model = types.ModuleType("models.vision_language_model")

    class VisionLanguageModel:
        @staticmethod
        def from_pretrained(_model_path):
            return model

    vision_language_model.VisionLanguageModel = VisionLanguageModel

    processors = types.ModuleType("data.processors")
    processors.get_tokenizer = lambda *_args: tokenizer

    def get_image_processor(*args):
        image_processor_calls.append(args)

        def process(_image):
            return torch.zeros((1, 3, 2, 2)), (1, 1)

        return process

    def get_image_string(used_tokenizer, ratios, token_length):
        image_string_calls.append((used_tokenizer, ratios, token_length))
        return "<image-tokens>" if ratios else ""

    processors.get_image_processor = get_image_processor
    processors.get_image_string = get_image_string

    models = types.ModuleType("models")
    models.__path__ = []
    data = types.ModuleType("data")
    data.__path__ = []
    monkeypatch.setitem(sys.modules, "models", models)
    monkeypatch.setitem(
        sys.modules, "models.vision_language_model", vision_language_model
    )
    monkeypatch.setitem(sys.modules, "data", data)
    monkeypatch.setitem(sys.modules, "data.processors", processors)

    scieval = types.ModuleType("scieval")
    scieval.__path__ = []
    vlm = types.ModuleType("scieval.vlm")
    vlm.__path__ = []
    base = types.ModuleType("scieval.vlm.base")

    class BaseModel:
        def __init__(self):
            self.dump_image_func = None

    base.BaseModel = BaseModel
    monkeypatch.setitem(sys.modules, "scieval", scieval)
    monkeypatch.setitem(sys.modules, "scieval.vlm", vlm)
    monkeypatch.setitem(sys.modules, "scieval.vlm.base", base)

    module_path = Path(__file__).parents[1] / "scieval" / "vlm" / "nanovlm.py"
    spec = importlib.util.spec_from_file_location("scieval.vlm.nanovlm", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, model, tokenizer, image_processor_calls, image_string_calls


def test_generate_prepares_image_prompt_and_decodes_output(
    nanovlm_module, tmp_path
):
    module, model, tokenizer, processor_calls, image_string_calls = nanovlm_module
    image_path = tmp_path / "cat.png"
    Image.new("RGB", (2, 2), "orange").save(image_path)

    adapter = module.NanoVLM(device="cpu", max_new_tokens=8)
    answer = adapter.generate_inner(
        [
            {"type": "image", "value": str(image_path)},
            {"type": "text", "value": " What animal is shown? "},
        ]
    )

    assert model.to_device == torch.device("cpu")
    assert model.eval_called
    assert processor_calls == [(2048, 512, True)]
    assert image_string_calls == [(tokenizer, [(1, 1)], 64)]
    assert tokenizer.conversation[0][0]["content"] == (
        "<image-tokens>What animal is shown?"
    )
    assert model.input_ids.shape == (1, 3)
    assert len(model.images) == 1
    assert model.images[0].device.type == "cpu"
    assert model.generation_kwargs == {"max_new_tokens": 8, "greedy": True}
    assert torch.equal(tokenizer.decoded_ids, torch.tensor([[7, 8]]))
    assert answer == "Cat."


def test_text_only_generation_passes_no_images(nanovlm_module):
    module, model, tokenizer, _processor_calls, image_string_calls = nanovlm_module
    adapter = module.NanoVLM(device="cpu")

    adapter.generate_inner([{"type": "text", "value": "hello"}])

    assert image_string_calls == [(tokenizer, [], 64)]
    assert model.images is None
    assert tokenizer.conversation[0][0]["content"] == "hello"
