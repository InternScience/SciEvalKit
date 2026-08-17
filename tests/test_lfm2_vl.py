import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from PIL import Image


class FakeBatch(dict):
    def __init__(self):
        super().__init__(input_ids=torch.tensor([[1, 2, 3]]))
        self.device = None

    def to(self, device):
        self.device = device
        return self


class FakeProcessor:
    def __init__(self):
        self.conversation = None
        self.template_kwargs = None
        self.decoded_ids = None

    def apply_chat_template(self, conversation, **kwargs):
        self.conversation = conversation
        self.template_kwargs = kwargs
        return FakeBatch()

    def batch_decode(self, generated_ids, **kwargs):
        self.decoded_ids = generated_ids
        return ["  final answer  "]


class FakeModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.to_device = None
        self.eval_called = False
        self.generation_kwargs = None

    def to(self, device):
        self.to_device = device
        self.device = torch.device(device)
        return self

    def eval(self):
        self.eval_called = True
        return self

    def generate(self, **kwargs):
        self.generation_kwargs = kwargs
        return torch.tensor([[1, 2, 3, 8, 9]])


@pytest.fixture()
def liquid_module(monkeypatch):
    processor = FakeProcessor()
    model = FakeModel()
    processor_auto = types.SimpleNamespace(
        from_pretrained=lambda _model_path: processor
    )
    model_auto = types.SimpleNamespace(
        from_pretrained=lambda _model_path, **_kwargs: model
    )
    transformers = types.ModuleType("transformers")
    transformers.AutoProcessor = processor_auto
    transformers.AutoModelForImageTextToText = model_auto
    monkeypatch.setitem(sys.modules, "transformers", transformers)

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

    module_path = Path(__file__).parents[1] / "scieval" / "vlm" / "liquid.py"
    spec = importlib.util.spec_from_file_location("scieval.vlm.liquid", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, processor, model


def test_generate_uses_multimodal_chat_template_and_decodes_new_tokens(
    liquid_module, tmp_path
):
    module, processor, model = liquid_module
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (2, 2), "red").save(image_path)

    adapter = module.LFM2VL(device="cpu", max_new_tokens=12)
    result = adapter.generate_inner(
        [
            {"type": "text", "value": "before"},
            {"type": "image", "value": str(image_path)},
            {"type": "text", "value": "after"},
        ]
    )

    content = processor.conversation[0]["content"]
    assert [item["type"] for item in content] == ["text", "image", "text", "text"]
    assert isinstance(content[1]["image"], Image.Image)
    assert content[-1]["text"] == adapter._BRIEF_INSTRUCTION
    assert processor.template_kwargs == {
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "return_dict": True,
        "tokenize": True,
    }
    assert model.to_device == "cpu"
    assert model.eval_called
    assert model.generation_kwargs["max_new_tokens"] == 12
    assert torch.equal(processor.decoded_ids, torch.tensor([[8, 9]]))
    assert result == "final answer"


def test_mm_ifeval_moves_images_first_without_adding_instruction(
    liquid_module, tmp_path
):
    module, processor, _model = liquid_module
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (2, 2), "blue").save(image_path)
    adapter = module.LFM2VL(device="cpu")

    adapter.generate_inner(
        [
            {"type": "text", "value": "first"},
            {"type": "image", "value": str(image_path)},
            {"type": "text", "value": "last"},
        ],
        dataset="MM-IFEval",
    )

    content = processor.conversation[0]["content"]
    assert [item["type"] for item in content] == ["image", "text", "text"]
    assert [item["text"] for item in content[1:]] == ["first", "last"]


def test_device_map_skips_explicit_model_move(liquid_module):
    module, _processor, model = liquid_module
    adapter = module.LFM2VL(
        device="cpu", model_kwargs={"device_map": "auto", "dtype": "auto"}
    )

    assert adapter.device == "cpu"
    assert model.to_device is None
