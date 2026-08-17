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
        self.to_device = None

    def to(self, device):
        self.to_device = device
        return self


class FakeProcessor:
    def __init__(self):
        self.conversation = None
        self.prompt = None
        self.images = None
        self.decoded_ids = None
        self.skip_special_tokens = None

    def apply_chat_template(self, conversation, **_kwargs):
        self.conversation = conversation
        return "formatted prompt"

    def __call__(self, text, images, return_tensors):
        self.prompt = text
        self.images = images
        assert return_tensors == "pt"
        return FakeBatch()

    def batch_decode(self, generated_ids, skip_special_tokens):
        self.decoded_ids = generated_ids
        self.skip_special_tokens = skip_special_tokens
        return ["  <doctag><text>Science</text>  "]


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
def granite_docling_module(monkeypatch):
    processor = FakeProcessor()
    model = FakeModel()
    model_load_calls = []

    processor_auto = types.SimpleNamespace(
        from_pretrained=lambda _model_path: processor
    )

    def load_model(model_path, **kwargs):
        model_load_calls.append((model_path, kwargs))
        return model

    model_auto = types.SimpleNamespace(from_pretrained=load_model)
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

    module_path = (
        Path(__file__).parents[1] / "scieval" / "vlm" / "granite_docling.py"
    )
    spec = importlib.util.spec_from_file_location(
        "scieval.vlm.granite_docling", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, processor, model, model_load_calls


def test_generate_preserves_multimodal_order_and_decodes_new_tokens(
    granite_docling_module, tmp_path
):
    module, processor, model, model_load_calls = granite_docling_module
    image_path = tmp_path / "page.png"
    Image.new("RGB", (2, 2), "white").save(image_path)

    adapter = module.GraniteDocling(device="cpu", max_new_tokens=12)
    result = adapter.generate_inner(
        [
            {"type": "text", "value": " Convert this page. "},
            {"type": "image", "value": str(image_path)},
        ]
    )

    assert model_load_calls == [
        (
            "ibm-granite/granite-docling-258M",
            {"dtype": torch.float32, "_attn_implementation": "sdpa"},
        )
    ]
    assert model.to_device == "cpu"
    assert model.eval_called
    assert processor.conversation == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Convert this page."},
                {"type": "image"},
            ],
        }
    ]
    assert processor.prompt == "formatted prompt"
    assert len(processor.images) == 1
    assert isinstance(processor.images[0], Image.Image)
    assert model.generation_kwargs["max_new_tokens"] == 12
    assert torch.equal(processor.decoded_ids, torch.tensor([[8, 9]]))
    assert processor.skip_special_tokens is False
    assert result == "<doctag><text>Science</text>"


def test_device_map_skips_explicit_model_move(granite_docling_module):
    module, _processor, model, _model_load_calls = granite_docling_module
    adapter = module.GraniteDocling(
        device="cpu", model_kwargs={"device_map": "auto", "dtype": "auto"}
    )

    assert adapter.device == "cpu"
    assert model.to_device is None
