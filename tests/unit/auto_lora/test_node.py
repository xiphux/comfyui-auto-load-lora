import asyncio
import json

import pytest

from auto_lora import node
from auto_lora.loader import LoraCache
from auto_lora.node import ROWS_WIDGET_TYPE, AutoLoraLoader, comfy_entrypoint


class FakeBackend:
    def __init__(self, names):
        self.names = names
        self.applied = []

    def list_loras(self):
        return list(self.names)

    def full_path(self, name):
        return f"/nonexistent/{name}" if name in self.names else None

    def load_file(self, path):
        return f"sd:{path}", None

    def apply(self, model, clip, lora, strength_model, strength_clip, metadata):
        self.applied.append((lora, strength_model))
        return model + [lora], clip


class NoStatCache(LoraCache):
    def get(self, path, load_file):
        return load_file(path)


@pytest.fixture
def backend(monkeypatch):
    fake = FakeBackend(["elara.safetensors", "kael.safetensors"])
    monkeypatch.setattr(node, "_backend", fake)
    monkeypatch.setattr(node, "_cache", NoStatCache())
    return fake


ROWS = json.dumps([
    {"lora": "elara.safetensors", "triggers": "Elara Voss", "strength": 1.0, "enabled": True},
    {"lora": "kael.safetensors", "triggers": "kaelen thorne, kael thorne",
     "strength": 0.8, "enabled": True},
    {"lora": "missing.safetensors", "triggers": "ghost", "strength": 1.0, "enabled": True},
    {"lora": "kael.safetensors", "triggers": "(bad", "regex": True},
])


def run(text, loras=ROWS, clip=None):
    out = AutoLoraLoader.execute(model=["base"], clip=clip, text=text, loras=loras)
    return out.args


class TestSchema:
    def test_inputs_and_outputs(self):
        schema = AutoLoraLoader.define_schema()
        inputs = {i.args[0]: i.kwargs for i in schema.kwargs["inputs"]}
        assert list(inputs) == ["model", "clip", "text", "loras"]
        assert inputs["clip"]["optional"] is True
        assert inputs["text"]["force_input"] is True
        assert inputs["loras"]["socketless"] is True
        assert inputs["loras"]["extra_dict"] == {"widgetType": ROWS_WIDGET_TYPE}
        assert len(schema.kwargs["outputs"]) == 3

    def test_entrypoint_lists_node(self):
        async def nodes():
            extension = await comfy_entrypoint()
            return await extension.get_node_list()

        assert asyncio.run(nodes()) == [AutoLoraLoader]


class TestValidateInputs:
    def test_valid(self):
        assert AutoLoraLoader.validate_inputs(loras=ROWS) is True
        assert AutoLoraLoader.validate_inputs(loras="[]") is True

    def test_invalid(self):
        assert "not valid JSON" in AutoLoraLoader.validate_inputs(loras="{nope")


class TestExecute:
    def test_no_trigger_passes_model_through(self, backend):
        model, clip, applied = run("a photo of a cat")
        assert model == ["base"]
        assert clip is None
        assert applied.startswith("(no LoRAs applied)")
        assert backend.applied == []

    def test_applies_matching_lora(self, backend):
        model, _, applied = run("photo of elara-voss at the beach")
        assert model == ["base", "sd:/nonexistent/elara.safetensors"]
        assert 'elara.safetensors @ 1 (matched "elara-voss")' in applied

    def test_applies_several(self, backend):
        run("Kael Thorne and Elara Voss")
        assert backend.applied == [
            ("sd:/nonexistent/elara.safetensors", 1.0),
            ("sd:/nonexistent/kael.safetensors", 0.8),
        ]

    def test_warnings_in_applied_output(self, backend):
        _, _, applied = run("ghost story")
        assert "WARNING: LoRA file not found: missing.safetensors" in applied
        assert "WARNING: invalid regex for kael.safetensors" in applied

    def test_empty_rows(self, backend):
        model, _, applied = run("Elara Voss", loras="")
        assert model == ["base"]
        assert applied == "(no LoRAs applied)"
