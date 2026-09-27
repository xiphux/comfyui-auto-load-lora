import os

import pytest

from auto_lora.loader import ApplyResult, LoraCache, apply_matches, resolve_lora_name
from auto_lora.matching import LoraRow, Match


class FakeBackend:
    """Records calls; "applying" a LoRA appends its path to a list standing in for the model."""

    def __init__(self, root, names):
        self.root = root
        self.names = names
        self.loads = []
        self.applies = []
        self.fail_on = set()
        for name in names:
            path = os.path.join(root, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(b"weights")

    def list_loras(self):
        return list(self.names)

    def full_path(self, name):
        return os.path.join(self.root, name) if name in self.names else None

    def load_file(self, path):
        self.loads.append(path)
        if os.path.basename(path) in self.fail_on:
            raise RuntimeError("corrupt file")
        return f"sd:{os.path.basename(path)}", {"file": os.path.basename(path)}

    def apply(self, model, clip, lora, strength_model, strength_clip, metadata):
        self.applies.append((lora, strength_model, strength_clip, metadata))
        new_clip = None if clip is None else [*clip, lora]
        return [*model, lora], new_clip


def match(lora, strength=1.0, text="trigger"):
    return Match(row=LoraRow(lora=lora, triggers=text, strength=strength), matched_text=text)


@pytest.fixture
def backend(tmp_path):
    return FakeBackend(
        str(tmp_path),
        ["a.safetensors", "chars/b.safetensors", "other/c.safetensors", "chars/c.safetensors"],
    )


AVAILABLE = ["a.safetensors", "chars/b.safetensors", "chars\\d.safetensors"]


class TestResolveLoraName:
    def test_exact(self):
        assert resolve_lora_name("chars/b.safetensors", AVAILABLE) == "chars/b.safetensors"

    def test_backslash_separators(self):
        assert resolve_lora_name("chars\\b.safetensors", AVAILABLE) == "chars/b.safetensors"
        assert resolve_lora_name("chars/d.safetensors", AVAILABLE) == "chars\\d.safetensors"

    def test_different_folder(self):
        assert resolve_lora_name("old/b.safetensors", AVAILABLE) == "chars/b.safetensors"

    def test_missing_extension(self):
        assert resolve_lora_name("b", AVAILABLE) == "chars/b.safetensors"

    def test_not_found(self):
        assert resolve_lora_name("zzz.safetensors", AVAILABLE) is None

    def test_exact_beats_basename(self):
        available = ["x/a.safetensors", "a.safetensors"]
        assert resolve_lora_name("a.safetensors", available) == "a.safetensors"


class TestLoraCache:
    def test_reuses_loaded_file(self, backend):
        cache = LoraCache()
        path = backend.full_path("a.safetensors")
        assert cache.get(path, backend.load_file) == cache.get(path, backend.load_file)
        assert backend.loads == [path]

    def test_reloads_when_file_changes(self, backend):
        cache = LoraCache()
        path = backend.full_path("a.safetensors")
        cache.get(path, backend.load_file)
        with open(path, "wb") as f:
            f.write(b"new, longer weights")
        cache.get(path, backend.load_file)
        assert backend.loads == [path, path]

    def test_evicts_least_recently_used(self, backend):
        cache = LoraCache(max_entries=2)
        a, b, c = (backend.full_path(n) for n in backend.names[:3])
        cache.get(a, backend.load_file)
        cache.get(b, backend.load_file)
        cache.get(a, backend.load_file)  # a becomes most recent
        cache.get(c, backend.load_file)  # evicts b
        assert len(cache) == 2
        cache.get(a, backend.load_file)
        cache.get(b, backend.load_file)
        assert backend.loads == [a, b, c, b]

    def test_failed_load_is_not_cached(self, backend):
        cache = LoraCache()
        backend.fail_on.add("a.safetensors")
        path = backend.full_path("a.safetensors")
        with pytest.raises(RuntimeError):
            cache.get(path, backend.load_file)
        assert len(cache) == 0


class TestApplyMatches:
    def test_no_matches_passes_through(self, backend):
        model, clip = ["base"], ["clip"]
        result = apply_matches(model, clip, [], backend, LoraCache())
        assert result.model is model and result.clip is clip
        assert result.summary() == "(no LoRAs applied)"
        assert backend.applies == []

    def test_applies_in_order(self, backend):
        result = apply_matches(
            ["base"],
            ["clip"],
            [match("a.safetensors", 0.5, "alice"), match("chars/b.safetensors", 1.0, "bob")],
            backend,
            LoraCache(),
        )
        assert result.model == ["base", "sd:a.safetensors", "sd:b.safetensors"]
        assert result.clip == ["clip", "sd:a.safetensors", "sd:b.safetensors"]
        assert [(a[1], a[2]) for a in backend.applies] == [(0.5, 0.5), (1.0, 1.0)]
        assert backend.applies[0][3] == {"file": "a.safetensors"}
        assert result.summary() == (
            'a.safetensors @ 0.5 (matched "alice")\nchars/b.safetensors @ 1 (matched "bob")'
        )

    def test_without_clip_uses_zero_clip_strength(self, backend):
        result = apply_matches(["base"], None, [match("a.safetensors", 0.8)], backend, LoraCache())
        assert result.clip is None
        assert backend.applies[0][1:3] == (0.8, 0.0)

    def test_zero_strength_is_skipped(self, backend):
        result = apply_matches(["base"], None, [match("a.safetensors", 0)], backend, LoraCache())
        assert result.model == ["base"]
        assert backend.loads == []
        assert result.warnings == []

    def test_negative_strength_is_applied(self, backend):
        apply_matches(["base"], None, [match("a.safetensors", -0.5)], backend, LoraCache())
        assert backend.applies[0][1] == -0.5

    def test_missing_file_warns_and_continues(self, backend, caplog):
        result = apply_matches(
            ["base"],
            None,
            [match("gone.safetensors"), match("a.safetensors")],
            backend,
            LoraCache(),
        )
        assert result.model == ["base", "sd:a.safetensors"]
        assert result.warnings == ["LoRA file not found: gone.safetensors"]
        assert "WARNING: LoRA file not found: gone.safetensors" in result.summary()
        assert "LoRA file not found: gone.safetensors" in caplog.text

    def test_load_failure_warns_and_continues(self, backend):
        backend.fail_on.add("a.safetensors")
        result = apply_matches(
            ["base"],
            None,
            [match("a.safetensors"), match("chars/b.safetensors")],
            backend,
            LoraCache(),
        )
        assert result.model == ["base", "sd:b.safetensors"]
        assert result.warnings == ["could not load a.safetensors: corrupt file"]

    def test_moved_file_resolves_by_name(self, backend):
        result = apply_matches(["base"], None, [match("old/b.safetensors")], backend, LoraCache())
        assert result.model == ["base", "sd:b.safetensors"]
        assert result.applied[0].startswith("chars/b.safetensors")

    def test_uses_cache_across_calls(self, backend):
        cache = LoraCache()
        apply_matches(["base"], None, [match("a.safetensors")], backend, cache)
        apply_matches(["base"], None, [match("a.safetensors")], backend, cache)
        assert len(backend.loads) == 1


def test_summary_with_only_warnings():
    result = ApplyResult(model=None, clip=None, warnings=["x"])
    assert result.summary() == "(no LoRAs applied)\nWARNING: x"
