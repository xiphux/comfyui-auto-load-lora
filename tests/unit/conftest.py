"""Minimal stand-ins for the ComfyUI modules imported at module load time.

Only ``comfy_api.latest`` is needed: ``folder_paths`` and ``comfy.*`` are imported
lazily by ``ComfyBackend`` and tests substitute a fake backend instead.
"""

import sys
import types


class _Recorder:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def _comfy_type(io_type):
    return type(io_type, (), {"io_type": io_type, "Input": _Recorder, "Output": _Recorder})


class _ComfyNode:
    pass


class _NodeOutput:
    def __init__(self, *args, ui=None):
        self.args = args
        self.ui = ui


class _ComfyExtension:
    async def on_load(self):
        pass


def _install_comfy_api_stub():
    io = types.ModuleType("comfy_api.latest.io")
    io.ComfyNode = _ComfyNode
    io.NodeOutput = _NodeOutput
    io.Schema = _Recorder
    io.Model = _comfy_type("MODEL")
    io.Clip = _comfy_type("CLIP")
    io.String = _comfy_type("STRING")

    latest = types.ModuleType("comfy_api.latest")
    latest.io = io
    latest.ComfyExtension = _ComfyExtension

    comfy_api = types.ModuleType("comfy_api")
    comfy_api.latest = latest

    sys.modules.update(
        {"comfy_api": comfy_api, "comfy_api.latest": latest, "comfy_api.latest.io": io}
    )


if "comfy_api" not in sys.modules:
    _install_comfy_api_stub()
