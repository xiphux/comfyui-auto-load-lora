"""Auto LoRA Loader: a ComfyUI custom node that applies LoRAs based on trigger words."""

WEB_DIRECTORY = "./src/web"

# ComfyUI imports this file as a package. Pytest also imports it (because the repo
# root has an __init__.py), but as a bare module, where the relative import fails.
if __package__:
    from .src.auto_lora.node import comfy_entrypoint

    __all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
