"""ComfyUI V3 node definition for the Auto LoRA Loader."""

from __future__ import annotations

import logging

from comfy_api.latest import ComfyExtension, io

from .loader import LOG_PREFIX, ComfyBackend, LoraCache, apply_matches
from .matching import RowsFormatError, find_matches, parse_rows

log = logging.getLogger(__name__)

# Must match the widget type registered in web/auto_lora.js.
ROWS_WIDGET_TYPE = "AUTO_LORA_ROWS"

# V3 nodes are stateless classmethods, so the cache lives at module level.
_cache = LoraCache()
_backend = ComfyBackend()


class AutoLoraLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AutoLoraLoaderByTrigger",
            display_name="Auto LoRA Loader",
            category="loaders",
            description=(
                "Applies LoRAs based on the prompt: each LoRA you add gets trigger words "
                "you define, and it is applied only when one of them appears in the prompt. "
                "If nothing matches, the model passes through unchanged."
            ),
            search_aliases=["lora", "trigger", "auto lora", "character lora"],
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input(
                    "clip",
                    optional=True,
                    tooltip="Optional. Only needed for LoRAs that also patch the text encoder.",
                ),
                io.String.Input(
                    "text",
                    force_input=True,
                    tooltip="The positive prompt to scan for trigger words.",
                ),
                io.String.Input(
                    "loras",
                    default="[]",
                    socketless=True,
                    tooltip="JSON list of {lora, triggers, regex, strength, enabled} rows.",
                    extra_dict={"widgetType": ROWS_WIDGET_TYPE},
                ),
            ],
            outputs=[
                io.Model.Output(display_name="MODEL"),
                io.Clip.Output(display_name="CLIP"),
                io.String.Output(
                    display_name="applied",
                    tooltip="Which LoRAs were applied, plus any warnings.",
                ),
            ],
        )

    @classmethod
    def validate_inputs(cls, loras: str = "[]", **kwargs) -> bool | str:
        try:
            parse_rows(loras)
        except RowsFormatError as err:
            return str(err)
        return True

    @classmethod
    def execute(cls, model, text: str, loras: str, clip=None) -> io.NodeOutput:
        rows = parse_rows(loras)
        matches, match_warnings = find_matches(rows, text)
        for warning in match_warnings:
            log.warning("%s %s", LOG_PREFIX, warning)

        result = apply_matches(model, clip, matches, _backend, _cache)
        result.warnings[:0] = match_warnings
        return io.NodeOutput(result.model, result.clip, result.summary())


class AutoLoraExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [AutoLoraLoader]


async def comfy_entrypoint() -> AutoLoraExtension:
    return AutoLoraExtension()
