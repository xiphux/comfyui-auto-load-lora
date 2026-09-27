# Auto LoRA Loader

A ComfyUI custom node that applies LoRAs based on the prompt. You list the LoRAs you
want available and the trigger words for each; when a trigger word appears in the
prompt, that LoRA is applied to the model. If nothing matches, the model passes
through unchanged.

This is meant for libraries of one-LoRA-per-character models that shouldn't all be
loaded at once. Name the character in the prompt and the right LoRA is picked up,
including when the workflow runs through the API.

## Installation

Clone into `ComfyUI/custom_nodes/` and restart ComfyUI:

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/xiphux/comfyui-auto-load-lora
```

No extra Python dependencies are needed.

## Usage

Add **Auto LoRA Loader** (category `loaders`) between your model loader and sampler.

| Input   | Description |
|---------|-------------|
| `model` | The diffusion model. |
| `clip`  | Optional. Only needed for LoRAs that also patch the text encoder. |
| `text`  | The **positive** prompt to scan (don't wire the negative prompt, or a name in it would load that LoRA). |

| Output    | Description |
|-----------|-------------|
| `MODEL`   | The model with the matched LoRAs applied. |
| `CLIP`    | The clip with the matched LoRAs applied (or passed through). |
| `applied` | A summary of which LoRAs were applied and any warnings. Handy for debugging API runs. |

A typical layout: a string/primitive node holds the prompt, wired both to this node's
`text` input and to your `CLIPTextEncode`.

### Configuring LoRAs

Each row has:

- **Enabled** checkbox
- **LoRA** file
- **Strength**, used for both the model and clip (clip is skipped when not connected)
- **Triggers**: comma-separated aliases, e.g. `elara voss, elara, e. voss`
- **Regex**: treat the triggers field as one regular expression instead

When you pick a LoRA whose file defines `modelspec.trigger_phrase` in its metadata,
the triggers field is filled in from it if it's empty. Metadata is sometimes wrong,
so check it. Nothing is guessed from training tags.

### Matching rules

- Case-insensitive.
- Whole words only: `ann` does not match `anna` or `planning`.
- Space, hyphen and underscore are interchangeable, and runs of them count as one:
  `elara voss` matches `Elara-Voss`, `elara_voss` and `(elara voss:1.2)`.
- In regex mode the whole field is a single Python regular expression, matched
  case-insensitively with no automatic word boundaries. Commas are regex syntax there,
  so use `|` for alternatives. Invalid expressions are highlighted in the editor and
  skipped (with a warning) at run time.
- Each LoRA file is applied at most once, even if several rows reference it.
- LoRAs are applied in row order.

### Missing files

If a row's LoRA isn't found, the node first looks for a file with the same name in a
different subfolder. If there is none, the row is skipped with a warning in the log
and the `applied` output. A missing file never fails the run.

### Presets

The bar at the top of the node saves and loads the row list as a named preset. Presets
are stored per user under `ComfyUI/user/<user>/auto_lora_presets/<name>.json`. Loading
a preset replaces the node's rows. Presets are never applied automatically, so
different workflows (e.g. for different base models) keep their own lists.

### API usage

The rows are stored in the node's `loras` input as a JSON string, so API-format
workflows exported from ComfyUI already contain them. To build one by hand:

```json
"inputs": {
  "model": ["1", 0],
  "text": ["2", 0],
  "loras": "[{\"lora\": \"characters/elara.safetensors\", \"triggers\": \"elara voss, elara\", \"regex\": false, \"strength\": 1.0, \"enabled\": true}]"
}
```

## Development

Uses [uv](https://docs.astral.sh/uv/). The node has no runtime dependencies; pytest and
ruff are dev-only.

```sh
uv sync                 # create .venv with the dev tools
uv run pytest           # tests
uv run ruff check .     # lint
uv run ruff format .    # format
```

CI (`.github/workflows/ci.yml`) runs ruff, a JavaScript syntax check, the tests on
Python 3.10 and 3.14, and zizmor over the workflows. Renovate
(`.github/renovate.json5`) keeps `uv.lock` and the pinned actions current, auto-merging
patch/minor updates once CI passes.

Layout:

- `__init__.py`: ComfyUI entry point (`comfy_entrypoint`, `WEB_DIRECTORY`)
- `src/auto_lora/matching.py`: row parsing and trigger matching (pure Python)
- `src/auto_lora/loader.py`: LoRA name resolution, cache, applying LoRAs
- `src/auto_lora/node.py`: the V3 node definition
- `src/web/auto_lora.js`: the row editor widget, trigger auto-fill and presets
- `tests/unit/`: mirrors `src/`, one test file per source file

The node uses the ComfyUI V3 schema (`comfy_api.latest`). The row editor is a DOM
widget, so it works in both the classic canvas and the Nodes 2.0 (Vue) renderer.

### Manual test checklist

The unit tests don't cover the UI or real model patching. After changes, check in a
real ComfyUI instance:

1. Add the node; add rows; pick a LoRA with `modelspec.trigger_phrase` and confirm the
   triggers field fills in.
2. Save the workflow, reload the page, and confirm the rows come back.
3. Save a preset, clear the rows, load it back; delete it.
4. Queue a prompt that names a trigger: the `applied` output lists the LoRA and the
   image reflects it. Queue one without a trigger: `(no LoRAs applied)`.
5. Toggle **Settings → Modern Node Design (Nodes 2.0)** and repeat steps 1 and 3.
6. Export the workflow in API format and queue it through `/prompt`.

## License

MIT
