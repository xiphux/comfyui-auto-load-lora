# Auto LoRA Loader (ComfyUI custom node)

Applies LoRAs when user-defined trigger words appear in the prompt. See README.md for
behavior and usage.

## Layout

- `__init__.py`: repo root must stay importable by ComfyUI as a package. The relative
  import is guarded by `if __package__:` because pytest imports this file as a bare module.
- `src/auto_lora/matching.py`: pure Python, no ComfyUI imports. All matching rules live here.
- `src/auto_lora/loader.py`: ComfyUI modules (`folder_paths`, `comfy.*`) are imported
  lazily inside `ComfyBackend` only; tests pass a fake backend.
- `src/auto_lora/node.py`: V3 node (`comfy_api.latest`). V3 nodes are stateless
  classmethods, so the LoRA cache is module-level.
- `src/web/auto_lora.js`: DOM widget registered via `getCustomWidgets` for widget type
  `AUTO_LORA_ROWS` (must match `ROWS_WIDGET_TYPE` in node.py). The row list is the
  value of the socketless `loras` STRING input, serialized as JSON.
- `tests/unit/` mirrors `src/` one-to-one. `tests/unit/conftest.py` stubs `comfy_api.latest`.

## Commands

- Setup: `uv sync`
- Tests: `uv run pytest`
- Lint/format: `uv run ruff check .` and `uv run ruff format .` (CI enforces both)
- After editing dev dependencies in pyproject.toml, run `uv lock` (CI uses `uv sync --locked`)
- Keep `[project].dependencies` empty unless a runtime dependency is truly needed; ComfyUI-Manager installs from it

## Constraints

- Keep the UI DOM-based (no canvas-drawn widgets, no `node.properties` for data) so it
  works in both the classic canvas and Nodes 2.0 (Vue) rendering, and so all data
  reaches the API prompt.
- Missing/unloadable LoRAs and invalid regexes are warnings, never errors.
- Only `modelspec.trigger_phrase` is used for trigger auto-fill; don't guess from tags.
- Uses core endpoints only (`/models/loras`, `/view_metadata/loras`, `/userdata`); no
  custom server routes.
- Never commit `*.safetensors` or other model files, or `.env`.
- Use fictional names in tests and docs, not real people.
- Reference ComfyUI checkout for API lookups: `~/code/ComfyUI` (not the user's runtime install).
