// Frontend for the Auto LoRA Loader node: a DOM-based row editor stored in the
// node's hidden "loras" widget as a JSON string. DOM widgets are re-parented into
// Vue nodes ("Nodes 2.0"), so the same code serves both rendering modes.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const WIDGET_TYPE = "AUTO_LORA_ROWS"; // must match ROWS_WIDGET_TYPE in node.py
const PRESET_DIR = "auto_lora_presets";
const TRIGGER_METADATA_KEY = "modelspec.trigger_phrase";

// Widget sizing on the litegraph canvas. The real content height is measured;
// the estimates are only used before the element has been laid out.
const MARGIN = 6;
const EST_BAR_HEIGHT = 26;
const EST_ROW_HEIGHT = 26;
const MIN_WIDTH = 640;

const STYLE = `
.auto-lora { font-size: 12px; color: var(--input-text, #ddd); box-sizing: border-box;
  width: 100%; overflow-y: auto; }
.auto-lora-content { display: flex; flex-direction: column; gap: 4px; }
.auto-lora * { box-sizing: border-box; font: inherit; }
.auto-lora input, .auto-lora select, .auto-lora button {
  background: var(--comfy-input-bg, #222); color: var(--input-text, #ddd);
  border: 1px solid var(--border-color, #444); border-radius: 4px; padding: 2px 4px; min-width: 0; }
.auto-lora button { cursor: pointer; }
.auto-lora button:hover { filter: brightness(1.2); }
.auto-lora-bar { display: flex; gap: 4px; align-items: center; }
.auto-lora-bar select { flex: 1; }
.auto-lora-rows { display: flex; flex-direction: column; gap: 3px; }
/* Every row (and the column header) shares one template so the columns line up:
   enabled | LoRA | triggers | regex | strength | up | down | remove */
.auto-lora-row { display: grid; gap: 4px; align-items: center;
  grid-template-columns: 16px minmax(140px, 1.4fr) minmax(120px, 1fr) 22px 56px 22px 22px 22px; }
.auto-lora-row.disabled > :not(.enabled) { opacity: 0.5; }
.auto-lora-row input[type=checkbox] { margin: 0; justify-self: center; }
.auto-lora-head { opacity: 0.6; font-size: 11px; }
.auto-lora-head > * { overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.auto-lora-head .center { text-align: center; }
.auto-lora .icon { width: 22px; padding: 2px 0; text-align: center; }
.auto-lora .invalid { border-color: var(--error-text, #f55); }
.auto-lora .missing { color: var(--error-text, #f55); }
.auto-lora-empty { opacity: 0.6; text-align: center; padding: 6px; }
`;

function injectStyle() {
  if (document.getElementById("auto-lora-style")) return;
  const style = document.createElement("style");
  style.id = "auto-lora-style";
  style.textContent = STYLE;
  document.head.append(style);
}

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key in node) node[key] = value;
    else node.setAttribute(key, value);
  }
  node.append(...children);
  return node;
}

// ---------------------------------------------------------------------------
// Row data

function newRow(lora = "") {
  return { enabled: true, lora, triggers: "", regex: false, strength: 1.0 };
}

function normalizeRow(row) {
  const strength = Number(row?.strength);
  return {
    enabled: row?.enabled !== false,
    lora: String(row?.lora ?? ""),
    triggers: String(row?.triggers ?? ""),
    regex: Boolean(row?.regex),
    strength: Number.isFinite(strength) ? strength : 1.0,
  };
}

function parseRows(value) {
  try {
    const data = typeof value === "string" ? JSON.parse(value || "[]") : value;
    return Array.isArray(data) ? data.map(normalizeRow) : [];
  } catch {
    return [];
  }
}

// The backend compiles with Python's `re`; JavaScript's RegExp is close enough
// to flag obvious mistakes while editing.
function regexError(expr) {
  if (!expr.trim()) return null;
  try {
    new RegExp(expr, "i");
    return null;
  } catch (err) {
    return err.message;
  }
}

function displayName(lora) {
  return lora.replace(/\.safetensors$/i, "");
}

// ---------------------------------------------------------------------------
// Server calls

let loraListPromise = null;

function getLoraList(refresh = false) {
  if (refresh || !loraListPromise) {
    loraListPromise = api
      .fetchApi("/models/loras")
      .then((r) => (r.ok ? r.json() : []))
      .then((list) => [...list].sort((a, b) => a.localeCompare(b)))
      .catch(() => []);
  }
  return loraListPromise;
}

async function getTriggerPhrase(lora) {
  if (!lora.endsWith(".safetensors")) return null;
  try {
    const resp = await api.fetchApi(`/view_metadata/loras?filename=${encodeURIComponent(lora)}`);
    if (!resp.ok) return null;
    const metadata = await resp.json();
    const phrase = metadata?.[TRIGGER_METADATA_KEY];
    return typeof phrase === "string" && phrase.trim() ? phrase.trim() : null;
  } catch {
    return null;
  }
}

function presetPath(name) {
  return `${PRESET_DIR}/${name}.json`;
}

async function listPresets() {
  try {
    const files = await api.listUserDataFullInfo(PRESET_DIR);
    return files
      .map((f) => f.path)
      .filter((p) => p.endsWith(".json") && !p.includes("/"))
      .map((p) => p.slice(0, -".json".length))
      .sort((a, b) => a.localeCompare(b));
  } catch {
    return [];
  }
}

async function loadPreset(name) {
  const resp = await api.getUserData(presetPath(name));
  if (!resp.ok) throw new Error(`Could not load preset "${name}" (${resp.status})`);
  const data = await resp.json();
  return parseRows(Array.isArray(data) ? data : data?.rows);
}

async function savePreset(name, rows) {
  await api.storeUserData(presetPath(name), { version: 1, rows });
}

// ---------------------------------------------------------------------------
// Dialogs (ComfyUI's own when available, browser fallbacks otherwise)

async function promptText(title, message, defaultValue = "") {
  const dialog = app.extensionManager?.dialog;
  if (dialog?.prompt) return dialog.prompt({ title, message, defaultValue });
  return window.prompt(message, defaultValue);
}

async function confirmAction(title, message) {
  const dialog = app.extensionManager?.dialog;
  if (dialog?.confirm) return (await dialog.confirm({ title, message, type: "delete" })) === true;
  return window.confirm(message);
}

function toast(severity, summary, detail) {
  const toastApi = app.extensionManager?.toast;
  if (toastApi?.add) toastApi.add({ severity, summary, detail, life: 4000 });
  else console[severity === "error" ? "error" : "log"](`[Auto LoRA] ${summary}: ${detail}`);
}

function sanitizePresetName(name) {
  return (name ?? "").trim().replace(/[\\/:*?"<>|]/g, "_");
}

// ---------------------------------------------------------------------------
// Widget

function createRowsWidget(node, inputName, inputData) {
  injectStyle();
  let rows = parseRows(inputData?.[1]?.default ?? "[]");
  let loras = [];

  const container = el("div", { class: "auto-lora" });
  const content = el("div", { class: "auto-lora-content" });
  const presetSelect = el("select", { title: "Saved presets" });
  const rowList = el("div", { class: "auto-lora-rows" });

  const widget = node.addDOMWidget(inputName, WIDGET_TYPE, container, {
    getValue: () => JSON.stringify(rows),
    setValue: (value) => {
      rows = parseRows(value);
      render();
    },
    margin: MARGIN,
    getMinHeight: contentHeight,
    getMaxHeight: contentHeight,
  });

  // ComfyUI sizes DOM widgets as `(widget.width ?? node.width) - margin * 2`. Some other
  // extensions assign widget.width, which would pin the editor to a fixed width while the
  // node is resized; ignore such assignments so the editor always follows the node.
  Object.defineProperty(widget, "width", { configurable: true, get: () => undefined, set: () => {} });

  // Height the widget needs: the rendered content if it has been laid out, otherwise an estimate.
  function contentHeight() {
    const measured = content.offsetHeight;
    const estimate = EST_BAR_HEIGHT * 2 + EST_ROW_HEIGHT * (rows.length + 1);
    return (measured > 0 ? measured : estimate) + 2 * MARGIN;
  }

  function changed() {
    widget.callback?.(widget.value);
    node.graph?.setDirtyCanvas?.(true, true);
  }

  // Snap the node's height to its content (growing or shrinking) after the DOM updates.
  function fitNode() {
    requestAnimationFrame(() => {
      if (!node.computeSize || !node.setSize) return;
      // Workflows saved with an older, narrower layout are widened too.
      const width = Math.max(node.size[0], MIN_WIDTH);
      const height = node.computeSize()[1];
      if (width !== node.size[0] || Math.abs(node.size[1] - height) > 1) node.setSize([width, height]);
      node.graph?.setDirtyCanvas?.(true, true);
    });
  }

  function update(index, patch) {
    rows[index] = { ...rows[index], ...patch };
    changed();
  }

  async function autofillTriggers(index) {
    const lora = rows[index].lora;
    const phrase = await getTriggerPhrase(lora);
    // Only fill if the row still points at the same LoRA and the user hasn't typed anything.
    if (phrase && rows[index]?.lora === lora && !rows[index].triggers.trim()) {
      update(index, { triggers: phrase });
      render();
    }
  }

  function renderRow(row, index) {
    const loraSelect = el("select", { class: "lora", title: row.lora });
    loraSelect.append(el("option", { value: "", textContent: "— select a LoRA —" }));
    if (row.lora && !loras.includes(row.lora)) {
      loraSelect.append(el("option", { value: row.lora, textContent: `${displayName(row.lora)} (missing)` }));
      loraSelect.classList.add("missing");
    }
    for (const name of loras) loraSelect.append(el("option", { value: name, textContent: displayName(name) }));
    loraSelect.value = row.lora;
    loraSelect.addEventListener("change", () => {
      update(index, { lora: loraSelect.value });
      render();
      if (loraSelect.value) autofillTriggers(index);
    });

    const triggers = el("input", {
      class: "triggers",
      type: "text",
      value: row.triggers,
      placeholder: row.regex ? "regular expression (use | for alternatives)" : "trigger words, comma separated",
      spellcheck: false,
    });
    const markRegex = () => {
      const error = rows[index].regex ? regexError(triggers.value) : null;
      triggers.classList.toggle("invalid", Boolean(error));
      triggers.title = error ? `Invalid regex: ${error}` : "";
    };
    triggers.addEventListener("input", () => {
      update(index, { triggers: triggers.value });
      markRegex();
    });
    markRegex();

    const strength = el("input", {
      class: "strength",
      type: "number",
      step: "0.05",
      value: String(row.strength),
      title: "Strength",
    });
    strength.addEventListener("change", () => {
      const value = Number(strength.value);
      update(index, { strength: Number.isFinite(value) ? value : 1.0 });
    });

    const enabled = el("input", { class: "enabled", type: "checkbox", checked: row.enabled, title: "Enabled" });
    enabled.addEventListener("change", () => {
      update(index, { enabled: enabled.checked });
      render();
    });

    const regex = el("input", {
      type: "checkbox",
      checked: row.regex,
      title: "Treat the triggers as one regular expression",
    });
    regex.addEventListener("change", () => {
      update(index, { regex: regex.checked });
      render();
    });

    const move = (delta) => {
      const target = index + delta;
      if (target < 0 || target >= rows.length) return;
      [rows[index], rows[target]] = [rows[target], rows[index]];
      changed();
      render();
    };

    const remove = () => {
      rows.splice(index, 1);
      changed();
      render();
    };

    return el(
      "div",
      { class: `auto-lora-row${row.enabled ? "" : " disabled"}` },
      enabled,
      loraSelect,
      triggers,
      regex,
      strength,
      el("button", { class: "icon", textContent: "↑", title: "Move up", disabled: index === 0, onclick: () => move(-1) }),
      el("button", {
        class: "icon",
        textContent: "↓",
        title: "Move down",
        disabled: index === rows.length - 1,
        onclick: () => move(1),
      }),
      el("button", { class: "icon", textContent: "✕", title: "Remove", onclick: remove })
    );
  }

  function renderHeader() {
    return el(
      "div",
      { class: "auto-lora-row auto-lora-head" },
      el("span"),
      el("span", { textContent: "LoRA" }),
      el("span", { textContent: "Triggers" }),
      el("span", { class: "center", textContent: ".*", title: "Regex" }),
      el("span", { textContent: "Strength" })
    );
  }

  function render() {
    rowList.replaceChildren(
      ...(rows.length
        ? [renderHeader(), ...rows.map(renderRow)]
        : [el("div", { class: "auto-lora-empty", textContent: "No LoRAs configured. Click “Add LoRA”." })])
    );
    fitNode();
  }

  async function refreshPresets(selected = presetSelect.value) {
    const names = await listPresets();
    presetSelect.replaceChildren(
      el("option", { value: "", textContent: names.length ? "— presets —" : "— no presets —" }),
      ...names.map((n) => el("option", { value: n, textContent: n }))
    );
    presetSelect.value = names.includes(selected) ? selected : "";
  }

  async function onLoadPreset() {
    const name = presetSelect.value;
    if (!name) return toast("warn", "Auto LoRA", "Choose a preset to load.");
    if (rows.length && !(await confirmAction("Load preset", `Replace the current ${rows.length} row(s) with preset "${name}"?`)))
      return;
    try {
      rows = await loadPreset(name);
      changed();
      render();
    } catch (err) {
      toast("error", "Auto LoRA", err.message);
    }
  }

  async function onSavePreset() {
    const name = sanitizePresetName(await promptText("Save preset", "Preset name:", presetSelect.value));
    if (!name) return;
    const existing = await listPresets();
    if (existing.includes(name) && !(await confirmAction("Overwrite preset", `Overwrite preset "${name}"?`))) return;
    try {
      await savePreset(name, rows);
      await refreshPresets(name);
      toast("success", "Auto LoRA", `Saved preset "${name}".`);
    } catch (err) {
      toast("error", "Auto LoRA", err.message);
    }
  }

  async function onDeletePreset() {
    const name = presetSelect.value;
    if (!name) return toast("warn", "Auto LoRA", "Choose a preset to delete.");
    if (!(await confirmAction("Delete preset", `Delete preset "${name}"?`))) return;
    await api.deleteUserData(presetPath(name));
    await refreshPresets("");
  }

  async function onRefresh() {
    loras = await getLoraList(true);
    await refreshPresets();
    render();
  }

  content.append(
    el(
      "div",
      { class: "auto-lora-bar" },
      presetSelect,
      el("button", { textContent: "Load", title: "Replace rows with the selected preset", onclick: onLoadPreset }),
      el("button", { textContent: "Save", title: "Save rows as a preset", onclick: onSavePreset }),
      el("button", { class: "icon", textContent: "🗑", title: "Delete the selected preset", onclick: onDeletePreset })
    ),
    rowList,
    el(
      "div",
      { class: "auto-lora-bar" },
      el("button", {
        textContent: "＋ Add LoRA",
        style: "flex: 1",
        onclick: () => {
          rows.push(newRow());
          changed();
          render();
        },
      }),
      el("button", { class: "icon", textContent: "⟳", title: "Refresh LoRA and preset lists", onclick: onRefresh })
    )
  );
  container.append(content);

  render();
  getLoraList().then((list) => {
    loras = list;
    render();
  });
  refreshPresets();

  return { widget, minWidth: MIN_WIDTH };
}

app.registerExtension({
  name: "xiphux.AutoLoraLoader",
  getCustomWidgets() {
    return { [WIDGET_TYPE]: createRowsWidget };
  },
});
