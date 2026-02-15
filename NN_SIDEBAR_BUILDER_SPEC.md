# NN Sidebar Builder Spec

This file documents the current Neural Network sidebar builder UI so it can be reused as prompt context for future changes.

## Scope

- Sidebar and tab flow for the NN page (`Build`, `Train`, `Test`, `Deploy`).
- Build tab layout and controls.
- Behavior rules (add-layer flow, edit rules, validation flow, tab gating).
- Key class names and component boundaries.

## Source of Truth (Current Implementation)

- `frontend/src/App.tsx`
- `frontend/src/ui/tabs/BuildTab.tsx`
- `frontend/src/index.css`
- Related state/hooks:
  - `frontend/src/store/graphStore.ts`
  - `frontend/src/store/trainingStore.ts`
  - `frontend/src/utils/graphOrder.ts`
  - `frontend/src/utils/graphSerializer.ts`

## High-Level Sidebar Layout

In `App.tsx`, the left sidebar (`.app-sidebar`) contains:

1. Tab strip (`.app-tab-strip`) with 4 tabs:
   - `Build` (internal key: `validate`)
   - `Train`
   - `Test`
   - `Deploy`
2. One active tab panel rendered at a time:
   - `BuildTab` for builder/validation
   - `TrainTab`
   - `TestTab`
   - `DeployTab`

Main viewport is on the right (`.app-viewport-panel`) with:
- 3D graph canvas (`<Viewport />`)
- Sidebar collapse toggle
- Low-detail toggle
- Align button

## Build Tab Layout (Order + Meaning)

`BuildTab` (`frontend/src/ui/tabs/BuildTab.tsx`) renders sections in this order:

1. **Layer List Card** (`panel-card panel-card-layers`)
   - Ordered list of graph layers (`layerItems`)
   - Each row shows:
     - Role dot (`input`, `hidden`, `output`)
     - Display name
     - Size label
   - Ends with `Add Layer` button.

2. **Layer Editor Card** (`panel-card layer-editor-card`) shown only when a node is selected
   - Editable layer name (button -> inline input on edit)
   - Type readout
   - Primary fields row:
     - Size OR read-only shape
     - Activation select OR read-only “Not applicable”
   - Optional extra fields (by layer type):
     - Dense: `Units`
     - Dropout: `Dropout Rate`
     - Output: `Output Classes`

3. **Build Summary Card** (`panel-card build-summary-card`)
   - Metric tiles:
     - Layers
     - Neurons
     - Weights
     - Biases
     - Layer Type
     - Shared Activation Function

4. **Build Feedback Card** (`panel-card build-feedback-card`)
   - Status bar (`idle` / `success` / `error`)
   - Warning list
   - Error list

5. **Actions**
   - Primary button: `Build` (calls backend validation endpoint)

## BuildTab Prop Contract

Core input shape from parent (`App.tsx`):

- List/selection state:
  - `layerItems`, `selectedNodeId`, `hasSelectedNode`, `selectedNodeType`
- Name editing:
  - `isEditingName`, `draftName`, `selectedDisplayName`
- Field state:
  - `selectedRows`, `selectedCols`, `selectedUnits`, `selectedDropoutRate`,
    `selectedOutputClasses`, `selectedActivation`, `selectedShapeLabel`
- Field capability flags:
  - `canEditSize`, `canEditActivation`, `canEditUnits`,
    `canEditDropoutRate`, `canEditOutputClasses`
- Summary stats:
  - `layerCount`, `neuronCount`, `weightCount`, `biasCount`,
    `layerTypeSummary`, `sharedNonOutputActivation`
- Validation feedback:
  - `buildStatus`, `buildStatusMessage`, `buildIssues`, `buildWarnings`
  - `validateDisabled`, `validateLabel`
- Event handlers:
  - selection/edit handlers and `onValidate`

## Behavior Rules (Important)

### Add Layer flow (`handleAddLayer` in `App.tsx`)

Auto-builds a sequential NN chain with these priorities:

1. If no `Input` exists -> add `Input`.
2. If no `Flatten` exists -> insert `Flatten` before `Output` if needed.
3. If no `Dense` exists -> insert default `Dense`.
4. If no `Output` exists -> add `Output`.
5. Otherwise, insert new `Dense` before `Output`.

Dense defaults:
- `rows = 4`
- `cols = 6`
- `units = 24`
- `activation = linear`

### Editable fields by node type

- `Input`:
  - Edits image size via `shape: [1, H, W]` (rows/cols inputs).
- `Dense`:
  - Edits size, activation, units.
- `Dropout`:
  - Edits dropout rate.
- `Output`:
  - Edits activation, output classes.
- Other nodes:
  - Mostly read-only in current Build tab editor.

### Build/Validate flow

- Build button calls `POST /api/model/validate` with serialized graph.
- Success:
  - marks `hasValidatedModel = true`
  - switches active tab to `Train`
- Failure:
  - sets `buildStatus = error`
  - shows structured issues

### Tab gating rules

- `Train` tab enabled when model has been validated, or training already started.
- `Test` tab enabled after training complete / inference available.
- `Deploy` tab enabled after training complete or existing deployment state.

## Visual/Layout CSS Anchors

Primary classes in `frontend/src/index.css`:

- Shell:
  - `.app-shell`, `.app-sidebar`, `.app-sidebar-inner`, `.app-viewport-panel`
- Tabs:
  - `.app-tab-strip`, `.app-tab-button`, `.app-tab-indicator`
- Build panel:
  - `.tab-panel`, `.panel-card`, `.panel-card-layers`
- Layer list:
  - `.layer-list-shell`, `.layer-list`, `.layer-list-item`, `.layer-dot-*`,
    `.layer-list-add-button`
- Layer editor:
  - `.layer-editor-card`, `.layer-editor-fields-row`, `.layer-editor-extra-grid`
- Feedback:
  - `.build-feedback-status-*`, `.build-feedback-item-*`
- Actions:
  - `.panel-actions`, `.btn`, `.btn-validate`

## Prompt-Ready Reuse Block

Use this in future prompts when asking for sidebar-related changes:

```text
Preserve the NN sidebar builder architecture from NN_SIDEBAR_BUILDER_SPEC.md.
Do not change the core section order in BuildTab:
1) Layer list, 2) Layer editor (selected node only), 3) Summary tiles,
4) Build feedback, 5) Build action.
Keep App.tsx tab gating and add-layer sequential auto-insert behavior intact unless explicitly requested.
Any UI changes should be implemented with existing CSS class families used by the NN sidebar.
```

## Notes

- This spec is intentionally tied to current code behavior, not an abstract design.
- If implementation changes, update this file first before using it in new prompt chains.
