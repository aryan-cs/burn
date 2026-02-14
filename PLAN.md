# MLCanvas — Unity for Machine Learning

## Vision

MLCanvas is a visual, interactive IDE for designing, training, and interpreting machine learning models. Just as Unity democratized 3D game development and Scratch made programming accessible, MLCanvas lets users **see, touch, and manipulate** neural networks in real time — no boilerplate code required.

Users drag layer blocks onto a 3D canvas, wire them together, configure hyperparameters through intuitive panels, hit "Train," and watch weights flow through connections that glow and pulse as the model learns.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                      FRONTEND                           │
│                                                         │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  3D Viewport  │  │  2D Overlay  │  │  State Store │  │
│  │  (R3F/Three)  │  │  (React UI)  │  │  (Zustand)   │  │
│  └───────┬───────┘  └──────┬───────┘  └───────┬──────┘  │
│          │                 │                  │         │
│          └─────────────────┼──────────────────┘         │
│                            │                            │
│                    ┌───────┴────────┐                   │
│                    │  Graph Engine  │                   │
│                    │  (DAG Manager) │                   │
│                    └───────┬────────┘                   │
└────────────────────────────┼────────────────────────────┘
                             │
                   REST + WebSocket
                             │
┌────────────────────────────┼────────────────────────────┐
│                      BACKEND                            │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  API Layer   │  │ Graph→Model  │  │  Training    │   │
│  │  (FastAPI)   │  │  Compiler    │  │  Engine      │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐                     │
│  │  Dataset     │  │  Weight      │                     │
│  │  Manager     │  │  Streamer    │                     │
│  └──────────────┘  └──────────────┘                     │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Frontend — React + Three.js

### 2.1 Tech Stack

| Concern            | Library                        | Why                                                       |
| ------------------ | ------------------------------ | --------------------------------------------------------- |
| 3D rendering       | React Three Fiber (R3F) + Drei | React components that render to WebGL via Three.js        |
| 2D node graph UI   | Custom (or ReactFlow for MVP)  | Palette, property panels, minimap                         |
| State management   | Zustand                        | Lightweight, works seamlessly with R3F's render loop      |
| Styling            | Tailwind CSS                   | Rapid iteration for the 2D overlay                        |
| WebSocket client   | native `WebSocket` / socket.io | Receives live weight streams from backend during training  |
| Drag & transform   | @react-three/drei transforms   | Built-in drag, rotate, scale gizmos for 3D objects        |

### 2.2 Canvas & Viewport

The main screen is a **full-screen 3D viewport** rendered by React Three Fiber, overlaid with 2D React panels.

```
┌──────────────────────────────────────────────────┐
│ [Palette]  ┌──────────────────────────┐  [Props] │
│            │                          │          │
│  Dense     │      3D VIEWPORT         │  Units:  │
│  Conv2D    │                          │  [128]   │
│  LSTM      │   ┌───┐    ┌───┐         │          │
│  Dropout   │   │ L1├────┤ L2│         │  Act:    │
│  Pool      │   └───┘    └───┘         │  [relu]  │
│  Flatten   │                          │          │
│  Output    │                          │          │
│            └──────────────────────────┘          │
│                                                  │
│  [Train]  [Stop]  Epoch: 12  Loss: 0.032         │
└──────────────────────────────────────────────────┘
```

**Palette (left):** Draggable layer tiles. Drag one onto the canvas to create a new node.

**Properties Panel (right):** Context-sensitive. Click a node to see its config (units, activation, kernel size, etc.). Click an edge to see the weight statistics.

**Toolbar (bottom):** Train/Stop controls, live metrics (loss, accuracy), epoch counter.

### 2.3 3D Node Representation

Each **layer node** is a 3D mesh group:

```
LayerNode3D
├── Body          → Box or Cylinder mesh (color-coded by type)
├── Label         → Text (layer name + shape info)
├── InputPort     → Small sphere on the "in" face
├── OutputPort    → Small sphere on the "out" face
└── GizmoWrapper  → Drei's TransformControls (drag/rotate/scale)
```

**Layer type → visual mapping:**

| Layer Type | 3D Shape              | Color    |
| ---------- | --------------------- | -------- |
| Dense      | Rounded box           | #4A90D9  |
| Conv2D     | Flat wide box (grid)  | #7B61FF  |
| LSTM/GRU   | Cylinder (loop motif) | #F5A623  |
| Dropout    | Semi-transparent box  | #9B9B9B  |
| Pooling    | Shrinking trapezoid   | #50E3C2  |
| Activation | Glowing ring          | #E94E77  |
| Input      | Flat plane            | #4A4A4A  |
| Output     | Cone/arrow            | #D0021B  |

### 2.4 Edge / Connection Rendering

Connections between layers are **3D tube geometries** (using Three.js `TubeGeometry` or `Line2`) that follow a Catmull-Rom spline from the source output port to the target input port.

**Weight visualization on edges:**

| Property            | Visual Encoding                               |
| ------------------- | --------------------------------------------- |
| Mean weight         | Color on a diverging scale (blue ← 0 → red)  |
| Weight magnitude    | Line thickness (thicker = larger magnitude)   |
| Gradient flow       | Animated dashes flowing along the tube        |
| Dead connections    | Gray, thin, no animation                      |

During training, edges **animate**: small particles (or dashes) travel from input→output, speed proportional to gradient magnitude. This gives users an intuitive sense of "where learning is happening."

### 2.5 Interaction System

All interactions are handled through a layered input system:

```
Priority 1: 2D UI panels (clicks on buttons, sliders, dropdowns)
Priority 2: 3D gizmo manipulation (drag/rotate a selected node)
Priority 3: Canvas navigation (orbit, pan, zoom via OrbitControls)
```

**Key interactions:**

| Action             | Input                         | Result                                    |
| ------------------ | ----------------------------- | ----------------------------------------- |
| Add layer          | Drag from palette → canvas    | New node created at drop position         |
| Select layer       | Click on node                 | Highlights node, shows props panel        |
| Move layer         | Drag gizmo (translate mode)   | Repositions node, edges follow            |
| Rotate layer       | Drag gizmo (rotate mode)      | Rotational transform of the node          |
| Connect layers     | Click output port → input port| Creates edge, validates shape compat      |
| Delete             | Select + Backspace            | Removes node/edge, cascades disconnects   |
| Camera orbit       | Right-click drag              | Orbits camera around scene center         |
| Camera pan         | Middle-click drag             | Pans the camera                           |
| Zoom               | Scroll wheel                  | Dollies camera in/out                     |

### 2.6 Graph State (Zustand Store)

```typescript
interface GraphState {
  nodes: Record<string, LayerNode>;
  edges: Record<string, Edge>;

  // Actions
  addNode: (type: LayerType, position: Vec3) => string;
  removeNode: (id: string) => void;
  updateNode: (id: string, patch: Partial<LayerNode>) => void;
  addEdge: (sourceId: string, targetId: string) => string;
  removeEdge: (id: string) => void;

  // Serialization
  toJSON: () => GraphJSON;
  fromJSON: (json: GraphJSON) => void;

  // Training state
  trainingStatus: "idle" | "training" | "paused" | "complete";
  currentEpoch: number;
  metrics: { loss: number; accuracy: number }[];
  updateWeights: (weightSnapshot: WeightSnapshot) => void;
}

interface LayerNode {
  id: string;
  type: LayerType;
  position: [number, number, number];
  rotation: [number, number, number];
  config: LayerConfig;       // units, activation, kernel_size, etc.
  weights?: Float32Array;    // populated during/after training
  shape: {
    input: number[] | null;
    output: number[] | null;
  };
}

interface Edge {
  id: string;
  source: string;           // node ID
  target: string;           // node ID
  weightStats?: {
    mean: number;
    std: number;
    min: number;
    max: number;
  };
}

type LayerType =
  | "Input" | "Dense" | "Conv2D" | "MaxPool2D"
  | "LSTM" | "GRU" | "Dropout" | "BatchNorm"
  | "Flatten" | "Reshape" | "Output";
```

---

## 3. Backend — Python / FastAPI

### 3.1 Tech Stack

| Concern           | Library         | Why                                              |
| ----------------- | --------------- | ------------------------------------------------ |
| API framework     | FastAPI         | Async, WebSocket support, auto-docs              |
| ML framework      | PyTorch         | Dynamic graphs, easy introspection of weights    |
| Serialization     | Pydantic        | Validates graph JSON from frontend               |
| Dataset loading   | torchvision     | Built-in MNIST, CIFAR-10, etc. for MVP           |
| Task queue (v2)   | Celery + Redis  | For long-running training jobs (post-hackathon)  |

### 3.2 API Endpoints

```
REST Endpoints
──────────────
POST   /api/model/validate       → Validates graph, returns inferred shapes
POST   /api/model/compile         → Compiles graph → PyTorch model, returns summary
POST   /api/model/train           → Starts training (returns job ID)
POST   /api/model/stop            → Stops active training
GET    /api/model/export          → Returns saved .pt file or Python code
GET    /api/datasets              → Lists available datasets
POST   /api/datasets/upload       → Upload custom CSV/images (v2)

WebSocket Endpoints
───────────────────
WS     /ws/training/{job_id}      → Streams epoch metrics + weight snapshots
```

### 3.3 Graph → PyTorch Compiler

This is the core backend logic. It walks the user's graph DAG and produces an executable PyTorch model.

```python
# graph_compiler.py

import torch
import torch.nn as nn
from collections import OrderedDict

class DynamicModel(nn.Module):
    def __init__(self, layers: OrderedDict, edge_map: dict):
        super().__init__()
        self.layers = nn.ModuleDict(layers)
        self.edge_map = edge_map  # {target_id: [source_id, ...]}
        self.execution_order = topological_sort(edge_map)

    def forward(self, x):
        outputs = {}
        for node_id in self.execution_order:
            if node_id == self.input_id:
                outputs[node_id] = x
            else:
                # Gather inputs from predecessor nodes
                sources = self.edge_map[node_id]
                if len(sources) == 1:
                    inp = outputs[sources[0]]
                else:
                    inp = torch.cat([outputs[s] for s in sources], dim=-1)
                outputs[node_id] = self.layers[node_id](inp)
        return outputs[self.output_id]


def compile_graph(graph_json: dict) -> DynamicModel:
    """Translate a frontend graph JSON into a live PyTorch model."""
    layers = OrderedDict()
    prev_output_size = None

    for node in topological_sort_nodes(graph_json):
        ntype = node["type"]
        config = node["config"]

        if ntype == "Input":
            prev_output_size = config["shape"][-1]
            layers[node["id"]] = nn.Identity()

        elif ntype == "Dense":
            in_f = infer_input_features(node, graph_json)
            layers[node["id"]] = nn.Sequential(
                nn.Linear(in_f, config["units"]),
                get_activation(config.get("activation", "relu"))
            )

        elif ntype == "Conv2D":
            layers[node["id"]] = nn.Sequential(
                nn.Conv2d(
                    in_channels=config["in_channels"],
                    out_channels=config["filters"],
                    kernel_size=config["kernel_size"],
                    padding=config.get("padding", 0)
                ),
                get_activation(config.get("activation", "relu"))
            )

        elif ntype == "Dropout":
            layers[node["id"]] = nn.Dropout(p=config.get("rate", 0.5))

        elif ntype == "Flatten":
            layers[node["id"]] = nn.Flatten()

        elif ntype == "Output":
            in_f = infer_input_features(node, graph_json)
            layers[node["id"]] = nn.Linear(in_f, config["num_classes"])

        # ... more layer types

    edge_map = build_edge_map(graph_json["edges"])
    return DynamicModel(layers, edge_map)
```

### 3.4 Training Engine with Live Streaming

```python
# training_engine.py

import asyncio
import json
import torch
from fastapi import WebSocket

async def train_and_stream(
    model: nn.Module,
    dataloader,
    optimizer,
    loss_fn,
    epochs: int,
    websocket: WebSocket
):
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (output.argmax(1) == batch_y).sum().item()
            total += batch_y.size(0)

        # Collect weight statistics for visualization
        weight_snapshot = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                weight_snapshot[name] = {
                    "mean": param.data.mean().item(),
                    "std": param.data.std().item(),
                    "min": param.data.min().item(),
                    "max": param.data.max().item(),
                    "histogram": torch.histc(param.data, bins=20).tolist()
                }

        # Stream update to frontend
        await websocket.send_json({
            "type": "epoch_update",
            "epoch": epoch + 1,
            "loss": running_loss / len(dataloader),
            "accuracy": correct / total,
            "weights": weight_snapshot
        })

        # Check for stop signal
        try:
            msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
            if json.loads(msg).get("command") == "stop":
                break
        except asyncio.TimeoutError:
            pass
```

### 3.5 Shape Inference & Validation

Before training, the backend validates the graph by propagating tensor shapes:

```python
def validate_graph(graph_json: dict) -> dict:
    """Walk the graph and infer output shapes at every node.
       Returns errors if shapes are incompatible."""

    shapes = {}
    errors = []

    for node in topological_sort_nodes(graph_json):
        input_shapes = [shapes[src] for src in get_sources(node, graph_json)]

        try:
            output_shape = infer_shape(node, input_shapes)
            shapes[node["id"]] = output_shape
        except ShapeMismatchError as e:
            errors.append({
                "node_id": node["id"],
                "message": str(e),
                "expected": e.expected,
                "got": e.got
            })

    return {
        "valid": len(errors) == 0,
        "shapes": shapes,      # Sent back so frontend can display on nodes
        "errors": errors       # Frontend highlights incompatible connections
    }
```

This is critical for UX — when the user connects two incompatible layers, the edge turns **red** and a tooltip explains the shape mismatch.

---

## 4. Communication Protocol

### 4.1 Graph JSON Format (Frontend → Backend)

```json
{
  "nodes": [
    {
      "id": "node_1",
      "type": "Input",
      "config": { "shape": [1, 28, 28] }
    },
    {
      "id": "node_2",
      "type": "Conv2D",
      "config": { "filters": 32, "kernel_size": 3, "activation": "relu" }
    },
    {
      "id": "node_3",
      "type": "Flatten",
      "config": {}
    },
    {
      "id": "node_4",
      "type": "Dense",
      "config": { "units": 128, "activation": "relu" }
    },
    {
      "id": "node_5",
      "type": "Output",
      "config": { "num_classes": 10, "activation": "softmax" }
    }
  ],
  "edges": [
    { "id": "e1", "source": "node_1", "target": "node_2" },
    { "id": "e2", "source": "node_2", "target": "node_3" },
    { "id": "e3", "source": "node_3", "target": "node_4" },
    { "id": "e4", "source": "node_4", "target": "node_5" }
  ],
  "training": {
    "dataset": "mnist",
    "epochs": 20,
    "batch_size": 64,
    "optimizer": "adam",
    "learning_rate": 0.001,
    "loss": "cross_entropy"
  }
}
```

### 4.2 WebSocket Message Types (Backend → Frontend)

```
epoch_update     → { epoch, loss, accuracy, weights: {...} }
shape_update     → { node_id, input_shape, output_shape }
training_done    → { final_loss, final_accuracy, model_path }
error            → { message, node_id?, details? }
```

---

## 5. Project Structure

```
mlcanvas/
├── frontend/
│   ├── package.json
│   ├── tailwind.config.js
│   ├── public/
│   │   └── index.html
│   └── src/
│       ├── App.tsx                    # Root — canvas + overlay layout
│       ├── main.tsx                   # Entry point
│       │
│       ├── store/
│       │   ├── graphStore.ts          # Zustand store for the DAG
│       │   └── trainingStore.ts       # Training status, metrics
│       │
│       ├── canvas/
│       │   ├── Viewport.tsx           # R3F Canvas, camera, lights
│       │   ├── SceneManager.tsx       # Renders all nodes + edges from store
│       │   ├── nodes/
│       │   │   ├── LayerNode.tsx      # Generic 3D layer node wrapper
│       │   │   ├── DenseNode.tsx      # Dense-specific mesh
│       │   │   ├── ConvNode.tsx       # Conv2D-specific mesh
│       │   │   ├── LSTMNode.tsx       # LSTM-specific mesh
│       │   │   └── PortMesh.tsx       # Input/output port spheres
│       │   ├── edges/
│       │   │   ├── Connection.tsx     # 3D tube between ports
│       │   │   └── WeightVisual.tsx   # Color/thickness from weight data
│       │   └── controls/
│       │       ├── DragControls.tsx   # Node dragging in 3D
│       │       └── CameraRig.tsx      # Orbit/pan/zoom
│       │
│       ├── ui/
│       │   ├── Palette.tsx            # Sidebar with draggable layer tiles
│       │   ├── PropertiesPanel.tsx    # Config for selected node
│       │   ├── Toolbar.tsx            # Train/Stop, metrics display
│       │   ├── MetricsChart.tsx       # Live loss/accuracy graph
│       │   └── ShapeError.tsx         # Tooltip for mismatched shapes
│       │
│       ├── hooks/
│       │   ├── useWebSocket.ts        # Manages WS connection to backend
│       │   ├── useDragToCanvas.ts     # Palette drag → 3D drop
│       │   └── useConnectionDraw.ts   # Port-to-port edge drawing
│       │
│       └── utils/
│           ├── graphSerializer.ts     # Zustand state → JSON for backend
│           ├── colorScale.ts          # Weight value → RGB
│           └── shapeUtils.ts          # Client-side shape preview
│
├── backend/
│   ├── requirements.txt
│   ├── main.py                        # FastAPI app entry point
│   ├── routers/
│   │   ├── model.py                   # /api/model/* endpoints
│   │   ├── datasets.py                # /api/datasets/* endpoints
│   │   └── websocket.py               # /ws/training/* handler
│   ├── core/
│   │   ├── graph_compiler.py          # Graph JSON → PyTorch nn.Module
│   │   ├── shape_inference.py         # Propagate shapes, detect errors
│   │   ├── training_engine.py         # Training loop + WS streaming
│   │   └── weight_extractor.py        # Extract weight stats per epoch
│   ├── models/
│   │   ├── graph_schema.py            # Pydantic models for graph JSON
│   │   └── training_config.py         # Pydantic models for training opts
│   └── datasets/
│       ├── loader.py                  # Dataset loading (MNIST, CIFAR, etc.)
│       └── registry.py               # Available dataset metadata
│
└── README.md
```

---

## 6. Data Flow — End to End

Here's what happens when a user builds and trains a model:

```
Step 1: DESIGN
    User drags "Input" → canvas           → graphStore.addNode("Input", pos)
    User drags "Dense" → canvas           → graphStore.addNode("Dense", pos)
    User clicks Input.output → Dense.input → graphStore.addEdge(id1, id2)
    Props panel: set units=128, act=relu  → graphStore.updateNode(id, config)

Step 2: VALIDATE
    User clicks "Train"
    Frontend serializes graph              → graphStore.toJSON()
    POST /api/model/validate               → backend runs shape_inference
    Response: shapes per node, or errors   → frontend colors edges red/green

Step 3: COMPILE
    POST /api/model/compile                → backend runs graph_compiler
    Response: model summary (param count, layer list)
    Frontend shows summary in a modal

Step 4: TRAIN
    POST /api/model/train                  → backend starts training loop
    WS /ws/training/{job_id} opened        → frontend listens
    Each epoch:
      Backend sends epoch_update           → trainingStore.addMetric(...)
      Weight snapshot arrives              → graphStore.updateWeights(...)
      3D edges re-color/re-thickness       → visual feedback in real time
      MetricsChart updates                 → loss/accuracy curve extends

Step 5: INSPECT
    Training complete
    User clicks an edge                    → sees weight histogram
    User clicks a node                     → sees activation distribution
    User can rotate/zoom into any part of the model
```

---

## 7. Key Interaction Details

### 7.1 Drag from Palette to 3D Canvas

This requires bridging 2D DOM events with the 3D scene:

1. User mousedown on a palette tile (2D React).
2. A "ghost" element follows the cursor (CSS `position: fixed`).
3. On mouseup over the 3D canvas, fire a **raycast** from the mouse position into the scene.
4. The intersection point (or a default position if no hit) becomes the node's 3D position.
5. `graphStore.addNode(type, worldPosition)` creates the node.

### 7.2 Port-to-Port Connection Drawing

1. User clicks an **output port** (small sphere on a node's right face).
2. A temporary 3D line follows the mouse cursor (raycast to a plane).
3. User clicks an **input port** on another node.
4. Frontend calls `/api/model/validate` with the proposed edge.
5. If shapes are compatible → edge created, colored green.
6. If shapes mismatch → edge created but colored red, tooltip shows error.

### 7.3 Node Transform Gizmos

When a node is selected, `@react-three/drei`'s `<TransformControls>` attach to it:

- **Translate mode (W key):** Move the node in 3D space. All connected edges update their spline endpoints.
- **Rotate mode (E key):** Rotate the node mesh (purely cosmetic — doesn't affect the model).
- **Scale mode (R key):** Scale the node visualization (could map to layer width for visual clarity).

---

## 8. Hackathon MVP Scope

For TreeHacks, scope down to a compelling demo:

### Must Have (Day 1)
- [ ] 3D canvas with orbit controls
- [ ] Palette with: Input, Dense, Flatten, Output
- [ ] Drag-to-add nodes
- [ ] Click-to-connect ports
- [ ] Properties panel (units, activation)
- [ ] Backend shape validation with visual error feedback
- [ ] Graph → PyTorch compilation
- [ ] Train on MNIST with live loss chart

### Must Have (Day 2)
- [ ] WebSocket weight streaming
- [ ] Edges colored/sized by weight values
- [ ] Node drag in 3D with gizmos
- [ ] Training stop/resume
- [ ] Conv2D + MaxPool2D layers

### Nice to Have (Stretch)
- [ ] Animated gradient flow particles along edges
- [ ] Per-neuron visualization (expand a Dense layer to see individual neurons)
- [ ] Export model as Python code
- [ ] Custom dataset upload (CSV)
- [ ] Save/load graph to JSON file
- [ ] Activation heatmaps on hover
- [ ] LSTM/GRU layers

---

## 9. Getting Started — Quick Commands

```bash
# Frontend
cd frontend
npm create vite@latest . -- --template react-ts
npm install three @react-three/fiber @react-three/drei zustand tailwindcss

# Backend
cd backend
python -m venv venv && source venv/bin/activate
pip install fastapi uvicorn torch torchvision pydantic websockets

# Run
cd frontend && npm run dev          # → localhost:5173
cd backend && uvicorn main:app --reload  # → localhost:8000
```

---

## 10. Judging Criteria Alignment

| Criteria       | How MLCanvas Delivers                                                     |
| -------------- | ------------------------------------------------------------------------- |
| Technical      | Real-time 3D WebGL + live PyTorch training over WebSocket                 |
| Innovation     | No existing tool offers drag-and-drop 3D neural network design + live viz |
| Impact         | Lowers barrier to ML for students, educators, and visual learners         |
| Design         | Polished 3D interface with intuitive drag/connect/train workflow          |
| Completeness   | End-to-end: design → validate → train → visualize → export               |
