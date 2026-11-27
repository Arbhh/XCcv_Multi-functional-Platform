# XCcv Multi-functional Vision Platform · User Guide

Version V0.8.3 ｜ Five steps: **Annotation → Training → Testing → Deployment**
(with flow-chart orchestration and model tools)
Maintained by: Arbhh

Switch platforms from the left navigation on the main window; the inference-service
status shows at the top-right. Each platform runs as an independent process.

## 1. Annotation Platform

- Open: the "Annotation" card on the main window, or drag a dataset folder onto the
  desktop shortcut.
- Smart polygon (S key / "Create smart polygon"): left-click on the target to include,
  Shift+left-click to exclude, right-click to undo the last anchor, double-click or
  Ctrl+left-click to finish. The first use loads the SAM model in the background; the
  orange banner at the top shows progress and disappears when ready.
- Brush (B key): left-drag to paint, wheel or [ ] to resize; right-drag to erase.
- Plain polygon (N) / rectangle (R) / save Ctrl+S. Annotations are stored as
  labelme json.
- Batch pre-annotation: "Tools → AI batch pre-annotation" runs the model over a
  folder first, then humans correct.

## 2. Training Platform

- New project: pick the dataset directory (labelme layout) → pick the model
  (yolo11n / resnet18 / maskrcnn18/50; unsupervised anomaly detection cif /
  patchcore / subspacead) → training parameters (quick mode works out of the box;
  expert mode exposes everything).
- The training-free paradigm (anomaly detection) replaces epoch semantics with
  "support-set size / augmented samples" tiers (quick / balanced / quality gallery)
  in quick mode.
- "Negative-sample training" is on by default (OK negatives suppress false
  detections); "P2 small-object head" as needed; "data health check" and
  "recommended configuration" help with parameter decisions.
- After training, review curves and metrics in "History"; "Compare selected"
  compares two runs.

## 3. Testing Platform

- Load a model: the project's latest model (best.pt) or pick a weights file
  (.pt/.engine) directly.
- Single-image prediction / batch testing (output directory, confidence); tick
  "tiled inference" for 4K images.
- Project review: run a batch once → drag sliders in the threshold panel to
  re-group instantly (no re-inference); double-click a thumbnail to inspect the
  overlay in the platform (green = above-threshold detections, red = filtered;
  anomaly/classification shows a score banner or predicted class); "Apply to
  detection flow" and "Generate flow template" offer a one-click "Open flow
  platform" hand-off.
- UI language: switch 中文/English in the main window's "Settings"; windows
  rebuild automatically.

## 4. Model Tools

- **Acceleration**: weights → TensorRT engine (.engine). Input size accepts `640`
  or `HxW` (e.g. `586x1920`; rectangular works best when it matches training);
  export needs an NVIDIA GPU + TensorRT. Accelerated engines load directly in the
  testing platform.
- **Pruning**: pre-training pruning for classifiers such as resnet18 (the YOLO
  family requires an expert-mode environment variable).
- **Defect synthesis**: compose defect samples on a web canvas; assets persist
  across sessions.
- **Flow chart**: 57 built-in operators (preprocess / locate / measure / logic /
  AI bridge / detection filters / output) mix on one canvas; five official
  templates to start from; batch runs over a directory output unified JSON +
  overlay images; a flow can be packed with its models into a plaintext .xvap
  package for the testing platform and the inference service.

## 5. Deployment Platform (inference service / model loading / C# test)

- **Inference service**: click "Start service"; default `127.0.0.1:8600` with an
  auto-generated API Key; logs go to `platform_data\logs\service.log` (startup
  failures are recorded there too). The service enforces upload guardrails
  (request body / pixel budget / batch size; over-limit returns 400/413 with a
  readable message); management operations such as remote shutdown require the
  API Key.
- **Model loading**: load .pt/.engine into a running service (plaintext
  artifacts, models stay in memory). The top of the window mirrors the real
  service state; loading is disabled while the service is down.
- **C# test**: one-click build of the Demo client that connects to the service
  (single-file SDK, .NET Framework 4.5+, compiled by the system csc).

## 6. Directory Layout (source run)

```
<repo root>\
  xccv\ labelme\    ← platform source (Python packages)
  run_main.bat      ← main workbench launcher (the only launcher at root)
  install.bat       ← one-click environment installer entry
  scripts\          ← per-platform launch bats (repo-root backtracking)
                       and run_tools.py
  docs\             ← user guide / operation manual / result contract
                       (result_schema.md) and the project introduction
  xccv\engine\weights\ ← engine weights folder (not distributed with the
                       repo, family subfolders below)
  platform_data\    ← user data (logs / projects / training artifacts)
```

Logs live under `platform_data\logs\` (crash.log / main.log / training.log /
service.log, etc.). Model weights and sample data are not distributed with the
repository: baseline weights (YOLO / torchvision / SAM / DINO) can be fetched
with `python tools/download_weights.py` (`--list` for the manifest); place the
rest per the "weights and sample data" table in the README; defect-synthesis
sample images ship as a separate data package on the release page.

## 7. FAQ

- **Engine load reports "GPU/TensorRT required"**: a .engine is bound to the GPU
  compute capability and TensorRT version — use the same machine it was built
  on, or fall back to .pt.
- **Smart polygon does not respond**: wait until the top banner disappears (the
  model is ready); when weights are missing the status bar shows the expected
  path.
- **Inference service fails to start**: check the tail of
  `platform_data\logs\service.log`; if the port is taken, change it or stop the
  stale python process.
- **Upload returns 400/413**: the request body, per-image pixel budget or batch
  size exceeds the service limits (defaults: 100 MB / 50M pixels / 32 per
  batch); reduce the batch or the resolution and retry.
