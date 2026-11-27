# XCcv Multi-functional Platform

[![CI](https://github.com/Arbhh/XCcv_Multi-functional-Platform/actions/workflows/ci.yml/badge.svg)](https://github.com/Arbhh/XCcv_Multi-functional-Platform/actions/workflows/ci.yml)

[中文](README.md) | English

XCcv is a fully open-source, **offline desktop** multi-functional deep learning vision
platform for industrial vision scenarios. It covers the complete pipeline of
**data annotation → model training → testing & evaluation → deployment inference service**,
plus **model tools** (structured pruning / TensorRT acceleration), a **flowchart
orchestrator** (57 built-in operators) and a **defect sample synthesis** tool.
A first-time user can take a vision model from raw data to trained weights — and to a
production inference service — entirely through the desktop UI, running fully offline so
data never leaves the local network.
The UI supports **one-click Chinese / English switching**, applied across the
whole platform.

![XCcv main window (AI platform)](docs/images/main.png)

> Main window: Annotate → Train → Test → Deploy in four steps; the left navigation
> opens Traditional Platform and Flowchart, and the cards include Model Tools and the
> Deployment platform with a live inference-service status pill.

## Feature Overview

| Module | Capabilities |
|---|---|
| Annotation | Integrated labelme annotator: rectangles / polygons, autosave, class management; **AI-assisted annotation** (S smart polygon, B brush shaping, L single-image smart annotation, folder batch pre-annotation, based on SAM / EfficientSAM); OK/good-part folder badges, annotation quality tips, AI review of suspected missed labels |
| Training | Project-based management; **5 task types, 14 built-in models** (classification / detection / segmentation / unsupervised and few-shot anomaly detection PatchCore, CIF, SubspaceAD / OCR CRNN, incl. ResNet / MobileNetV3 / YOLO11 / Faster R-CNN / Mask R-CNN); quick-config and expert-mode dual-track parameters; industrial optimizations (CBAM/ECA, DCNv2, Focal-family losses); training accelerations (frozen backbone, AMP, TF32, early stop, grayscale, deterministic runs); P2 small-object head, negative-sample training, tiled training; per-class metrics table, data grid, split donut chart redrawn live with parameters, real-time preview of all augmentation operators, loss diagnosis, run manifest, data health check, multi-run comparison |
| Testing | Single-image / batch inference; live confidence and IOU threshold tuning; class filtering; 4K tiled inference; unified JSON / labelme / COCO export; **Standard / Project Review dual pages** (evaluated images group into Correct / Missed / Over-kill, unevaluated production images group into OK / NG; dimension-threshold dragging with sweep curves in the same semantics); review reflow; model A/B comparison (cached results + overall verdict bar); latency / VRAM statistics; score semantics follow the model declaration (confidence / probability / image score z) |
| Deployment | **HTTP inference service** (FastAPI): load models/flows as resident APIs, single-frame & batch inference, overlay rendering, WebSocket binary-frame streaming, per-stage timing (`X-XCcv-Timing` header); API-key auth, concurrency gating, model path whitelist, self-hosted offline Swagger docs; a **Model Loading** window and a **C# Test** demo (`service/csharp` single-file SDK + WinForms demo, compiled with the system csc); one-click consistency check (local inference vs service inference, box-by-box) |
| Model Tools | **Structured channel pruning** (Torch-Pruning DepGraph; pre-training pruning for transfer learning / post-training pruning, isolated subprocess with RAM watchdog and emergency stop); **TensorRT engine export** (YOLO via the native ultralytics path; classification via in-memory ONNX→TRT; Mask R-CNN feature-segment acceleration with cross-GPU compute-capability precheck and friendly fallback) |
| Flowchart | A standalone module at the same level as the AI platform: **57 built-in AI & common operators** (model inference / classic vision operators: blob analysis, template matching, find line/circle, measurement / detection filtering / crop & merge / logic verdict / external-operator plugin backend / output); drag-and-drop web canvas, node library split into "AI operators / Classic operators"; run preview, save JSON, pack / unpack `.xvap`, batch runs; 9 templates (AI filter, cascade crop-merge, detection→segmentation cascade, template-match prefilter, etc.) |
| Defect Sample Synthesis | Blends defect patches onto good-part images with Poisson fusion to expand scarce defect samples (separate process with an embedded web UI) |
| UI Language | **One-click Chinese / English switching**: choose in the workbench "Settings" dialog; the whole platform switches automatically; resident UI texts and frequent messages are translated |

> The flowchart module orchestrates AI and classic vision operators visually
> (66 nodes / 9 templates), covering model inference, classic locate/measure steps,
> detection filtering, crop & merge, logic verdicts and result output for production lines.

## Requirements

- Windows 10 / 11 (x64)
- Python 3.11 (a dedicated virtual environment is recommended)
- GPU: optional. An NVIDIA GPU greatly speeds up training and inference; CPU also works
  (small models recommended)
- Disk: dependencies (PyTorch etc.) need about 6–8 GB
- Optional: TensorRT runtime (needed for model acceleration / `.engine` inference;
  `pip install tensorrt`, or set `TENSORRT_DIR` to a folder containing `nvinfer_10.dll`)

## Installation

**One-click install (recommended)**: double-click `install.bat` in the repo root —
it detects your GPU and driver, picks the torch index, creates the `venv`, installs the
dependencies, writes `python_path.txt`, and finishes with a bilingual environment
diagnosis; failure branches (no GPU / offline / proxy / permission) carry guidance.

**Manual install** (equivalent steps):

```bat
:: 1) Create and activate a virtual environment (Python 3.11)
python -m venv venv
venv\Scripts\activate

:: 2) Install PyTorch (version matrix: torch==2.9.1 + torchvision==0.24.1, same as CI;
::    pick the index by GPU, CUDA 12.8 shown; ultralytics 8.4.7 requires torch<2.10 —
::    do not install torch 2.10 or newer)
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

:: 3) Install the remaining dependencies
pip install -r requirements.txt
```

> **Torch index and version matrix**: versions are pinned to torch==2.9.1 +
> torchvision==0.24.1. Pick the index by hardware — recent NVIDIA drivers / RTX 50
> series use `cu128`, older drivers may use `cu126`; **CI uses the CPU index**
> (`--index-url https://download.pytorch.org/whl/cpu`), and machines without an NVIDIA
> GPU use the CPU index too. The three sources (README / requirements.txt comment /
> ci.yml) are kept consistent by the dependency check in `tools/doc_reconcile.py`.

## Quick Start

> **Launcher environment (important)**: the workbench, flow canvas and defect synthesis all
> depend on PyQtWebEngine. The root `run_main.bat` and `install.bat` are pure-ASCII and
> **do not require a venv in this folder**; they probe candidates in this order; the
> other platform entries live in `scripts\` (they locate the repo root automatically
> and work from anywhere):
>
> 1. the `XCCV_PY` environment variable;
> 2. `python_path.txt` in this folder (one line: full path to python.exe);
> 3. `python` on PATH;
> 4. the `py` launcher;
> 5. `venv` / `.venv` in this folder.
>
> If none works, the scripts print English guidance and run `tools\env_check.py` for a detailed
> diagnosis. It is recommended to write one line into `python_path.txt` on the target machine,
> e.g. `<project>\venv\Scripts\python.exe` (the file is gitignored).

Double-click `run_main.bat` in the repo root (or `python -m xccv.app`) to open the
workbench, then enter each platform from its card:

1. **Annotate**: create a project → import images → draw rectangles / polygons
   (optionally press S for AI-assisted polygons);
2. **Train**: pick the project and model (classification / detection / segmentation /
   anomaly detection / OCR) → quick config or expert mode → start training with live curves
   and per-class metrics;
3. **Test**: load the project's latest model (best.pt) → single-image / batch inference →
   results on the Standard page, or switch to the "Project Review" page: tune thresholds under
   "Evaluated" (Correct / Missed / Over-kill), then switch to "Unevaluated" to check production
   images as OK / NG; review reflow and A/B comparison supported;
4. **Deployment platform**: start the HTTP inference
   service → load models/flows as resident APIs → single / batch / WebSocket inference;
   the C# Test card compiles and runs the WinForms demo in one click;
5. **Flowchart**: pick the data path and a flow JSON → "Run Flow" for
   single-image results, or "Edit Flow" to orchestrate AI / classic operators on the canvas,
   pack as `.xvap` and run batches;
6. **Model Tools**: pre/post-training pruning for lightweight models; export best.pt to a
   TensorRT engine (.engine) for faster production inference;
7. **Defect Sample Synthesis** (optional): enter from the workbench card to expand the
   training set;
8. **UI Language**: workbench "Settings" → "Language" → English; all platform windows switch
   automatically after saving.

Command-line entries (equivalent to the workbench cards, all under `scripts\`):

```bat
scripts\run_annotation.bat   :: Annotation platform
scripts\run_training.bat     :: Training platform
scripts\run_testing.bat      :: Testing platform
scripts\run_flow.bat         :: Flowchart (standalone window; also in the left navigation)
scripts\run_modeltools.bat   :: Model tools (pruning / TensorRT acceleration)
scripts\run_deploy.bat       :: Deployment platform (inference service / model loading / C# test)
scripts\run_service.bat      :: Inference service (headless service process)
scripts\run_defectgen.bat    :: Defect sample synthesis
```

The inference service can also start directly: `python -m xccv.service.main --host 127.0.0.1
--port 8600` (an API key is generated on first start into
`platform_data/service/config.json`; API docs at `http://127.0.0.1:8600/docs`).
C# demo: one-click build via `xccv\service\csharp\build_demo.bat` (uses the built-in
.NET Framework csc, no third-party dependencies).

For the detailed getting-started guide see
[docs/quickstart_EN.md](docs/quickstart_EN.md) (中文: [docs/quickstart.md](docs/quickstart.md)).

**Documentation index**: project introduction in
[docs/XCcv多功能视觉平台_项目简介_EN.docx](docs/XCcv多功能视觉平台_项目简介_EN.docx)
(中文: [docs/XCcv多功能视觉平台_项目简介.docx](docs/XCcv多功能视觉平台_项目简介.docx));
FAQs on environment, weights, annotation, training and deployment in
[docs/FAQ_EN.md](docs/FAQ_EN.md) (中文: [docs/FAQ.md](docs/FAQ.md)).

### Optional: weights and sample data

> **This repository ships no model weights and no sample data.** Weights follow their
> publishers' licenses and must be obtained by the user; sample data is distributed as a
> separate data package. The repo and source package contain only code, docs and UI icons.

| Resource | How to obtain | Where to put it |
|---|---|---|
| SAM family (AI-assisted annotation) | `python tools/download_weights.py --family sam` (license notes printed along) | `xccv/engine/weights/sam/` |
| DINO ViT-B/8 + DINOv2 (CIF / SubspaceAD) | `python tools/download_weights.py --family dino`; mainland networks may set `HF_ENDPOINT=https://hf-mirror.com` | `xccv/engine/weights/dino/`, `xccv/engine/weights/dinov2/` (or `XCCV_DINO_CKPT` / `XCCV_DINOV2_CKPT`) |
| YOLO COCO / torchvision baselines | `python tools/download_weights.py --family yolo` (reuses the training-pipeline multi-source download and checks) | family subfolders under `xccv/engine/weights/` (`--list` shows targets) |
| Defect-synthesis sample data (~33 MB) | Download `XCcv-Platform-testdata-0.8.0.zip` | Extract under `xccv/defectgen/` (yields `testdata/`) |
| maskrcnn18 slim initial weights | Build with `python tools/build_maskrcnn18_init.py`, or convert mmdet weights with `tools/convert_mmdet_r18.py` | `xccv/engine/weights/` |

When a resource is missing, the affected features show clear guidance and everything else
keeps working. All model weights live in the engine weights folder
`xccv/engine/weights/` (installed state: `<install root>/weights/`);
`.gitignore` excludes its runtime artifacts, `xccv/defectgen/testdata/` and
`*.pt / *.pth / *.onnx / *.engine / *.xva / *.xcs`.

## Repository Layout

```
XCcv-Multi-functional-Platform/
├── xccv/                  # Main package
│   ├── annotation/        #   Annotation platform (labelme host + AI-assisted annotation)
│   ├── training/          #   Training platform
│   ├── testing/           #   Testing platform (Standard + Project Review pages)
│   ├── deploy_platform.py #   Deployment platform (service window / model loading / C# test)
│   ├── service/           #   HTTP inference service (FastAPI + C# SDK and demo)
│   ├── modeltools/        #   Model-tools window (pruning / TensorRT / Toolbox) + flow canvas host
│   ├── defectgen/         #   Defect sample synthesis (FastAPI + embedded web)
│   ├── flow_platform.py   #   Flowchart page (same level as the AI platform)
│   ├── workbench.py       #   Classic-operator workbench
│   ├── engine/            #   Pure-logic engine (data pipeline / model registry / metrics)
│   │   ├── models/        #     14 built-in models (with pruning / export support)
│   │   └── flow/          #     Flow engine (typed-port IR / executor / 57 operators)
│   └── assets/            #   Icons
├── labelme/               # vendored labelme 5.8.1 (GPLv3; modifications in MODIFICATIONS.md)
├── tests/                 # 39 domain test modules + tests/modules behavior locks
├── scripts/               # Per-platform launch scripts (repo-root auto-backtracking) + run_tools.py
├── tools/                 # Installer / weight downloader / env check / doc reconcile / test runner
├── docs/                  # Quick start / FAQ / user guide / images
├── licenses/              # License texts (AGPL-3.0 / GPLv3 / MIT / Apache-2.0 / BSD-3)
├── requirements.txt
├── run_main.bat           # Main workbench launcher (the only launcher at root)
└── install.bat            # One-click environment installer entry
```

Runtime data (projects, logs, preferences) is generated under `platform_data/` and is not
distributed with the repository; engine weights (including runtime-downloaded
family folders sam/ dino/ dinov2/) live in `xccv/engine/weights/`
(installed state: `<install root>/weights/`, gitignored).

## License

- This project's code is released under the **GNU Affero General Public License v3.0
  (AGPL-3.0)**, see [LICENSE](LICENSE).
- The `labelme/` directory is a vendored copy of labelme 5.8.1 with local modifications,
  under **GPLv3**; see
  [licenses/labelme-LICENSE-GPLv3.txt](licenses/labelme-LICENSE-GPLv3.txt).
- ultralytics (YOLO11 / SAM) used in the training pipeline is **AGPL-3.0**; for closed-source
  commercial use, please evaluate its enterprise licensing yourself.
- The CIF backend is adapted from [Sunny5250/CIF](https://github.com/Sunny5250/CIF) (MIT);
  the SubspaceAD backend is adapted from [CLendering/SubspaceAD](https://github.com/CLendering/SubspaceAD)
  (Apache-2.0); pruning depends on [Torch-Pruning](https://github.com/VainF/Torch-Pruning) (MIT).
  Modifications are licensed under AGPL-3.0; see `licenses/`.
- The Swagger UI static assets bundled with the deployment service are Apache-2.0 (Apache
  Software Foundation); TensorRT is an optional external runtime (Apache-2.0), not distributed
  with this repository.
- Third-party dependency list and notices: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md);
  paper citations: [CITATIONS.md](CITATIONS.md).

## Disclaimer

This project is provided "as is", without warranty of any kind, express or implied. Before
deploying models trained with this tool to production, perform your own thorough validation.
