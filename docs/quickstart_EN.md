# Quick Start

This guide walks you through your first detection project end to end: **Annotate → Train → Test**.

(Chinese version: [quickstart.md](quickstart.md))

## 0. Prepare Data

Put the images to annotate into one folder (jpg / jpeg / png / bmp / webp supported,
sub-folders included).

## 1. Annotate

1. Open the workbench (`run_main.bat`) and click the "Annotation Platform" card;
2. Menu "File → Open Directory" and pick the image folder;
3. Press `Q` to create a polygon / `W` to create a rectangle (the right-side buttons work too);
   the first label prompts to create the class file (labels.txt);
4. AI assist (optional; place SAM weights under `weights/sam/` as described in the README):
   - Press `S`: smart polygon — left-click the target to preview the outline live,
     right-click to undo anchors, Enter to confirm;
   - Press `B`: brush mode — left strokes add (union), right strokes erase (difference);
   - Press `L`: smart-annotate the current image; "Tools → AI Batch Pre-annotation" handles a
     whole folder;
5. Labels are saved automatically as same-name `.json` files (labelme format).

## 2. Train

1. Back in the workbench, click the "Training Platform" card → "New Project":
   - fill the project name and choose the dataset folder (from step 1) and the model output folder;
   - choose the task type: classification / detection / instance segmentation /
     unsupervised anomaly detection / OCR;
2. Enter the project → pick a model (`yolo11n` recommended for detection,
   `yolo11n-seg` or `maskrcnn18` for segmentation, `resnet18` for classification,
   `patchcore` (zero-training baseline) / `cif` (few-shot hypergraph) /
   `subspacead` (few-shot PCA subspace) for anomaly detection, `crnn` for OCR);
3. "Quick Config" recommends parameters automatically; switch to "Expert Mode" to fine-tune
   (augmentation, industrial optimizations, accelerations, tiling, ...). The project info page
   offers the data grid (filter by label / OK negatives / unlabeled), the split donut chart that
   redraws live with the train/val ratios, and the real-time augmentation preview;
4. Click "Start Training" and watch loss / mAP curves, the per-class metrics table and the loss
   diagnosis on the training page; `best.pt` is produced in the project's model folder.

> Too few defect samples? Use the "Defect Sample Synthesis" card to blend defect patches onto
> good-part images first, then come back to training.

## 3. Test

1. Click the "Testing Platform" card in the workbench;
2. Pick the project on top → "Load latest model of project" (or "Select Weights…" to pick a
   `.pt` directly);
3. "Open Image… / Open Folder…" to choose images; A / D switches previous / next;
4. Tune the confidence / IOU sliders (defaults 0.25 / 0.20) and see results live;
   "Batch Run…" infers a whole folder and exports:
   - `检出结果/`: one JSON per image (unified contract / labelme / COCO selectable);
   - `可视化/`: rendered images;
   - `摘要.json`: per-class counts and latency (incl. GT matching stats for the threshold panel).

For very large images (e.g. 4K), tick "4K tiled inference" to improve small-object detection.

### Project Review (threshold tuning)

Switch to the "Project Review" tab at the bottom: after picking the project and a training run,
the platform batch-infers the training dataset once; afterwards every threshold change is free
of re-inference.

The review page decouples **Evaluated / Unevaluated** (toggle on top):

- **Evaluated**: results group into Correct / Missed / Over-kill thumbnails; drag the threshold
  line or type a value in the dimension-threshold panel (confidence / area / short side /
  long side / gray difference) — the sweep curves show the image count of each group at that
  threshold (same semantics as the groups) and the groups re-split instantly;
- **Unevaluated**: unlabeled production images group into OK / NG (NG = at least one detection
  above threshold); the curves and the number row switch semantics accordingly;
- review verdicts can be reflowed back to the dataset's "to-annotate" folder; "Model A/B
  Compare" compares two weights image by image (results for the same weights and parameters are
  cached — reopening the dialog does not re-run inference) and shows an overall verdict bar.

## 4. Flowchart (AI + classic operators)

Click "Flowchart" in the workbench's left navigation (same level as the AI platform):
1. "Select Folder" picks the image directory (the image tree supports preview);
2. "Model Directory" chooses where flow JSONs live; "New" creates a blank flow, or pick an
   existing flow / template (9 built-in templates: AI filter, cascade crop-merge,
   template-match prefilter, circle locate, etc.);
3. "Edit Flow" opens the canvas: drag **AI operators** (detection / instance segmentation /
   classification / per-image inference) and **classic operators** (crop, grid tiling, result
   merge, detection filters, blob analysis / template matching / find line & circle /
   measurement, logic verdict, external-operator plugin backend, output) and connect them;
   model nodes need a `.pt` weight picked in the property panel;
4. "Run Flow" executes a single image and shows the result on the right; "Batch Run" executes
   the whole directory and exports results;
5. "Pack Flow" bundles the flow and its models into one `.xvap` file for reuse on other machines.

## 5. Deployment platform and inference service

Click "Deployment" in the workbench's left navigation — three cards:

1. **Inference service**: "Start Service" launches the HTTP inference service
   (default `127.0.0.1:8600`; an API key is generated on first start and can be copied /
   reset in the panel). Endpoints:
   - `POST /v1/models/load`: load a `.pt` / `.engine` model as a resident API;
   - `POST /v1/infer/image`, `POST /v1/infer/batch`, `POST /v1/render/overlay`:
     single-frame / batch inference and overlay rendering;
   - `POST /v1/flows/load` + `POST /v1/flows/run`: resident flow packages;
   - `WS /v1/ws/infer`: WebSocket binary-frame streaming inference;
   - `GET /docs`: self-hosted offline Swagger documentation;
2. **Model Loading**: load models / flow packages as resident APIs from a window — no code
   needed (status table refreshes every 2 s);
3. **C# Test**: one-click build & run of the WinForms demo (same as
   `xccv\service\csharp\build_demo.bat`) — load a model → infer → inspect the unified JSON
   and overlay image. The deployment platform also offers a "Consistency Check"
   (the same image through local inference and the HTTP service, compared box by box).

For headless deployment use `run_service.bat` (or `python -m xccv.service.main --port 8600`).

## 6. Model tools

The workbench "Model Tools" card (or `run_modeltools.bat`):

1. **Model pruning**:
   - Pre-training pruning: builds a pruned model from the project dataset (saved under the
     project's model folder `pruned/`; it appears in the training platform's model dropdown —
     pick it to start transfer learning);
   - Post-training pruning: processes a training run's `best.pt` and writes `best_pruned.pt`
     next to it, ready to compare head-to-head in the testing platform;
2. **Model acceleration (TensorRT)**: export `best.pt` to a TensorRT engine (`.engine`,
   FP16/FP32; needs GPU + TensorRT; YOLO via the native ultralytics path, classification via
   in-memory ONNX→TRT, Mask R-CNN feature-segment acceleration). The testing platform loads
   `.engine` directly; engines are bound to the building GPU model — cross-GPU loading gives
   a clear message.

## FAQ

- **First training is slow / stuck downloading**: pretrained weights (YOLO / torchvision) are
  downloaded from the official source on first use. For offline machines, download them on a
  connected machine beforehand and place them under `weights/` (see the comments in
  `xccv/engine/offline.py`);
- **AI annotation says weights missing**: place the SAM weights under `weights/sam/` as
  described in the README;
- **CIF / SubspaceAD reports missing backbone weights**: CIF needs the DINO ViT-B/8 weights at
  `weights/dino/dino_vitbase8_pretrain.pth` (or `XCCV_DINO_CKPT`); SubspaceAD prefers a local
  DINOv2 model directory under `weights/dinov2/` (or `XCCV_DINOV2_CKPT`) and may let
  transformers download the standard DINOv2 weights from Hugging Face when no local copy is
  present (use `HF_ENDPOINT` for a mirror). Weights are not distributed with this repository;
  use them under their publishers' terms;
- **Class mismatch errors**: transfer-learning weights must match the current dataset's classes;
- **Out of GPU memory**: reduce the batch size, or enable "Freeze backbone",
  "Lightweight mask head" and other acceleration options;
- **Switching the UI language**: workbench "Settings" → "Language" → English; all windows
  switch automatically after saving (no restart needed). The preference is stored in
  `platform_data/ui_lang.json`; a few uncovered texts fall back to Chinese;
- **Are model weights plaintext?** Yes. Since 0.8.0 all training / pruning / acceleration
  artifacts are plaintext (`best.pt`, `best_pruned.pt`, `.engine`); the legacy encrypted
  containers (`.xva` / `.xcs`) are no longer supported — retrain or re-export with the
  current version;
- **Which TensorRT to install**: `pip install tensorrt` matching your CUDA major version,
  or set `TENSORRT_DIR` to a folder containing `nvinfer_10.dll`. Engines are bound to the
  building GPU's compute capability (an engine built on an RTX 30 card will not load on a
  20-series card);
- **Where is the inference service API key**: `platform_data/service/config.json`
  (generated on first start; the deployment panel can reset it in one click);
- **Why are there no weights or sample images in the repo**: this project ships no model
  weights and no sample data. Weights follow their publishers' licenses and are obtained by
  the user (placement table in the README "Optional: weights and sample data"); defect-synthesis
  sample images live in the separate `XCcv-Platform-testdata-0.8.0.zip` package;
- **Where do the sample images go**: extract `XCcv-Platform-testdata-0.8.0.zip` under
  `xccv/defectgen/` (yielding `xccv/defectgen/testdata/`). Without it the synthesis UI still
  runs — just point it at your own background/patch folders.

## License Note

This software is open-sourced under AGPL-3.0 (see README and THIRD_PARTY_NOTICES.md).
ultralytics, used in the training pipeline, is also AGPL-3.0 — make sure you meet its
compliance obligations when serving models over a network.
