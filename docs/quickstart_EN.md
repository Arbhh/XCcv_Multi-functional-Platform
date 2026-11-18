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

## 4. Flowchart (AI + common operators)

Click "Flowchart" in the workbench's left navigation (same level as the AI platform):
1. "Select Folder" picks the image directory (the image tree supports preview);
2. "Model Directory" chooses where flow JSONs live; "New" creates a blank flow, or pick an
   existing flow / template;
3. "Edit Flow" opens the canvas: drag **AI operators** (detection / instance segmentation /
   classification / per-image inference) and **common operators** (crop, grid tiling, result
   merge, detection filters, logic verdict, output) and connect them; model nodes need a `.pt`
   weight picked in the property panel;
4. "Run Flow" executes a single image and shows the result on the right; "Batch Run" executes
   the whole directory and exports results;
5. "Pack Flow" bundles the flow and its models into one `.xvap` file for reuse on other machines.

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
  `platform_data/ui_lang.json`; a few uncovered texts fall back to Chinese.

## License Note

This software is open-sourced under AGPL-3.0 (see README and THIRD_PARTY_NOTICES.md).
ultralytics, used in the training pipeline, is also AGPL-3.0 — make sure you meet its
compliance obligations when serving models over a network.
