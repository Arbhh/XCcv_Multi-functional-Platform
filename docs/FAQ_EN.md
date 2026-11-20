# XCcv Multi-functional Vision Platform · FAQ (English)

> Applies to: V0.8.0 (fully open source). Troubleshooting order: check this FAQ
> first; if unresolved, collect the last lines of `platform_data/logs/<platform>.log`
> and `crash.log` when reporting. For a quick start see
> [quickstart_EN.md](quickstart_EN.md); for terms and architecture see
> [XCcv多功能视觉平台_项目简介_EN.docx](XCcv多功能视觉平台_项目简介_EN.docx).
> Chinese version: [FAQ.md](FAQ.md).

**Contents**

1. [Installation and environment](#1-installation-and-environment)
2. [Weights and sample data](#2-weights-and-sample-data)
3. [Data annotation](#3-data-annotation)
4. [Model training](#4-model-training)
5. [Testing and project review](#5-testing-and-project-review)
6. [Model tools (pruning / TensorRT)](#6-model-tools-pruning--tensorrt)
7. [Deployment and inference service](#7-deployment-and-inference-service)
8. [Flow chart and classical operators](#8-flow-chart-and-classical-operators)
9. [Open source, licensing, and migration](#9-open-source-licensing-and-migration)

---

## 1. Installation and environment

**Q1.1  What are the requirements?**
Windows 10 / 11 (x64) and Python 3.11. GPU is optional: with an NVIDIA GPU, training and inference are significantly faster; CPU also works (prefer small models and small input sizes). Dependencies occupy roughly 6–8 GB.

**Q1.2  `pip install -r requirements.txt` reports a torch version conflict?**
Install torch separately first (choose the official index for your CUDA version), then the rest:

```bat
python -m venv venv && venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

On CPU-only machines replace `cu128` with the `cpu` index. ultralytics imposes an upper bound on torch; follow requirements.txt and do not bump major torch versions arbitrarily.

**Q1.3  Double-clicking `run_main.bat` does nothing / flashes and exits?**
The launcher probes interpreters in this order, testing whether `PyQt5.QtWebEngineWidgets` imports:
`XCCV_PY` → `python_path.txt` in the repo root → `python` on PATH → `py -3` → local `venv` / `.venv`.
Writing one line with the full python.exe path into `python_path.txt` (gitignored) is recommended. When nothing works the script calls `tools\env_check.py`, which prints why each candidate failed.

**Q1.4  Error `Could not find QtWebEngineProcess` / blank pages?**
PyQtWebEngine and PyQt5 versions must match (requirements pins 5.15.7 / 5.15.9). If other versions were installed manually, `pip uninstall PyQt5 PyQtWebEngine` first and reinstall per requirements. Corporate environments should also make sure security software is not blocking `QtWebEngineProcess.exe`.

**Q1.5  Are non-ASCII (Chinese) paths supported?**
Yes. Datasets, weights, and output directories may contain Chinese characters; the platform uses Unicode-safe I/O everywhere. A few third-party components (certain OpenCV builds) are sensitive to non-ASCII paths; the platform ships compatibility wrappers.

**Q1.6  Startup hangs on the splash card for over 30 seconds?**
Usually antivirus scanning QtWebEngine or extraction caches on first run. Whitelist the repository folder in your antivirus; if still slow, check in Task Manager whether the python process is alive and inspect `platform_data/logs/`.

**Q1.7  GPU present but training is slow / clocks drop?**
First confirm `torch.cuda.is_available()` is true and the device resolves to the GPU. Laptops with hybrid graphics may throttle; see the device and service-VRAM panels in Settings; disable system power saving if needed.

**Q1.8  Can I skip the virtual environment?**
Possible but not recommended. The `run_*.bat` launchers do not require a venv in the folder — `XCCV_PY` can point to any Python 3.11 environment. Mixing multiple torch versions is error-prone; prefer one environment per project.

---

## 2. Weights and sample data

**Q2.1  Why are there no `.pt` / `.pth` weights in the repository?**
This project **does not distribute any model weights**: weight licenses differ (Apache-2.0 / AGPL-3.0 / vendor-specific) and redistribution risks license contamination. The platform provides download/build scripts; users obtain weights under each publisher's terms.

**Q2.2  AI-assisted annotation says weights are missing?**
SAM-family weights are required (e.g. `sam2.1_t.pt`, `mobile_sam.pt`), placed in `weights/sam/`. When missing, the smart polygon, brush, and batch pre-annotation buttons show guidance; all other annotation features are unaffected.

**Q2.3  CIF / SubspaceAD reports a missing backbone?**
CIF needs the DINO ViT-B/8 weights (`weights/dino/dino_vitbase8_pretrain.pth`, or env `XCCV_DINO_CKPT`). SubspaceAD prefers a local `weights/dinov2/` directory (or `XCCV_DINOV2_CKPT`); when absent locally it may download standard DINOv2 weights via transformers from Hugging Face. On restricted networks set `HF_ENDPOINT=https://hf-mirror.com`.

**Q2.4  How to prepare baseline pretrained weights (YOLO COCO / ImageNet) in one shot?**

```bat
python xccv\engine\weights\download_weights.py
```

The script downloads YOLO and torchvision weights into `xccv/engine/weights/` (gitignored). For offline machines, run the same script on a connected machine and copy the whole folder.

**Q2.5  Where does the maskrcnn18 slim init weight come from?**

- Build it (recommended): `python tools\build_maskrcnn18_init.py` (six self-checks, produces `weights/maskrcnn18_slim_init.pt`);
- Convert from mmdet: `python tools\convert_mmdet_r18.py weights.pth --out weights/maskrcnn18_slim_coco.pt`.

Both outputs are plaintext `.pt`. The training panel's "slim head FPN64" + pretrained option injects them automatically; the self-built init takes precedence.

**Q2.6  Where are the defect-synthesis sample images?**
The sample backgrounds and defect cutouts (~33 MB) are distributed separately: download `XCcv-Platform-testdata-0.8.0.zip` and extract into `xccv/defectgen/` (yielding `testdata/`). Without it the synthesis UI still works — point it at your own background and cutout folders.

**Q2.7  Can weights be committed to git?**
Not recommended and not necessary. `.gitignore` excludes `weights/`, `xccv/engine/weights/`, and `*.pt/.pth/.onnx/.engine/.xva/.xcs`; for team sharing use object storage or release-page attachments.

---

## 3. Data annotation

**Q3.1  Which annotation forms are supported?**
Rectangles and polygons. For classification use rectangle annotations with `label` set to the class (or organize by folder); for OCR use rectangle annotations with `label` set to the text. Results are auto-saved as labelme-compatible `.json` files named after the image.

**Q3.2  What are "OK / good / normal / qualified / negative / background" folders for?**
Unannotated images under such folders are treated as pure background samples in training and validation (the training panel enables "negative-sample training" by default). The model learns that nothing should be detected there, directly reducing false positives. Aim for OK : defect ≈ 1:1–1:2.

**Q3.3  How do I use the three AI annotation modes?**

- `S` smart polygon: left-click the target, preview the contour live, right-click to undo an anchor, Enter to confirm;
- `B` brush: left stroke unions into the shape, right stroke subtracts;
- `L` single-image smart annotation: generates annotations for the current image; "Tools → AI batch pre-annotation" pre-generates for a whole folder.

**Q3.4  Is the annotation format labelme-compatible?**
Yes. `shapes/label/points/shape_type/imagePath` are standard fields; the platform adds a few additive fields (e.g. provenance markers) that do not affect opening files in labelme.

**Q3.5  How do I maintain the class list?**
First annotation prompts to create `labels.txt`; the annotation page's class merge/rename tools unify class names; export and training validate against both `labels.txt` and the JSONs to avoid class mismatch.

**Q3.6  How do I fix a wrong annotation quickly?**
Select a shape in the list to edit its vertices; `Ctrl+Z` undoes; "AI review this image (suspected misses)" lists detections without annotations for one-click confirmation and completion.

---

## 4. Model training

**Q4.1  `CUDA out of memory` — what now?**
From cheapest to most involved: ① reduce batch size (4→2→1); ② reduce training input size (augmentation "resize"); ③ enable "freeze pretrained backbone" and AMP; ④ for R-CNN enable the "lightweight mask head"; ⑤ the P2 small-object head costs about 2x memory — start from batch size 2.

**Q4.2  Loss flat / NaN?**
Check the **loss auto-diagnosis** on the training-completion page: underfit → more epochs; overfit → use best weights or fewer epochs; NaN → lower LR and inspect empty/corrupt annotations; enable "deterministic training" when you need reproducibility.

**Q4.3  Validation metrics look great, but the line leaks/over-flags?**
Five checks: ① are lighting/fixture consistent with training data; ② overfitting (see diagnosis); ③ same parameters locally and in deployment (use the deployment "consistency self-check"); ④ data imbalance (OK share); ⑤ **a threshold tuned on the test set is not a production guarantee** — re-validate on a fresh batch of line images.

**Q4.4  Two runs on the same project differ?**
Deep learning is nondeterministic by default (cuDNN autotuning, parallel sampling). With "deterministic training" enabled, same machine + seed + dependencies yields identical metrics (slightly slower), and `run_manifest.json` marks `deterministic=true`.

**Q4.5  Which model is fastest?**
Classification < detection < segmentation; within a task, n/s variants < R-CNN50. Speed first: `yolo11n` (detection), `yolo11n-seg` (segmentation), `resnet18` (classification); TensorRT FP16 export speeds things up further.

**Q4.6  How to choose among the three anomaly-detection backends?**

| Backend | Training | Use case |
|---|---|---|
| PatchCore | Unsupervised, OK images only | Zero-training baseline; first choice with sufficient data |
| CIF | Few-shot (k-shot) + DINO features + fuzzy hypergraph | Very few defect samples, complex textures |
| SubspaceAD | Few-shot + DINOv2 PCA subspace | Few-shot with limited compute |

All three hide epoch-like parameters automatically and show stage-based progress.

**Q4.7  How to configure tile-based training?**
Enable "tile-based training" in the training panel (R-CNN family) and set the tile side (multiples of 32 are best); background tiles keep 15%, 10% full images are mixed in, and validation always evaluates full images. **Inference must tile as well** (testing platform "4K tiled inference" or service `tiled=true`), otherwise the scales disagree.

**Q4.8  What does training produce?**
`best.pt` (plaintext weights) + `metrics.json` + `run_manifest.json` (environment fingerprint and configuration snapshot); with "save per-epoch weights" also `epoch_weights/epoch_NNN.pt` — a new run clears old epochs.

**Q4.9  Transfer-learning weights have a class mismatch?**
The platform validates the class table and rejects mismatches with a "regenerate" hint. For cross-class transfer, first generate pruned/init weights on a dataset with the same classes, or use "pre-training pruning" to rebuild for the current dataset.

---

## 5. Testing and project review

**Q5.1  How do I tune thresholds?**
Sliders on the Standard page act in real time; on the Project review page, dragging dimension thresholds (confidence / area / short side / long side / grayscale delta) only re-groups already-computed results — zero repeated inference.

**Q5.2  What do "evaluated / not evaluated" mean on the review page?**
Evaluated = images with GT or an OK mark (grouped into correct / miss / over-flag); not evaluated = unlabeled production images (grouped into OK / NG by whether detections pass the threshold). The two are decoupled; curves switch semantics with the current scope.

**Q5.3  Why aren't anomaly scores 0–1?**
Anomaly models use z scores (`1/conf−1`, unbounded); sliders and curves render with a dynamic axis bound. Classification uses probabilities, detection uses confidence — semantics are declared by the model and the UI adapts.

**Q5.4  Does A/B comparison re-run inference?**
No. Comparisons with identical weights and parameters are cached and reused; closing and reopening the window does not re-run.

**Q5.5  How to handle 4K images?**
Enable "4K tiled inference" on the testing platform: sliding-window tiling with overlap-aware merge; pair with "tile-based training" for consistent scales.

**Q5.6  What export formats are available?**
Unified JSON (`schema_version=1`), labelme, and COCO; batch runs also emit rendered images and summaries (per-class counts, timings, GT-matching statistics).

---

## 6. Model tools (pruning / TensorRT)

**Q6.1  Accuracy dropped after pruning — how to recover?**
Pruning cuts channels by L1 norm, aligned to multiples of 8, always skipping the classifier head, detection heads, RPN, and industrial modules. Prune 20–40% then fine-tune a few epochs on the original dataset; A/B the pruned model against the original directly on the testing platform.

**Q6.2  Why can't YOLO be pruned?**
Torch-Pruning's grouping on the ultralytics graph explodes memory exponentially, so the platform disables it deliberately with an explanation. Expert mode `XCCV_YOLO_PRUNE=1` force-enables it (run in a separate process; the memory watchdog kills runaway jobs).

**Q6.3  Can pruning/acceleration be stopped midway?**
Yes. "Emergency stop" terminates the subprocess immediately; atomic writes leave no half files, and source weights are untouched.

**Q6.4  TensorRT export says "TensorRT runtime not found"?**
`pip install tensorrt` matching your local CUDA major version; or set `TENSORRT_DIR` to a folder containing `nvinfer_10.dll`. Export requires an NVIDIA GPU.

**Q6.5  The engine fails to load on another machine?**
TensorRT engines are bound to the GPU compute capability and TensorRT version at build time. Across GPUs you get an explicit "please regenerate on this machine" message instead of a misleading corruption error. Re-export from the source weights on the target machine.

**Q6.6  Is the exported `.engine` plaintext?**
Yes. The container header (XCVE) records classes, input size, build compute capability, and TensorRT version, unencrypted; the loader pre-checks compatibility before deserialization.

---

## 7. Deployment and inference service

**Q7.1  The service won't start / port is taken?**
Check the tail of `platform_data/logs/service.log` in the deployment platform; change the port (editable in the panel) or stop the old service first. The service is single-worker (multiple workers would double-occupy VRAM).

**Q7.2  Where is the API key, and how do I rotate it?**
Generated on first start, stored in `platform_data/service/config.json`; the deployment panel can copy / regenerate it (the service restarts automatically after the port is released). Header: `X-XCcv-Key`.

**Q7.3  Can other machines access it?**
By default it listens on `127.0.0.1` only. Before switching to `0.0.0.0`: configure the model-path whitelist (enforced for non-loopback), safeguard the API key, and review network isolation policy.

**Q7.4  The C# test card fails to compile?**
The system .NET Framework `csc.exe` is required (present by default on Windows 10/11). You can also run `xccv\service\csharp\build_demo.bat` manually to see the compiler errors; the demo is WinForms and needs a desktop session.

**Q7.5  "Consistency self-check" reports a mismatch?**
The self-check runs one image through both local inference and the HTTP service, comparing boxes by IoU (≥0.98) and score deltas (<0.02). Mismatches usually mean different weights/parameters/preprocessing on the two sides: check loaded model paths, conf/iou, and whether tiling is enabled.

**Q7.6  Does the service ever return weights?**
Never. No endpoint returns weight bytes; models load only inside the service process, and endpoints return unified JSON or overlay images.

**Q7.7  How to register it as a Windows service?**
`xccv\service\install_service.bat` (needs NSSM and admin rights) provides a template: set `NSSM`, `SVC_EXE`, `SVC_ARGS`, `APPDIR`, then verify with `sc query XCcvInferenceService`.

---

## 8. Flow chart and classical operators

**Q8.1  How many operators are there?**
57 built-in (preprocessing 18, locating 7, measurement 2, logic 5, detection filter 3, crop/per-image 6, AI inference 5, output 4, external 1), plus 9 legacy-chain delegate nodes — 66 on the canvas; 9 official templates. `python tools\doc_reconcile.py` audits these numbers any time.

**Q8.2  Workbench ("classical platform") vs flow chart — what's the difference?**
The workbench is for **single-operator trials and chain validation** (quick visual checks and parameter tuning); the flow chart is for **production-grade orchestration** (multi-branch, per-image execution, result merging, packaging). Both share the same operators and port contracts.

**Q8.3  Are custom operators safe?**
Custom operators run your script locally via `exec` with an AST whitelist (only numpy/cv2/math/statistics imports, no double-underscore attributes, restricted builtins) — **a controlled environment against mistakes, not a sandbox against malicious code**. Do not run flow JSON files or operator packages from unknown sources.

**Q8.4  Do external operators (HALCON / VisionMaster) still work?**
V0.8.0 turns external operators into **plugin backends**: the repository ships an OpenCV demo backend (common operator names run out of the box); real vendor SDKs are implemented by users against the `ExternalEngine` interface and registered — nodes and flows unchanged. No vendor adapters or SDK files are distributed.

**Q8.5  How do I use `.xvap`?**
"Pack flow" seals the flow JSON and the selected models into a single `.xvap` (plaintext container); "unpack flow" on another machine restores the flow and a `flow models/` folder, rewriting model references to local paths; the inference service can also load `.xvap` as a resident flow.

**Q8.6  A node fails when the flow runs?**
Node errors are isolated: the node turns red, downstream is marked skipped, and independent branches continue. Check the node-level error in the run log (bottom-right of the workbench/flow page) and trial-run nodes one by one.

---

## 9. Open source, licensing, and migration

**Q9.1  What is the license? Can I use it commercially?**
This project is AGPL-3.0; `labelme/` is a GPLv3 vendored copy; the training stack depends on ultralytics (AGPL-3.0). AGPL allows commercial use, but **offering it over a network triggers the source-disclosure obligation**; for closed-source integration, evaluate ultralytics' enterprise licensing and replacing PyQt5 with an LGPL stack.

**Q9.2  Where are third-party components and paper citations recorded?**
`THIRD_PARTY_NOTICES.md` (components, licenses, usage) and `CITATIONS.md` (papers); full license texts in `licenses/`.

**Q9.3  Can I submit PRs?**
Yes, welcome. Contributions are accepted under DCO 1.1 (`git commit -s`); the PR template requires declaring third-party code provenance. See `CONTRIBUTING.md` for the third-party admission matrix and contamination protocol.

**Q9.4  How do I report a security issue?**
Do not open a public issue. Use the private channel in `SECURITY.md` (GitHub Security Advisories or email) with reproduction steps and environment details.

**Q9.5  What should I watch out for when upgrading from 0.7.x?**

- Weight containers `.xva/.xcs` are no longer supported: retrain or re-export with 0.8.0; loading old files yields a clear migration hint;
- Training artifacts, epoch weights, flow packages, and engines are all plaintext (`best.pt`, `epoch_NNN.pt`, `.xvap`, `.engine`);
- Model tools no longer have "weight encrypt / decrypt" cards;
- Old flow JSON files still load (the port migration table remains compatible); external-operator nodes must switch to the plugin backend.

**Q9.6  Where is the documentation, and how does it stay current?**
`README*` → `docs/quickstart*` → this FAQ → `CHANGELOG*`. Operator/model counts and supported versions are audited by `python tools\doc_reconcile.py` — any drift fails the build; UI English strings are guarded by a coverage check (currently 673 sites / 1719 entries, 100% coverage).
