# Changelog

Notable changes to the XCcv Multi-functional Platform. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

Chinese changelog: [CHANGELOG.md](CHANGELOG.md)

## [0.8.0] - 2026-11-20

> **Major update**: the project is renamed to **XCcv Multi-functional Platform**
> and becomes **fully open source** from this release — the four previously
> closed capabilities (deployment inference service, model tools, TensorRT/ONNX
> export, classic vision operators) are now published with the source; the
> weight encryption/decryption system is **removed** (training artifacts are now
> plaintext `.pt` / `.engine`), and the external operator is redesigned as a
> pluggable-backend architecture.

### Added (full open source)

- **Deployment platform & HTTP inference service** (`xccv/deploy_platform.py` +
  `xccv/service/`): a FastAPI service loads models/flows as resident APIs
  (LRU pool + per-model lock), with single / batch / overlay / WebSocket
  binary-frame inference and `X-XCcv-Timing` per-stage timing; constant-time
  API-key comparison, concurrency gating, model path whitelist, self-hosted
  offline Swagger docs. The deployment platform provides the service console
  (host/port, API-key reset, desktop island, GPU insight), a model-loading
  window, a C# test card and a one-click local-vs-service consistency check.
- **C# SDK and demo** (`xccv/service/csharp/`): single-file `XCcv.Vision.Sdk.cs`
  thin client (.NET Framework 4.5+, compiled by the system csc, zero third-party
  dependencies) plus a bilingual WinForms test program (drag-and-drop inference,
  overlay preview).
- **Model-tools platform** (`xccv/modeltools/modeltools_window.py`): structured
  channel **pruning** (Torch-Pruning DepGraph; pre-training and post-training
  modes, isolated subprocess with a RAM watchdog and emergency stop); **TensorRT
  engine export** (YOLO via the native ultralytics path; classification via
  in-memory ONNX→TRT; Mask R-CNN feature-segment acceleration, with a metadata
  header recording classes/input size/build compute capability and a cross-GPU
  precheck on load); the matching `.engine` production runtime
  (`xccv/engine/trt_runtime.py`).
- **Classic vision operators** (`xccv/engine/flow/ops/`): 17 preprocessing
  (filtering/threshold/morphology/frequency/flat-field etc.), 7 locating
  (blob analysis, template matching, find line/circle/corner, line pair,
  rectangle fit), 2 measuring, 5 logic verdicts, 3 classic detection filters,
  4 output, 6 crop — built-in operators grow from 39 to **57** (66 nodes), with
  5 classic-vision flow templates (template-match prefilter, circle locate,
  line-pair level, blob prefilter, crop export).
- **External-operator plugin backend**: the `external_op` node is redesigned as
  a pluggable-backend architecture (`ExternalEngine` protocol + registry); the
  bundled OpenCV demo backend (mock) lets the node run end-to-end out of the
  box; real vendor SDK backends plug in via the same interface with zero node
  changes.
- **Platform experience**: workbench card wall and page stack, deployment /
  model-tools navigation, service-health capsule and desktop island, GPU
  monitor, classic-operator workbench, operator editor, training page split
  (`training/panels/`), double-click detection-result rendering, review-page
  "Open Flowchart" jump.
- **Test matrix**: five module behavior locks under `tests/modules/` and a
  training-page snapshot lock; `tools/run_tests.py` per-module subprocess test
  runner (a stable way to run the large suite on Windows).

### Changed

- **Renamed**: repository/docs renamed to XCcv Multi-functional Platform; the
  `-community` version suffix is dropped; the first fully-open-source release
  is 0.8.0.
- **Removed the weight encryption system**: training / pruning / per-epoch
  weights / flow packages / TensorRT exports are all plaintext (`best.pt`,
  `best_pruned.pt`, `epoch_NNN.pt`, plaintext `.xvap` container, plaintext
  `.engine`); loading a legacy `.xva/.xcs` container raises a clear migration
  message.
- The model-loading entry (registry) is refactored: routing by ckpt format /
  `.engine` metadata; TensorRT engines now use the `.engine` suffix.
- `requirements.txt` adds `torch-pruning`, `onnx`, `psutil`, `python-multipart`; `onnxruntime` is
  kept for EfficientSAM-assisted annotation.
- Testing-platform texts/filters referencing `.xva` now use the plaintext
  `.pt` / `.engine` wording.

### Removed

- The weight encrypt/decrypt tool window (the "Weight Encrypt / Decrypt" card in
  Model Tools) and the `engine/model_crypto.py` container implementation.
- The HALCON / VisionMaster vendor bridge adapters (in-process invocations and
  the HALCON operator migration mapper); the capability now lives behind the
  external-operator plugin-backend interface and vendor adapters are no longer
  distributed with the repository.

### Tests

- Full regression: unit tests (historical locks + module locks + page snapshot
  locks) and per-platform offscreen self-tests; results are in the release
  acceptance notes. Use `python tools/run_tests.py` to run per module.

### Third-party license changes

- New dependencies: Torch-Pruning (MIT), onnx (MIT), psutil (BSD-3-Clause),
  swagger-ui static assets (Apache-2.0, self-hosted at the service /docs).
- TensorRT is an optional external runtime (Apache-2.0), not distributed here.
- Attribution and full texts: `THIRD_PARTY_NOTICES.md` / `licenses/`.

## [0.7.3-community] - 2026-11-18

> Feature release: two few-shot anomaly-detection backends and a training-free
> paradigm UI, unified score semantics and review grouping, broader language
> rebuild coverage, and English UI for the defect synthesis tool.

### Added

- **Training: CIF few-shot anomaly backend** (DINO ViT-B/8 + fuzzy hypergraph,
  RGB branch; coreset memory, Bi-TF-MP propagation, hyperedge retrieval, z
  calibration; backbone weights are user-supplied).
- **Training: SubspaceAD few-shot anomaly backend** (DINOv2 features + PCA
  subspace; k-shot support 1–12, augmentation count configurable; reconstruction
  residual scoring and z calibration).
- **Training-free paradigm UI**: epoch/batch/learning-rate parameters hidden;
  progress shown by stage (features & memory / calibration / validation); loss
  area states that no loss curve exists; AUROC curve supported.
- **Score-semantics metadata**: models declare `conf` / `prob` / `z`; testing
  sliders, review threshold panel and unified JSON follow the declared semantics
  (z uses an unbounded dynamic axis).
- **Weak-GT review**: ANOMALY uses source folders as image-level GT (OK with
  detection = over-kill, NG with none = missed, NG with detection = correct);
  classification groups correct/wrong by folder label.
- **UI language**: training project windows follow host rebuilds; a loaded model
  prompts a window reopen; the defect synthesis tool is localized (DOM dictionary
  + canvas labels).
- **Compliance**: source attribution and license texts for CIF (MIT) and
  SubspaceAD (Apache-2.0), notice entries, `CITATIONS.md`, DCO and PR template.

### Changed

- Built-in models: 12 → **14** (CIF, SubspaceAD).
- `requirements.txt` adds pinned `timm`, `scikit-learn`, `anomalib`, `transformers`.
- README main-UI screenshot and docx Figure 1 updated to the 0.7.3 render.
- Training outputs remain plaintext `best.pt`; backbone weights are not
  distributed with the repository or source package.

### Fixed

- Platform windows did not rebuild after another process changed the language
  preference file (stale in-process cache).
- ANOMALY review counted OK-image false detections as TP, inflating precision
  and recall.
- Switching language with a loaded model lost the loaded weights; the UI now asks
  the user to reopen the window.

### Tests

- `python -m unittest discover -s tests -p "test_*.py"`: **239 tests passed**
  (194 from 0.7.2 + 45 CIF/review/semantics/language regressions; 5 CIF
  integration tests skipped by design when DINO weights are absent).
- `compileall` and the four-platform offscreen self-tests: passed.

### Third-party license changes

- Added CIF (MIT), SubspaceAD (Apache-2.0), timm (Apache-2.0), scikit-learn
  (BSD-3-Clause), anomalib (Apache-2.0) and transformers (Apache-2.0); all are
  AGPL-3.0-compatible. See `THIRD_PARTY_NOTICES.md` and `licenses/`.
- Standard DINO / DINOv2 weights are Apache-2.0; the repository does not
  distribute weights — users provide them.

## [0.7.2-community] - 2026-11-16

> Feature release synchronizing V0.7.2: four project-review fixes in the
> testing platform plus platform-wide **one-click Chinese / English switching**.

### Added

- **UI language: one-click Chinese / English switching**: choose the language in the workbench
  "Settings" dialog and every platform window follows after saving (no restart required —
  a preference-file watcher rebuilds the workbench immediately and each platform window
  automatically). Resident UI texts, message boxes / file dialogs / status bar / toasts,
  matplotlib and pyqtgraph chart texts, and the flowchart web canvas (node library / property
  panel / in-page texts) are translated; uncovered texts fall back to Chinese. The preference
  lives in `platform_data/ui_lang.json`.
- **Testing platform: Evaluated / Unevaluated decoupling**: the project review page gains a
  scope toggle — "Evaluated" shows the Correct / Missed / Over-kill groups, "Unevaluated" shows
  unlabeled production images as OK / NG (NG = at least one detection above threshold); the
  statistics row switches semantics with the scope.
- **Testing platform: A/B comparison cache and overall bar**: comparison results are cached by
  (model A, parameters, B weights, batch fingerprint), so accidentally closing the dialog and
  reopening reuses the result instead of re-running inference; the verdict statistics gain a
  four-segment stacked overall bar (A better / B better / Equal / Unevaluated).

### Changed

- **Testing platform: threshold sweep curves redesigned**: the curves now read "the **image
  count** of each group at threshold = x" — exactly the same semantics as the groups and the
  statistics row (always Correct + Over-kill + Missed = evaluated images; the score dimension
  also gets a missed curve). Switching Evaluated / Unevaluated re-bases the curves and the
  number row, while the axis still follows the dataset only (no axis re-layout on switching,
  so the threshold line is never pushed out of view).
- Per-detection records of unevaluated production images now enter the batch statistics
  (the data source for the OK / NG sweep and grouping, with zero re-inference).
- UI text trims: verbose hints on the flowchart page and the review page were removed
  (details live in the user documentation).

### Fixed

- Bottom-axis tick labels of the dimension threshold panel were squeezed together (root cause:
  a width-clamping API was misapplied to the bottom axis, compressing ticks into a few tens of
  pixels on the left); now protected by a rendering-level regression test.
- The missed curve on the project review page displayed incorrectly or not at all (the old
  curve used a detection-instance semantics and could only be drawn for geometry dimensions;
  resolved by the sweep-curve redesign).
- The review-and-reflow and model-A/B dialogs crashed on open (a translation helper was used
  without importing it, affecting both Chinese and English modes); a repository-wide static
  scan now guards against recurrence.

### Tests

- `python -m unittest discover -s tests -p "test_*.py"`: **194 tests passed** (the 170 from
  0.7.1 + 20 new 0.7.2 regression tests + 4 existing tests updated to the new semantics;
  covering threshold-panel rendering, sweep-curve numerics, scope-switch wiring, A/B cache
  reuse, the bilingual replacement layer / tree localization / language persistence /
  real-window integration, and the missing-import static scan).
- Offscreen self-tests of four platforms (workbench / training / testing / flowchart): passed.
- English-mode UI verified: the workbench and the testing platform render fully in English.

## [0.7.1-community] - 2026-11-14

### Added

- **Testing platform: dimension threshold panel rewritten**: fixed axes (confidence always
  0–1; area / short side / long side / gray difference expand in rounded steps from the current
  batch); smoothed scatter curves (normal / over-kill / missed); thresholds accept both line
  dragging and numeric typing; the foreground-ratio dimension was removed from the panel
  (the flow node keeps it).
- **Testing platform: missed-detection dimension data**: the geometry (area / short side /
  long side) of missed GT boxes now feeds the threshold statistics, so the missed curve can be
  drawn.
- **Training platform: real-time preview of all augmentation operators**: ticking, probability
  or parameter changes re-render the preview, and the preview notes which operators took effect.
- **Training platform: split donut chart follows parameters**: redrawn live from the train / val
  ratios, with counts computed by the same rule as the split implementation.
- **Engineering: data contract tests** `tests/test_data_contracts.py` (labelme → internal
  annotation → YOLO row round-trip, flowchart port schema rejection semantics);
  `pyproject.toml` project metadata; CI unit-test coverage extended with the tests.

### Changed

- IOU default is 0.2 platform-wide (testing sliders, project review page, auto annotation,
  threshold flow template and flow node defaults).
- UI text trims: parenthetical explanations moved into tooltips and the user manual.
- Project review page's left column is 520px; the per-class metrics table no longer stretches
  its last column and scrolls horizontally per pixel.

### Fixed

- The waiting banner did not appear when reopening the training platform and clicking a project
  card (forced synchronous repaint + minimum visible duration).
- Thumbnails went blank after filtering or switching labels in the data grid (cached icons are
  re-applied).
- The AP@50 column of the per-class metrics table stretched abnormally after restoring a
  maximized window, and the horizontal scrollbar was hard to drag.
- The same blank-thumbnail rendering issue in the project review page's data grid.

### Tests

- `python -m unittest discover -s tests -p "test_*.py"`: **170 tests passed** (test_v07_* 126 +
  engine purity 2 + test_v071_fixes 30 + data contracts 12; two legacy-expectation cases were
  updated with 0.7.1).
- `python -m compileall -q xccv labelme tests tools`: passed.

## [0.7.0-community] - 2026-11-13

> Feature release synchronizing V0.7.0. The community edition covers annotation / training /
> testing / flowchart (AI + common operators) / defect sample synthesis.

### Added

- **Training platform: unsupervised anomaly detection (PatchCore-lite)**: new TaskType.ANOMALY
  and the PatchCore model; builds a feature memory bank from OK/good images and outputs an
  anomaly heatmap, connected-component boxes and an image-level score; supports image-level
  AUROC/AP and the OK false-alarm rate; `best.pt` loads directly in the testing platform.
- **Training platform: OCR (CRNN + CTC)**: new TaskType.OCR and the CRNN model; labelme
  rectangle labels carry the text content; training produces the charset and weights, the
  testing result list shows the recognized text, and unified JSON detections gain a `text` key.
- **Training platform: per-class metrics table**: class, GT, detections, TP/FP/FN, precision,
  recall and AP@50 on the training page for completed and historical runs.
- **Training platform: data grid**: a thumbnail wall with background scanning, lazy loading and
  an LRU cache; filter by label / OK / unlabeled / ignore regions.
- **Training platform: split donut chart, loss diagnosis, post-resize small-instance check**.
- **Training platform: grayscale training, deterministic runs, P2 small-object head,
  negative-sample training**.
- **Training platform: run_manifest.json**: environment fingerprint, data fingerprint, full
  parameter snapshot and product paths.
- **Testing platform: Standard / Project Review dual pages**: the review page batch-infers the
  training set for a project + run and shows Correct / Missed / Over-kill / Unevaluated
  thumbnails; after one inference, threshold changes only re-group with zero re-inference.
- **Testing platform: six-dimension threshold panel**: confidence / area / short side / long
  side / foreground ratio / gray difference, stacked histogram with draggable lines and live
  P/R/OK false alarms; one-click generation of a threshold flow template.
- **Testing platform: review and one-click reflow**: four persistent states
  (normal / missed / over-kill / doubtful); selected results reflow into the dataset's
  "to-annotate" folder for continued labeling.
- **Testing platform: model A/B comparison**: the second model re-runs in the background and
  each image is marked A better / B better / Equal / Unevaluated.
- **Testing platform: inference statistics**: first-image warm-up, min / p50 / avg / max
  latency, VRAM watermark and a suspected GPU-throttling hint.
- **Testing platform: OK image presentation and statistics**: OK negative-sample ratio,
  false detections per image during training, and the annotation platform's [OK] badge and
  "mark this folder as OK/good-part".
- **Annotation platform: quality tips and AI review of suspected missed labels**: the help menu
  gains quality tips and a detection-model review that lets you accept each AI-detected but
  unlabeled region as a rectangle.
- **Annotation platform: AI waiting banner**: a visible top banner while a model loads or
  infers; failures unlock the UI for retry.
- **Flowchart: 3 detection-filter operators**: short/long side, foreground ratio and gray
  difference filters, bringing the community operator count from 36 to 39; plus a new
  "detection → crop → classification" cascade template (3 → 4 templates).
- **Flowchart: image-pick fallback before running**: if no image is selected, a picker opens
  instead of loading the model first and only then discovering the missing selection.
- **Engineering: 0.7 regression tests shipped with the package**: 128 tests under `tests/`,
  reproducible with a single command.

### Changed

- Version bumped to `0.7.0-community`.
- Data preparation supports independent negative-sample splits and approximate filtering of
  `__ignore__` regions; YOLO training/validation uses letterbox geometry consistently.
- Project configuration gains `use_negatives` / `p2_head` / `color_mode` / `deterministic`
  fields; old projects pick up the defaults automatically.
- **Docs**: README feature table, operator count and quick start updated for 0.7.0.

### Fixed

- Segmentation validation on OK negative images lost a whole batch to an empty-tensor index
  error (now isolated per image and coordinates restored by the validator).
- The annotation platform's `_ai_loading` flag stuck after an AI model load failure, making
  later clicks ineffective.
- Threshold-line out-of-range dragging, leftover review cache and crowded Standard-page entries
  found during 0.7 testing.

### Tests

- `python -m unittest discover -s tests -p "test_*.py"`: 128 tests passed (30.3s).
- `python -m compileall -q xccv labelme`: passed.
- Offscreen UI smoke tests: training platform (5 task types), testing platform (dual pages /
  review page), flowchart page, threshold panel and data grid all instantiate.
- UI acceptance: workbench → training task dropdown (anomaly / OCR) → testing
  Standard / Project Review switch → annotation [OK] / help menu → flowchart editing, all
  passed.

## [0.6.2-community] - 2026-11-12

### Fixed

- `run_*.bat` launcher scripts are now **pure ASCII**, eliminating mojibake in start-up errors
  caused by Windows code-page differences; the Chinese diagnosis moved to `tools/env_check.py`.

### Added

- Interpreter hint file `python_path.txt` (one full path per line; gitignored, may differ per
  machine); the launcher candidate order was extended and each candidate is probed
  (`XCCV_PY` → `python_path.txt` → PATH `python` → `py -3` → `C:\XCcv\venv` → local venv/.venv).
- Environment diagnosis script `tools/env_check.py`: prints the usability and failure reason of
  each candidate interpreter.
- Appendix `docs/XCcv社区版_项目简介.docx` and workbench screenshot `docs/images/main.png`.

## [0.6.1-community] - 2026-11-11

First open-source release (published after a pre-release review).

### Added

- Annotation platform: integrated labelme 5.8.1 (vendored, with local modifications, see
  `labelme/MODIFICATIONS.md`), rectangle/polygon annotation, autosave, class management;
  AI-assisted annotation (S smart polygon / B brush / L single-image smart annotation /
  folder batch pre-annotation, based on SAM / EfficientSAM; weights not distributed).
- Training platform: project-based management; 10 built-in models for classification /
  detection / instance segmentation; quick config and expert mode; industrial optimization
  modules (attention / DCNv2 / Focal-family losses); accelerations (AMP / TF32 / early stop);
  tiling training; live curves and data health check.
- Testing platform: single-image / batch inference, live threshold tuning, tiled inference for
  large images, result export (unified JSON / labelme / COCO).
- Flowchart module: 36 AI and common operators, web canvas orchestration (LiteGraph.js),
  `.xvap` packing, 3 templates.
- Defect sample synthesis: Poisson-fusion augmentation of defect samples (local FastAPI +
  embedded web, loopback only).
