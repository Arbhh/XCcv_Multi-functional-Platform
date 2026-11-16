# Changelog

Notable changes to XCcv Community. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

Chinese changelog: [CHANGELOG.md](CHANGELOG.md)

## [0.7.2-community] - 2026-11-16

### Added

- **UI language: one-click Chinese / English switching**: switch in the workbench "Settings"
  dialog and every platform window follows, no restart required. Resident UI texts, dialogs,
  status bar, chart texts and the flowchart web canvas are translated; uncovered texts fall
  back to Chinese.
- **Testing platform: Evaluated / Unevaluated decoupling**: the project review page gains a
  scope toggle — "Evaluated" shows Correct / Missed / Over-kill, "Unevaluated" shows unlabeled
  production images grouped as OK / NG; the statistics row follows the selected scope.
- **Testing platform: A/B comparison cache and overall bar**: comparison results are cached by
  (model A, parameters, B weights, batch fingerprint) and reused when the dialog is reopened;
  verdict statistics include a four-segment bar (A better / B better / Equal / Unevaluated).

> Language coverage: some settings dialogs and the defect synthesis tool are not yet localized
> and currently fall back to Chinese; English coverage will be extended in later releases.

### Changed

- Threshold sweep curves now read "image count of each group at threshold = x", matching the
  groups and statistics row; switching scope re-bases the curves and number row, while axes
  still follow the dataset only.
- Per-detection records of unevaluated production images now feed the batch statistics for
  OK / NG sweeping and grouping, with zero re-inference.
- UI text trims: verbose hints on the flowchart and review pages were removed (details live in
  the user documentation).

### Fixed

- Bottom-axis tick labels of the threshold panel were squeezed together (misapplied
  width-clamping API); now protected by a rendering-level regression test.
- The missed curve on the project review page displayed incorrectly or not at all; resolved by
  the sweep-curve redesign.
- The review-and-reflow and model-A/B dialogs crashed on open (missing translation import);
  a repository-wide static scan now guards against recurrence.

### Tests

- `python -m unittest discover -s tests -p "test_*.py"`: 194 tests passed; offscreen self-tests
  of four platforms passed; English-mode UI verification passed.

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
  found during 0.7 real-machine testing.

### Tests

- `python -m unittest discover -s tests -p "test_*.py"`: 128 tests passed (30.3s).
- `python -m compileall -q xccv labelme`: passed.
- Offscreen UI smoke tests: training platform (5 task types), testing platform (dual pages /
  review page), flowchart page, threshold panel and data grid all instantiate.
- Real-machine UI acceptance: workbench → training task dropdown (anomaly / OCR) → testing
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
