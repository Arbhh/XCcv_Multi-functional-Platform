"""纯逻辑测试：开发者选项、数据体检（提示/警告两档）、预检与智能推荐。"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.synthetic import make_classification_dataset, make_labelme_dataset
from xccv.engine.dataset.base import TaskType
from xccv.training import dev_options as dev_mod
from xccv.training.data_health import (
    LEVEL_INFO, LEVEL_WARN, run_health_check,
)
from xccv.training.dev_options import DevOptions
from xccv.training.preflight import (
    DatasetFacts, dataset_facts, param_findings, recommend_settings,
)


class _TempHome(unittest.TestCase):
    """把开发者选项文件重定向到临时目录，避免污染真实 platform_data。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._patcher = mock.patch.object(
            dev_mod, "options_path",
            return_value=Path(self._tmp.name) / "dev_options.json")
        self._patcher.start()
        self.addCleanup(self._patcher.stop)


class DevOptionsTest(_TempHome):
    def test_defaults_and_roundtrip(self):
        opt = dev_mod.load_options()
        self.assertEqual(opt.imbalance_share, 0.05)  # 占比口径
        self.assertEqual(opt.min_cls_samples, 15)
        self.assertEqual(opt.min_det_instances, 30)
        self.assertTrue(opt.preflight_enabled)
        opt.imbalance_share = 0.03
        opt.min_det_instances = 10
        opt.preflight_enabled = False
        dev_mod.save_options(opt)
        loaded = dev_mod.load_options()
        self.assertEqual(loaded.imbalance_share, 0.03)
        self.assertEqual(loaded.min_det_instances, 10)
        self.assertFalse(loaded.preflight_enabled)

    def test_corrupt_file_falls_back(self):
        p = dev_mod.options_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{"imbalance_share": "abc", "min_cls_samples": 30}',
                     encoding="utf-8")
        opt = dev_mod.load_options()
        self.assertEqual(opt.imbalance_share, 0.05)  # 坏字段回退默认
        self.assertEqual(opt.min_cls_samples, 30)    # 好字段保留


class HealthCheckTest(_TempHome):
    def setUp(self):
        super().setUp()
        self._dir_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir_tmp.cleanup)
        self.root = Path(self._dir_tmp.name)

    def test_levels_only_info_warn(self):
        """铁律：所有条目只允许 提示/警告 两档（不阻止训练）。"""
        det, _ = make_labelme_dataset(self.root / "det", n_images=4)
        report = run_health_check(TaskType.DETECTION, det)
        for issue in report.issues:
            self.assertIn(issue.level, (LEVEL_INFO, LEVEL_WARN))

    def _relabel(self, det_dir: Path, rule):
        """按 rule(i) 重写各 json 的类别（构造指定分布）。"""
        import json as _json

        for i, jp in enumerate(sorted(det_dir.glob("img*.json"))):
            data = _json.loads(jp.read_text(encoding="utf-8"))
            data["shapes"][0]["label"] = rule(i)
            jp.write_text(_json.dumps(data, ensure_ascii=False),
                          encoding="utf-8")
        (det_dir / "classes.txt").write_text("cat\ndog\n", encoding="utf-8")

    def test_imbalance_and_few_samples_detection(self):
        det, _ = make_labelme_dataset(self.root / "det", n_images=30,
                                      classes=("cat", "dog"))
        # 占比口径：最小类 1/30 = 3.3% < 5% 才触发
        self._relabel(det, lambda i: "cat" if i < 29 else "dog")
        report = run_health_check(TaskType.DETECTION, det, val_ratio=0.25)
        codes = {i.code for i in report.issues}
        self.assertIn("imbalance", codes)
        self.assertIn("few_samples", codes)
        imb = next(i for i in report.issues if i.code == "imbalance")
        self.assertEqual(imb.level, LEVEL_WARN)

    def test_balanced_big_dataset_no_few_warning(self):
        opt = DevOptions(min_det_instances=2, min_cls_samples=2)
        det, _ = make_labelme_dataset(self.root / "det", n_images=8)
        report = run_health_check(TaskType.DETECTION, det, options=opt,
                                  val_ratio=0.25)
        self.assertNotIn("few_samples", {i.code for i in report.issues})
        self.assertNotIn("imbalance", {i.code for i in report.issues})

    def test_classification_few_samples(self):
        cls, _ = make_classification_dataset(self.root / "cls", per_class=3)
        report = run_health_check(TaskType.CLASSIFICATION, cls)
        few = next(i for i in report.issues if i.code == "few_samples")
        self.assertEqual(few.level, LEVEL_WARN)
        self.assertIn("15", few.detail)

    def test_duplicate_images_detected(self):
        import numpy as np

        from xccv.engine.io_utils import imwrite_unicode, imread_unicode

        det, _ = make_labelme_dataset(self.root / "det", n_images=4)
        # 复制一张图为两个新名字（同一内容 → dHash 相同），json 一并复制入扫
        src = self.root / "det" / "img00.png"
        img = imread_unicode(src)
        imwrite_unicode(self.root / "det" / "dup_a.png", img)
        imwrite_unicode(self.root / "det" / "dup_b.png", img)
        (self.root / "det" / "dup_a.json").write_text(
            src.with_suffix(".json").read_text(encoding="utf-8"), encoding="utf-8")
        (self.root / "det" / "dup_b.json").write_text(
            src.with_suffix(".json").read_text(encoding="utf-8"), encoding="utf-8")
        report = run_health_check(TaskType.DETECTION, self.root / "det")
        dup = next((i for i in report.issues if i.code == "duplicates"), None)
        self.assertIsNotNone(dup)
        self.assertEqual(dup.level, LEVEL_INFO)

    def test_size_outlier_detected(self):
        import json as _json

        import numpy as np

        from xccv.engine.io_utils import imwrite_unicode

        det, _ = make_labelme_dataset(self.root / "det", n_images=6, size=(96, 96))
        big = np.zeros((480, 640, 3), dtype=np.uint8)
        imwrite_unicode(self.root / "det" / "huge.png", big)
        meta = _json.loads((det / "img00.json").read_text(encoding="utf-8"))
        meta["imagePath"] = "huge.png"
        meta["imageWidth"], meta["imageHeight"] = 640, 480
        (det / "huge.json").write_text(_json.dumps(meta, ensure_ascii=False),
                                       encoding="utf-8")
        report = run_health_check(TaskType.DETECTION, self.root / "det")
        self.assertIn("size_outlier", {i.code for i in report.issues})

    def test_val_class_missing(self):
        # val_ratio 很小 + 每类样本少 → 某类可能不进验证集（提示级）
        opt = DevOptions(min_cls_samples=1, min_det_instances=1)
        det, _ = make_labelme_dataset(self.root / "det", n_images=4)
        report = run_health_check(TaskType.DETECTION, det, options=opt,
                                  val_ratio=0.0)
        # val_ratio=0 时不检查覆盖
        self.assertNotIn("val_missing", {i.code for i in report.issues})

    def test_empty_dataset_warn(self):
        empty = self.root / "empty"
        empty.mkdir()
        report = run_health_check(TaskType.DETECTION, empty)
        issue = next(i for i in report.issues if i.code == "empty")
        self.assertEqual(issue.level, LEVEL_WARN)


class PreflightTest(unittest.TestCase):
    def _project(self, **kw):
        from xccv.training.project import Project

        base = dict(name="t", dataset_dir="d", model_dir="m",
                    task="detection", model_name="yolo11n",
                    epochs=30, patience=10, learning_rate=0.01,
                    val_ratio=0.2, batch_size=4, multiply_factor=1,
                    augment_ops={"resize": {"width": 640, "height": 640}},
                    device="auto", use_amp=True)
        base.update(kw)
        return Project(**base)

    def test_patience_warning_with_apply(self):
        p = self._project(patience=50)
        findings = param_findings(p, DatasetFacts(), cuda_available=True)
        f = next(f for f in findings if f.code == "patience")
        self.assertEqual(f.level, LEVEL_WARN)
        self.assertEqual(f.field, "patience")
        self.assertEqual(f.value, 10)  # max(1, 30//3)

    def test_lr_outlier(self):
        p = self._project(learning_rate=0.5)
        f = next(f for f in param_findings(p) if f.code == "learning_rate")
        self.assertEqual(f.level, LEVEL_WARN)
        self.assertEqual(f.value, 0.01)

    def test_val_zero_and_batch_hints(self):
        p = self._project(val_ratio=0.0, batch_size=32)
        codes = {f.code: f for f in param_findings(p)}
        self.assertIn("val_zero", codes)
        self.assertEqual(codes["val_zero"].level, LEVEL_INFO)
        self.assertIn("batch_large", codes)
        self.assertEqual(codes["batch_large"].value, 8)

    def test_resize_off_with_aspect(self):
        p = self._project(augment_ops={})
        facts = DatasetFacts(class_counts={"a": 10}, aspect_ratio=2.5)
        f = next(f for f in param_findings(p, facts) if f.code == "resize_off")
        self.assertEqual(f.field, "resize")
        self.assertEqual(f.value, (640, 640))

    def test_no_cuda_device_warning(self):
        p = self._project(device="cuda")
        codes = {f.code: f for f in param_findings(p, cuda_available=False)}
        self.assertIn("device_no_cuda", codes)
        self.assertEqual(codes["device_no_cuda"].level, LEVEL_WARN)
        self.assertEqual(codes["device_no_cuda"].value, "auto")

    def test_all_findings_are_two_level(self):
        p = self._project(patience=99, learning_rate=0.9, val_ratio=0.0,
                          batch_size=64, multiply_factor=20)
        for f in param_findings(p, DatasetFacts(aspect_ratio=3.0), cuda_available=False):
            self.assertIn(f.level, (LEVEL_INFO, LEVEL_WARN))

    def test_recommend_small_dataset(self):
        p = self._project(freeze_backbone=False, industrial_opt=False)
        facts = DatasetFacts(class_counts={"scrach": 120}, n_images=120)
        recs = recommend_settings(p, facts, cuda_available=True)
        keys = [k for k, _, _ in recs]
        self.assertIn("freeze_backbone", keys)
        self.assertIn("industrial_opt", keys)
        self.assertIn("augment_probs", keys)

    def test_recommend_rcnn_and_no_amp_on_cpu(self):
        p = self._project(model_name="maskrcnn")
        recs = recommend_settings(p, DatasetFacts(class_counts={"a": 900}),
                                  cuda_available=False)
        keys = [k for k, _, _ in recs]
        self.assertIn("rpn_reduced", keys)
        self.assertIn("use_amp", keys)

    def test_recommend_conservative_silent(self):
        p = self._project(freeze_backbone=True, industrial_opt=True)
        facts = DatasetFacts(class_counts={"a": 2000}, n_images=4000)
        self.assertEqual(recommend_settings(p, facts, cuda_available=True), [])

    def test_dataset_facts_from_samples(self):
        det, _ = make_labelme_dataset(Path(tempfile.mkdtemp()) / "det", n_images=4)
        from xccv.engine.dataset.reader import scan_labelme_dataset

        samples, _, _ = scan_labelme_dataset(det)
        facts = dataset_facts(samples)
        self.assertEqual(facts.n_images, 4)
        self.assertEqual(facts.class_counts, {"cat": 2, "dog": 2})
        self.assertAlmostEqual(facts.aspect_ratio, 1.0)


if __name__ == "__main__":
    unittest.main()

