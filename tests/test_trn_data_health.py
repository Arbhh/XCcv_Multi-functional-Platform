"""训练平台测试：3 个历史测试模块的合并模块。

由以下模块按域合并而成，用例逐段原样迁移、断言未改：
- test_data_health
- test_data_contracts
- test_v073_round16
"""


# == 来源模块：test_data_health ==
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

# == 来源模块：test_data_contracts ==
"""数据契约测试：labelme JSON → 内部标注 → YOLO 行的往返一致性，
以及流程图 IR 的端口 schema 拒绝语义（评审建议的契约回归锁）。

这些测试锁住"标注格式"与"流程图结构"两条对外数据契约：
改动任何一侧的序列化/校验行为时，必须显式更新本文件并说明原因。
"""

import json
import tempfile
import unittest
from pathlib import Path

from xccv.engine.dataset.base import Sample, Shape
from xccv.engine.dataset.converters import _yolo_det_line, _yolo_seg_line, split_counts, split_samples
from xccv.engine.dataset.reader import is_negative_dir_name, read_labelme_json


def _rect(label, x1, y1, x2, y2):
    return Shape(label=label, shape_type="rectangle",
                 points=[[x1, y1], [x2, y2]])


class YoloLineContract(unittest.TestCase):
    """YOLO 行契约：归一化到 [0,1]、越界裁剪、画面外目标返回 None。"""

    def test_det_line_normalized(self):
        sh = _rect("缺陷", 100, 80, 300, 240)
        line = _yolo_det_line(sh, 640, 480, 2)
        parts = line.split(" ")
        self.assertEqual(len(parts), 5)
        self.assertEqual(parts[0], "2")
        cx, cy, w, h = (float(v) for v in parts[1:])
        self.assertTrue(all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)))
        self.assertAlmostEqual(w, 200.0 / 640.0, places=6)
        self.assertAlmostEqual(h, 160.0 / 480.0, places=6)
        self.assertAlmostEqual(cx, 200.0 / 640.0, places=6)
        self.assertAlmostEqual(cy, 160.0 / 480.0, places=6)

    def test_det_line_clipped_to_frame(self):
        sh = _rect("缺陷", -50, -40, 2000, 1600)
        line = _yolo_det_line(sh, 640, 480, 0)
        _c, cx, cy, w, h = (float(v) for v in line.split(" "))
        self.assertLessEqual(cx + w / 2, 1.0 + 1e-6)
        self.assertLessEqual(cy + h / 2, 1.0 + 1e-6)
        self.assertGreaterEqual(cx - w / 2, -1e-6)

    def test_det_line_none_when_out_of_frame(self):
        sh = _rect("缺陷", 800, 600, 900, 700)
        self.assertIsNone(_yolo_det_line(sh, 640, 480, 0))

    def test_seg_line_closed_ring(self):
        pts = [[10, 10], [110, 10], [110, 110], [10, 110]]
        sh = Shape(label="缺陷", shape_type="polygon", points=pts)
        line = _yolo_seg_line(sh, 640, 480, 1)
        vals = [float(v) for v in line.split(" ")]
        cls, coords = vals[0], vals[1:]
        self.assertEqual(cls, 1)
        self.assertEqual(len(coords) % 2, 0)
        self.assertGreaterEqual(len(coords), 6)   # 至少 3 点
        self.assertTrue(all(0.0 <= v <= 1.0 for v in coords))
        # 多边形坐标顺序保持（首点 = 原始首点归一化）
        self.assertAlmostEqual(coords[0], 10.0 / 640.0, places=6)
        self.assertAlmostEqual(coords[1], 10.0 / 480.0, places=6)


class SplitConsistencyContract(unittest.TestCase):
    """split_counts 与 split_samples 同口径（环状图推算的契约）。"""

    @staticmethod
    def _samples(n):
        return [Sample(image_path=Path(f"{i}.png"), width=640, height=480,
                       label=f"c{i % 3}") for i in range(n)]

    def test_counts_match_samples_mutually_exclusive(self):
        for n in (10, 67, 120):
            tr, va = split_samples(self._samples(n), 0.2, seed=7, train_ratio=0.8)
            ctr, cva = split_counts(n, 0.2, 0.8)
            self.assertEqual((len(tr), len(va)), (ctr, cva), f"n={n}")

    def test_counts_match_samples_overlap(self):
        for n in (50, 100):
            tr, va = split_samples(self._samples(n), 0.3, seed=7, train_ratio=0.9)
            ctr, cva = split_counts(n, 0.3, 0.9)
            self.assertEqual((len(tr), len(va)), (ctr, cva), f"n={n}")

    def test_counts_match_samples_val_zero_reuse(self):
        tr, va = split_samples(self._samples(30), 0.0, seed=7, train_ratio=1.0)
        ctr, cva = split_counts(30, 0.0, 1.0)
        self.assertEqual((len(tr), len(va)), (ctr, cva))
        self.assertEqual(len(tr), len(va))   # 验证集复用训练集


class LabelmeReaderContract(unittest.TestCase):
    """labelme JSON → Sample 的读取契约。"""

    def test_read_labelme_json_shapes(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            img = d / "a.png"
            img.write_bytes(b"\x89PNG fake")
            jp = d / "a.json"
            jp.write_text(json.dumps({
                "version": "5.8.1", "flags": {},
                "shapes": [
                    {"label": "黑点", "points": [[1, 2], [30, 40]],
                     "shape_type": "rectangle", "group_id": None, "flags": {}},
                    {"label": "划痕", "points": [[5, 5], [20, 8], [25, 30]],
                     "shape_type": "polygon", "group_id": None, "flags": {}},
                ],
                "imagePath": "a.png", "imageData": None,
                "imageHeight": 480, "imageWidth": 640,
            }), encoding="utf-8")
            s = read_labelme_json(jp)
            self.assertEqual(s.image_path, img)
            self.assertEqual([sh.label for sh in s.shapes], ["黑点", "划痕"])
            self.assertEqual(s.shapes[0].shape_type, "rectangle")
            self.assertEqual(s.shapes[1].points[2], [25.0, 30.0])

    def test_ok_dir_names(self):
        self.assertTrue(is_negative_dir_name("OK"))
        self.assertTrue(is_negative_dir_name("ok"))
        self.assertTrue(is_negative_dir_name("良品"))
        self.assertFalse(is_negative_dir_name("NG"))
        self.assertFalse(is_negative_dir_name("缺陷"))


class FlowSchemaContract(unittest.TestCase):
    """流程图 IR 结构契约：未知类型 / 槽越界 / 类型不匹配 必须被拒绝。

    图内 type 用注册名（无 xccv2/ 前缀——流程文件保存时带前缀，
    由 flow_session._normalize_types 剥除后进入 Executor，此处同语义）。
    """

    @staticmethod
    def _graph(nodes, links):
        return {"nodes": nodes, "links": links}

    @classmethod
    def setUpClass(cls):
        from xccv.engine.flow.ops import load_all

        load_all()

    def test_unknown_node_type_rejected(self):
        from xccv.engine.flow.executor import Executor, FlowError

        g = self._graph(
            [{"id": 1, "type": "no_such_node", "properties": {}}],
            [])
        with self.assertRaises(FlowError):
            Executor(g).validate()

    def test_slot_overflow_rejected(self):
        # 连线为 LiteGraph 5 元组 [link_id, src, sslot, dst, dslot]；
        # 输出槽越界在构造期即被拒绝（IndexError），validate 前不可能通过
        from xccv.engine.flow.executor import Executor

        g = self._graph(
            [{"id": 1, "type": "image_loader", "properties": {"path": "x"}},
             {"id": 2, "type": "result_output", "properties": {}}],
            [[1, 1, 99, 2, 0]])   # 输出槽 99 越界
        with self.assertRaises(Exception):
            Executor(g).validate()

    def test_port_type_mismatch_rejected(self):
        from xccv.engine.flow.executor import Executor, FlowError

        g = self._graph(
            [{"id": 1, "type": "image_loader", "properties": {"path": "x"}},
             {"id": 2, "type": "result_output", "properties": {}}],
            [[1, 1, 0, 2, 1]])    # image 口 → result 口，类型不匹配
        with self.assertRaises(FlowError):
            Executor(g).validate()

# == 来源模块：test_v073_round16 ==
"""回归锁：遗留待办 §二/§三/§四 + SubspaceAD。

- §二 复盘统一口径：batch_match ANOMALY 图像级弱 GT（OK 过检/NG 漏检/
  NG 检出=正确）、分类对错分组；review_logic 徽章/统计行随任务改字；
- §三 分数语义：BaseModel.score_semantics 元数据、ANOMALY z 语义覆写、
  阈值面板动态轴（无上界）、result_json additive 键；
- §四 免训练范式：registry MODEL_META、阶段化进度编码（1/2/3）、
  TrainConfig.model_params、阶段化进度回调；
- SubspaceAD：PCA 数学核心（两遍流式拟合/重构残差）、层选择、聚合、
  发现链、注册项（骨干权重缺失时数学项照测）。
"""

import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from xccv.engine.dataset.base import TaskType
from xccv.engine.metrics.batch_match import collect_match_stats
from xccv.testing.review_logic import classify_images, stats_line

# watcher 对象跨测试保活（处理完文件变化即析构会撞原生竞态）
_WATCHER_KEEPER: list = []


class LangWatcherCacheTest(unittest.TestCase):
    """语言偏好文件是跨进程共享状态，watcher
    _fire 曾走 lang() 进程缓存——非"保存"进程缓存恒为旧值，平台窗口
    永不重建。"""

    def test_external_file_change_fires_rebuild(self):
        from PyQt5.QtWidgets import QApplication

        from xccv.i18n import _LANG_FILE, install_language_watcher, lang, set_lang

        set_lang("zh")                      # 预热进程缓存（模拟平台启动态）
        self.assertEqual(lang(), "zh")
        app = QApplication.instance() or QApplication([])
        fired = []
        w = install_language_watcher(lambda: fired.append(1), parent=None)
        if w is None:
            self.skipTest("环境不可监听")
        _WATCHER_KEEPER.append(w)           #1429进程结束前不析构
        try:
            # 模拟另一进程保存 EN：直写文件（不经本进程 set_lang）
            _LANG_FILE.write_text(json.dumps({"lang": "en"}), encoding="utf-8")
            deadline = time.time() + 4
            while time.time() < deadline and not fired:
                app.processEvents()
                time.sleep(0.05)
            self.assertTrue(fired, "外部改语言文件未触发重建")
            self.assertEqual(lang(), "en")  # 进程缓存已同步（重建渲染 EN 的前提）
        finally:
            set_lang("zh")                  #1937复位偏好与缓存



def _res(path: str, score=None, dets=None, classifications=None):
    r = {"file": Path(path).name, "path": path, "width": 64, "height": 64,
         "detections": dets or []}
    if score is not None:
        r["image_score"] = score
    if classifications is not None:
        r["classifications"] = classifications
    return r


class AnomalyImageLevelTest(unittest.TestCase):
    """§二：ANOMALY 图像级口径（来源目录 = 弱 GT）。"""

    def test_ok_ng_groups(self):
        results = [
            _res("d:/ds/OK/a.png", score=5.0),    # OK 有过阈检出 → 过检
            _res("d:/ds/OK/b.png", score=0.5),    # OK 低分（阈值下无检出）→ 正确
            _res("d:/ds/NG/c.png", score=8.2),    # NG 有检出 → 正确
            _res("d:/ds/NG/d.png", score=None),   # NG 零检出 → 漏检
            _res("d:/ds/other/e.png", score=3.0),  # 目录不在类别集 → 未评估
        ]
        weak = {"OK", "NG"}
        stats = collect_match_stats(results, iou_thr=0.5, task="anomaly",
                                    score_semantics={"key": "z"},
                                    weak_classes=weak)
        self.assertEqual(stats["task"], "anomaly")
        self.assertEqual(stats["n_ok_images"], 2)
        self.assertEqual(stats["gt_total"], 2)          # NG 图数
        groups = classify_images(stats, {"score": 3.0})
        self.assertEqual([e["path"][-5] for e in groups["over"]], ["a"])
        self.assertEqual([e["path"][-5] for e in groups["missed"]], ["d"])
        self.assertEqual(sorted(e["path"][-5] for e in groups["correct"]),
                         ["b", "c"])             # OK 低分 + NG 检出
        self.assertEqual(len(groups["unjudged_list"]), 1)
        self.assertEqual(stats_line(groups, "evaluated", "anomaly").count("过检"), 1)
        # 回归锁：OK 图检出记录 matched=False（误报不进 TP）——
        # 否则精确率/召回率把 OK 误报计成命中
        ok_entry = next(e for e in stats["per_image"] if e["path"].endswith("a.png"))
        self.assertFalse(ok_entry["dets"][0][1])

    def test_score_semantics_propagated(self):
        stats = collect_match_stats([_res("d:/ds/NG/x.png", score=5.0)],
                                    task="anomaly",
                                    score_semantics={"key": "z", "name": "图像分数"})
        self.assertEqual(stats["score_semantics"]["name"], "图像分数")

    def test_threshold_sweep_image_level(self):
        """阈值现算复用：阈值 1.0 时 OK(1.2) 过检、NG(8.2) 正确；阈值 3.0
        时两者皆正确（OK 检出被阈值筛掉）。"""
        results = [_res("d:/ds/OK/a.png", score=1.2),
                   _res("d:/ds/NG/c.png", score=8.2)]
        stats = collect_match_stats(results, task="anomaly",
                                    weak_classes={"OK", "NG"})
        low = classify_images(stats, {"score": 1.0})
        self.assertEqual([e["path"][-5] for e in low["over"]], ["a"])
        self.assertEqual([e["path"][-5] for e in low["correct"]], ["c"])
        high = classify_images(stats, {"score": 3.0})
        self.assertEqual(sorted(e["path"][-5] for e in high["correct"]),
                         ["a", "c"])
        self.assertEqual(len(high["over"]), 0)


class ClassificationImageLevelTest(unittest.TestCase):
    """§二：分类对/错分组（预测类别 vs 目录真值）。"""

    def test_correct_wrong_groups(self):
        results = [
            _res("d:/ds/划伤/a.png", classifications=[{"label": "划伤", "score": 0.9}]),
            _res("d:/ds/划伤/b.png", classifications=[{"label": "污点", "score": 0.8}]),
            _res("d:/ds/污点/c.png", classifications=[{"label": "污点", "score": 0.7}]),
        ]
        stats = collect_match_stats(results, task="classification",
                                    weak_classes={"划伤", "污点"})
        self.assertEqual(stats["task"], "classification")
        self.assertEqual(stats["gt_total"], 3)
        groups = classify_images(stats, {"score": 0.0})
        self.assertEqual([e["path"][-5] for e in groups["correct"]],
                         ["a", "c"])
        self.assertEqual([e["path"][-5] for e in groups["over"]], ["b"])
        self.assertEqual([e["path"][-5] for e in groups["missed"]], [])
        self.assertIn("错误 1", stats_line(groups, "evaluated", "classification"))
        self.assertIn("错1", groups["over"][0]["badges"])

    def test_unknown_dir_unjudged(self):
        results = [_res("d:/ds/未知/a.png",
                        classifications=[{"label": "划伤", "score": 0.9}])]
        stats = collect_match_stats(results, task="classification",
                                    weak_classes={"划伤", "污点"})
        groups = classify_images(stats, {})
        self.assertEqual(len(groups["unjudged_list"]), 1)


class ScoreSemanticsTest(unittest.TestCase):
    """§三：分数语义元数据。"""

    def test_base_default(self):
        from xccv.engine.models.base import BaseModel

        self.assertEqual(BaseModel.score_semantics["key"], "conf")
        self.assertEqual(BaseModel.score_semantics["range"], [0.0, 1.0])

    def test_anomaly_model_semantics(self):
        from xccv.engine.models.anomaly import AnomalyModel

        m = AnomalyModel(1, device="cpu", use_pretrained=False)
        self.assertEqual(m.task, TaskType.ANOMALY)
        self.assertEqual(m.score_semantics["key"], "z")
        self.assertIsNone(m.score_semantics["range"][1])
        self.assertEqual(m.train_mode if hasattr(m, "train_mode") else "free",
                         getattr(m, "train_mode", "free"))

    def test_classifier_semantics(self):
        from xccv.engine.models.classifier import ClassifierModel

        m = ClassifierModel("resnet18", 2, device="cpu", use_pretrained=False)
        self.assertEqual(m.task, TaskType.CLASSIFICATION)
        self.assertEqual(m.score_semantics["key"], "prob")

    def test_registry_meta(self):
        from xccv.engine.models.registry import MODEL_META, list_models

        self.assertEqual(MODEL_META["cif"]["train_mode"], "free")
        self.assertTrue(MODEL_META["subspacead"]["support_k_shot"])
        items = list_models(TaskType.ANOMALY)
        names = {m["name"] for m in items}
        self.assertIn("subspacead", names)
        for m in items:
            self.assertIn("train_mode", m)
            self.assertIn("support_k_shot", m)

    def test_result_json_additive(self):
        from xccv.engine.models.base import Prediction
        from xccv.engine.result_json import build_result_json

        payload = build_result_json(
            task="anomaly", source={"file": "a.png", "path": "d:/a.png",
                                    "width": 8, "height": 8},
            prediction=Prediction(probs=[3.2]),
            classes=["异常"],
            score_semantics={"key": "z", "name": "图像分数",
                             "range": [0.0, None], "higher": "越高越异常"})
        self.assertEqual(payload["score_semantics"]["key"], "z")
        self.assertIsNone(payload["score_semantics"]["range"][1])
        # 不传语义时不输出键（schema 只增不改，旧消费方零感知）
        payload2 = build_result_json(task="detection",
                                     source={"file": "a.png", "path": "p",
                                             "width": 8, "height": 8},
                                     prediction=Prediction())
        self.assertNotIn("score_semantics", payload2)


class FreeTrainModeTest(unittest.TestCase):
    """§四：免训练范式引擎侧契约。"""

    def test_train_config_model_params(self):
        from xccv.engine.models.base import TrainConfig

        cfg = TrainConfig(task=TaskType.ANOMALY,
                          dataset_dir=Path("."), output_dir=Path("."),
                          model_name="cif",
                          model_params={"k_shot": 6, "aug_count": 30})
        self.assertEqual(cfg.model_params["k_shot"], 6)

    def test_free_models_declared(self):
        """ANOMALY 后端实例声明 z 语义与 free 形态（CIF 需 DINO 权重，
        缺失时由 test_v073_cif 的 skipUnless 路径覆盖）。"""
        from xccv.engine.models.anomaly import AnomalyModel
        from xccv.engine.models.subspacead import SubspaceAdModel

        m = AnomalyModel(1, device="cpu", use_pretrained=False)
        self.assertEqual(m.score_semantics["key"], "z")
        s = SubspaceAdModel(1, device="cpu", use_pretrained=False)
        self.assertEqual(s.score_semantics["key"], "z")
        self.assertEqual(s.train_mode, "free")


class SubspaceAdMathTest(unittest.TestCase):
    """SubspaceAD 数学核心（无需骨干权重）。"""

    def test_layer_indices(self):
        from xccv.engine.models.subspacead import _layer_indices

        self.assertEqual(_layer_indices(24), [-12, -13, -14, -15, -16, -17, -18])
        self.assertEqual(_layer_indices(40), [-12, -13, -14, -15, -16, -17, -18])
        self.assertEqual(_layer_indices(12), [-1, -2, -3, -4, -5, -6, -7])

    def test_pca_residual_low_in_subspace(self):
        """2 维高斯数据落在 1 维子空间 → 子空间内残差 ≈ 0、法向外点残差大。"""
        from xccv.engine.models.subspacead import SubspaceAdModel

        rng = np.random.default_rng(0)
        t = rng.normal(0, 3, size=(2000, 1))
        data = np.hstack([t, t * 2.0]) + rng.normal(0, 1e-6, (2000, 2))
        batches = [torch.from_numpy(data.astype(np.float32))]
        pca = SubspaceAdModel._fit_pca(batches, 2, data.shape[0], "cpu", 0.999)
        self.assertEqual(pca["k"], 1)
        mu, comp = pca["mu"], pca["components"]
        inlier = torch.from_numpy(data[:100].astype(np.float32))
        out = np.array([[5.0, -6.0]], dtype=np.float32)  # 法向偏离
        r_in = SubspaceAdModel._residual_scores(inlier, mu, comp).numpy()
        r_out = SubspaceAdModel._residual_scores(
            torch.from_numpy(out), mu, comp).numpy()
        self.assertLess(float(np.median(r_in)), 1e-3)
        self.assertGreater(float(r_out[0]), float(np.median(r_in)) * 100)

    def test_topk_mean(self):
        from xccv.engine.models.subspacead import _topk_mean

        arr = np.zeros((10, 10), dtype=np.float32)
        arr[0, 0] = 8.0
        arr[0, 1] = 6.0     # top1% = top1 像素（100 像素 → k=1）
        self.assertAlmostEqual(_topk_mean(arr), 8.0, places=4)

    def test_post_process_shape(self):
        from xccv.engine.models.subspacead import _post_process

        out = _post_process(np.ones((7, 7), dtype=np.float32), 56)
        self.assertEqual(out.shape, (56, 56))

    def test_find_backbone_returns(self):
        """DINOv2 骨干可发现时返回 (来源, 16 位指纹)；缺失则跳过（首用时自动接管下载）。"""
        from xccv.engine.models.subspacead import _find_backbone

        try:
            src, sha = _find_backbone()
        except FileNotFoundError:
            self.skipTest("未放置 DINOv2 特征骨干（可选；训练/推理首用时自动接管下载）")
            return
        self.assertIsInstance(src, str)
        self.assertEqual(len(sha), 16)

    def test_registry_load_format_known(self):
        """注册名与容器格式契约（load_model 分支键一致）。"""
        from xccv.engine.models.registry import MODEL_SPECS

        self.assertIn("subspacead", MODEL_SPECS)
        task, desc, _f = MODEL_SPECS["subspacead"]
        self.assertEqual(task, TaskType.ANOMALY)


import torch  # noqa: E402 （SubspaceAdMathTest 用；置底避免顶部拖慢其它用例）


# == 数据工具纯逻辑（dataops）契约 ==

def _mk_png(path, w=80, h=60, color=(200, 200, 200)):
    import cv2
    import numpy as np
    ok, buf = cv2.imencode(".png", np.full((h, w, 3), color, np.uint8))
    assert ok
    buf.tofile(str(path))
    return path


def _mk_pair(dir_path, stem, label, w=80, h=60):
    import json
    _mk_png(dir_path / (stem + ".png"), w, h)
    data = {
        "version": "5.2.1",
        "shapes": [{
            "label": label, "points": [[10, 10], [40, 10], [40, 40]],
            "group_id": None, "shape_type": "polygon", "flags": {},
        }],
        "imagePath": stem + ".png", "imageData": None,
        "imageHeight": h, "imageWidth": w,
        "flags": {label: True} if label == "划痕" else {},
    }
    (dir_path / (stem + ".json")).write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return dir_path / (stem + ".json")


class DataopsScanTest(unittest.TestCase):
    """scan_pairs：图片锚定成对、未标注纯图、孤儿 json 三类。"""

    def test_scan_counts(self):
        import tempfile
        from xccv.engine.dataops.labelops import scan_pairs
        d = Path(tempfile.mkdtemp())
        _mk_pair(d, "a", "划痕")
        _mk_png(d / "b.png")           # 未标注
        (d / "c.json").write_text("{}", encoding="utf-8")   # 孤儿 json
        pairs = scan_pairs(d)
        self.assertEqual(len(pairs), 3)
        labeled = [p for p in pairs if p.labeled]
        self.assertEqual(len(labeled), 2)
        unlabeled = [p for p in pairs if not p.labeled]
        self.assertEqual(len(unlabeled), 1)
        self.assertEqual(unlabeled[0].image.name, "b.png")


class DataopsRecursiveScanTest(unittest.TestCase):
    """递归扫描：子目录成对/未标注、同名不同目录不串味。"""

    def test_scan_includes_subdirectories(self):
        import tempfile
        from xccv.engine.dataops.labelops import scan_pairs
        root = Path(tempfile.mkdtemp())
        (root / "train").mkdir()
        (root / "val").mkdir()
        _mk_pair(root, "a", "划痕")
        _mk_pair(root / "train", "b", "划痕")
        _mk_png(root / "val" / "c.png")
        pairs = scan_pairs(root)
        self.assertEqual(len(pairs), 3)
        self.assertTrue(any(p.labeled and p.json.parent.name == "train"
                            for p in pairs))
        self.assertTrue(any(not p.labeled and p.image.parent.name == "val"
                            for p in pairs))

    def test_same_stem_in_different_dirs_stays_separate(self):
        import tempfile
        from xccv.engine.dataops.labelops import scan_pairs
        root = Path(tempfile.mkdtemp())
        a, b = root / "a", root / "b"
        a.mkdir(), b.mkdir()
        _mk_pair(a, "img", "划痕")
        _mk_png(b / "img.png")     # 同名未标注图，不得借到 a 的 json
        pairs = scan_pairs(root)
        labeled = [p for p in pairs if p.labeled]
        unlabeled = [p for p in pairs if not p.labeled]
        self.assertEqual(len(labeled), 1)
        self.assertEqual(labeled[0].json.parent, a)
        self.assertEqual(len(unlabeled), 1)
        self.assertEqual(unlabeled[0].image, b / "img.png")


class DataopsReplaceTest(unittest.TestCase):
    """替换标签：形状 label + flags 键同步；正则整词。"""

    def _prepare(self):
        import tempfile
        d = Path(tempfile.mkdtemp())
        _mk_pair(d, "a", "划痕")
        return d

    def test_replace_plain_and_flags(self):
        from xccv.engine.dataops.labelops import (
            apply_replace, iter_labels, plan_replace, scan_pairs)
        d = self._prepare()
        entries = plan_replace(scan_pairs(d), "划痕", "裂纹")
        self.assertEqual(len(entries), 1)
        apply_replace(entries)
        labels = iter_labels(scan_pairs(d)[0])
        self.assertIn("裂纹", labels)
        self.assertNotIn("划痕", labels)

    def test_replace_regex_fullmatch(self):
        from xccv.engine.dataops.labelops import (
            apply_replace, iter_labels, plan_replace, scan_pairs)
        d = self._prepare()
        entries = plan_replace(scan_pairs(d), "划.*", "裂纹", regex=True)
        self.assertEqual(len(entries), 1)
        apply_replace(entries)
        self.assertIn("裂纹", iter_labels(scan_pairs(d)[0]))
        entries2 = plan_replace(scan_pairs(d), "痕", "X", regex=True)
        self.assertEqual(len(entries2), 0)  # 整词不匹配子串


class DataopsDeleteTest(unittest.TestCase):
    """删除标签：删形状 / 整图过滤；删空 json 消失；可选备份。"""

    def test_delete_shapes_and_empty_json_removed(self):
        import json
        import tempfile
        from xccv.engine.dataops.labelops import apply_delete, plan_delete, scan_pairs
        d = Path(tempfile.mkdtemp())
        _mk_pair(d, "a", "划痕")
        plan = plan_delete(scan_pairs(d), ["划痕"], mode="shapes")
        self.assertEqual(len(plan), 1)
        apply_delete(plan)
        self.assertFalse((d / "a.json").exists())   # 空标注不落盘
        self.assertTrue((d / "a.png").exists())

    def test_delete_images_with_backup(self):
        import tempfile
        from xccv.engine.dataops.labelops import apply_delete, plan_delete, scan_pairs
        d = Path(tempfile.mkdtemp())
        _mk_pair(d, "a", "划痕")
        backup = Path(tempfile.mkdtemp())
        plan = plan_delete(scan_pairs(d), ["划痕"], mode="images")
        apply_delete(plan, backup_dir=backup)
        self.assertFalse((d / "a.png").exists())
        self.assertFalse((d / "a.json").exists())
        self.assertTrue((backup / "a.png").exists())
        self.assertTrue((backup / "a.json").exists())


class DataopsMoveTest(unittest.TestCase):
    """迁移：按标签命中、未标注纳入、同名冲突跳过。"""

    def _prepare(self):
        import tempfile
        d = Path(tempfile.mkdtemp())
        _mk_pair(d, "a", "划痕")
        _mk_png(d / "b.png")
        return d

    def test_move_labeled_and_unlabeled(self):
        import tempfile
        from xccv.engine.dataops.labelops import apply_move, plan_move, scan_pairs
        d = self._prepare()
        dst = Path(tempfile.mkdtemp()) / "out"
        plan = plan_move(scan_pairs(d), labels=["划痕"], include_unlabeled=True)
        self.assertEqual(len(plan), 2)
        report = apply_move(plan, dst)
        self.assertEqual(report["pairs"], 2)
        self.assertEqual(report["files"], 3)   # a.png+a.json+b.png
        self.assertTrue((dst / "a.json").exists())

    def test_move_conflict_skipped(self):
        import tempfile
        from xccv.engine.dataops.labelops import apply_move, plan_move, scan_pairs
        d = self._prepare()
        plan = plan_move(scan_pairs(d), labels=["划痕"], include_unlabeled=False)
        apply_move(plan, d)   # 目标=源：同名全部占用
        report = apply_move(plan, d)
        self.assertEqual(report["skipped"], 1)


class DataopsTilingTest(unittest.TestCase):
    """滑窗裁图：默认口径对齐训练切图；归属去重；可见率过滤；导出成对。"""

    def _prepare(self, w=80, h=60):
        import json
        import tempfile
        d = Path(tempfile.mkdtemp())
        _mk_png(d / "a.png", w, h)
        data = {
            "version": "5.2.1",
            "shapes": [{
                "label": "缺陷", "points": [[30, 25], [50, 25], [50, 35], [30, 35]],
                "group_id": None, "shape_type": "rectangle", "flags": {},
            }],
            "imagePath": "a.png", "imageData": None,
            "imageHeight": h, "imageWidth": w, "flags": {},
        }
        (d / "a.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return d

    def test_plan_default_aligned_and_dedup(self):
        from xccv.engine.dataops.labelops import scan_pairs
        from xccv.engine.dataops.tiling_export import plan_tiles
        d = self._prepare()
        plan = plan_tiles(scan_pairs(d))    # 默认 tile=640：小图单窗全含
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0].w, 80)
        self.assertEqual(len(plan[0].shapes), 1)   # 唯一归属

    def test_overlap_windows_attributed_once(self):
        from xccv.engine.dataops.labelops import scan_pairs
        from xccv.engine.dataops.tiling_export import plan_tiles
        d = self._prepare()
        plan = plan_tiles(scan_pairs(d), tile=48, overlap=0.5,
                          bg_tile_keep=0.0, min_visibility=0.3)
        with_obj = [i for i in plan if i.has_objects]
        self.assertGreaterEqual(len(with_obj), 1)
        total_shapes = sum(len(i.shapes) for i in plan)
        self.assertEqual(total_shapes, 1)   # 重叠区去重：同一目标只出现一次

    def test_export_writes_pairs(self):
        import json
        import tempfile
        from xccv.engine.dataops.labelops import scan_pairs
        from xccv.engine.dataops.tiling_export import export_tiles, plan_tiles
        d = self._prepare()
        dst = Path(tempfile.mkdtemp()) / "tiles"
        report = export_tiles(plan_tiles(scan_pairs(d), tile=64, bg_tile_keep=0.0), dst)
        self.assertEqual(report["images"], report["jsons"])
        pngs = sorted(dst.glob("*.png"))
        self.assertTrue(pngs)
        data = json.loads((dst / (pngs[0].stem + ".json")).read_text(encoding="utf-8"))
        self.assertEqual(data["imagePath"], pngs[0].name)


class DataopsTilingRecursiveTest(unittest.TestCase):
    """递归裁图：子目录同名源图输出名带目录前缀，互不覆盖。"""

    def test_same_stem_subdirs_unique_output_names(self):
        import tempfile
        from xccv.engine.dataops.labelops import scan_pairs
        from xccv.engine.dataops.tiling_export import plan_tiles
        root = Path(tempfile.mkdtemp())
        for sub in ("a", "b"):
            (root / sub).mkdir()
            _mk_png(root / sub / "img.png")
        plan = plan_tiles(scan_pairs(root), bg_tile_keep=1.0, root=root)
        stems = [item.stem for item in plan]
        self.assertEqual(len(stems), 2)
        self.assertEqual(len(set(stems)), 2)
        self.assertTrue(any(s.startswith("a_") for s in stems))
        self.assertTrue(any(s.startswith("b_") for s in stems))


class DataopsFramesTest(unittest.TestCase):
    """视频抽帧：步长估算 / 间隔导出 / 感知哈希去重。"""

    def _mk_video(self, frames=6):
        import tempfile
        import cv2
        import numpy as np
        path = Path(tempfile.mkdtemp()) / "v.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"XVID"),
                                 30.0, (32, 24))
        if not writer.isOpened():
            self.skipTest("VideoWriter 不可用")
        for _ in range(frames):
            writer.write(np.zeros((24, 32, 3), np.uint8))
        writer.release()
        return path

    def test_estimate_and_export(self):
        from xccv.engine.dataops.video_frames import (
            FramePlan, export_frames, plan_estimate)
        v = self._mk_video(6)
        est = plan_estimate(FramePlan(video=v, dst=v.parent, interval=3))
        self.assertTrue(est["ok"])
        self.assertEqual(est["step"], 3)
        self.assertEqual(est["estimate"], 2)
        dst = v.parent / "out"
        result = export_frames(FramePlan(video=v, dst=dst, interval=3))
        self.assertEqual(result.frames, 2)
        self.assertEqual(len(list(dst.glob("*.png"))), 2)

    def test_dedup_skips_identical_frames(self):
        from xccv.engine.dataops.video_frames import FramePlan, export_frames
        v = self._mk_video(6)   # 全零帧：内容完全一致
        dst = v.parent / "out_dedup"
        result = export_frames(FramePlan(video=v, dst=dst, interval=1, dedup=True,
                                         dedup_threshold=4))
        self.assertEqual(result.frames, 1)
        self.assertEqual(result.skipped_dup, 5)

if __name__ == '__main__':
    unittest.main()
