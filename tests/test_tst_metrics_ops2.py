"""测试平台测试：4 个历史测试模块的合并模块。

由以下模块按域合并而成，用例逐段原样迁移、断言未改：
- test_v07_o2ext_dims
- test_v07_o3_o4_o5
- test_v07_o6_ok
- test_v07_o8_ignore
"""


# == 来源模块：test_v07_o2ext_dims ==
"""V0.7 O2 扩展（追加）：多维度阈值——算子/统计/生成器/面板回归锁。"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from xccv.engine.flow.node_base import RunContext
from xccv.engine.flow.ops.det_filters import (FilterBboxSide,
                                             FilterForeground, FilterGrayDiff,
                                             box_dims, box_image_dims)
from xccv.engine.flow.threshold_flow import (build_threshold_graph,
                                             flow_filename,
                                             save_threshold_flow)
from xccv.engine.io_utils import imwrite_unicode
from xccv.engine.metrics.batch_match import (collect_match_stats,
                                             threshold_metrics_for)
from xccv.engine.models.base import Prediction


def _scene(bright_box=(100, 100, 150, 150), bright=210, bg=30):
    """200×200 暗背景 + 亮块；返回 (BGR 图, 灰度图)。"""
    import cv2

    img = np.full((200, 200, 3), bg, dtype=np.uint8)
    x1, y1, x2, y2 = bright_box
    img[y1:y2, x1:x2] = bright
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def _pred(boxes, scores=None):
    return Prediction(labels=[0] * len(boxes),
                      scores=scores or [0.9] * len(boxes),
                      boxes=[[float(v) for v in b] for b in boxes])


class BoxDimsTest(unittest.TestCase):

    def test_geometry(self):
        d = box_dims((10, 20, 110, 70))
        self.assertAlmostEqual(d["area"], 100 * 50)
        self.assertEqual(d["short"], 50)
        self.assertEqual(d["long"], 100)

    def test_image_dims_bright_box(self):
        _img, gray = _scene()
        d = box_image_dims(gray, (100, 100, 150, 150))
        self.assertIsNotNone(d["fg_ratio"])
        self.assertGreater(d["fg_ratio"], 0.9)      # 亮块全是前景
        self.assertGreater(d["gray_diff"], 100)     # 与暗背景差悬殊

    def test_image_dims_flat_box(self):
        _img, gray = _scene()
        d = box_image_dims(gray, (10, 10, 60, 60))  # 与背景同灰度
        self.assertLess(d["fg_ratio"], 0.05)
        self.assertLess(d["gray_diff"], 3)

    def test_image_dims_box_at_border(self):
        _img, gray = _scene()
        d = box_image_dims(gray, (180, 180, 260, 260))  # 越界框不崩溃
        self.assertIsInstance(d, dict)


class FilterNodeTest(unittest.TestCase):

    def test_side_filter(self):
        pred = _pred([(0, 0, 200, 8), (0, 0, 60, 60), (0, 0, 100, 30)])
        out = FilterBboxSide().run({"result": pred},
                                   {"short_min": 20.0, "long_min": 0.0},
                                   RunContext())
        kept = out["result"].boxes
        self.assertEqual(len(kept), 2)   # 8px 高的横条被滤掉
        self.assertEqual(len(out["removed"].boxes), 1)

    def test_foreground_filter_needs_image(self):
        img, _gray = _scene()
        pred = _pred([(100, 100, 150, 150), (10, 10, 60, 60)])
        out = FilterForeground().run({"image": img, "result": pred},
                                     {"min_ratio": 0.05, "gray_thr": 20},
                                     RunContext())
        self.assertEqual(len(out["result"].boxes), 1)   # 平块被滤
        self.assertEqual(out["removed"].boxes[0], [10.0, 10.0, 60.0, 60.0])

    def test_gray_diff_filter(self):
        img, _gray = _scene()
        pred = _pred([(100, 100, 150, 150), (10, 10, 60, 60)])
        out = FilterGrayDiff().run({"image": img, "result": pred},
                                   {"min_diff": 30.0}, RunContext())
        self.assertEqual(len(out["result"].boxes), 1)

    def test_no_image_passes_through(self):
        pred = _pred([(0, 0, 50, 50)])
        out = FilterForeground().run({"result": pred},
                                     {"min_ratio": 0.99, "gray_thr": 20},
                                     RunContext())
        self.assertEqual(len(out["result"].boxes), 1)   # 缺图放行 + 告警


class DimsCollectTest(unittest.TestCase):

    def test_dims_buckets_and_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            imwrite_unicode(str(d / "a.png"), np.zeros((8, 8, 3), np.uint8))
            (d / "a.json").write_text(json.dumps({
                "shapes": [{"label": "黑点", "points": [[0, 0], [100, 100]],
                            "shape_type": "rectangle"}],
                "imagePath": "a.png", "imageData": None,
                "imageWidth": 200, "imageHeight": 200}), encoding="utf-8")
            results = [{"path": str(d / "a.png"), "detections": [
                {"label": "黑点", "score": 0.9, "bbox": [0, 0, 100, 100],
                 "_dims": {"fg_ratio": 0.8, "gray_diff": 120.0}},
                {"label": "黑点", "score": 0.4, "bbox": [150, 150, 170, 170],
                 "_dims": {"fg_ratio": 0.01, "gray_diff": 3.0}}]}]
            stats = collect_match_stats(results)
            dims = stats["dims"]
            self.assertEqual(len(dims["area"]["green"]), 1)
            self.assertEqual(dims["area"]["green"][0], 10000.0)
            self.assertEqual(dims["short"]["red_defect"], [20.0])
            self.assertEqual(dims["fg_ratio"]["green"], [0.8])
            self.assertEqual(dims["gray_diff"]["red_defect"], [3.0])
            # GT 共 1 实例（大框 TP、小框 FP）：短边阈值 25 → 只留 TP
            m = threshold_metrics_for(stats, "short", 25.0)
            self.assertEqual(m["precision"], 1.0)
            self.assertEqual(m["recall"], 1.0)
            # 阈值降到 15 → FP 也保留：精确率掉到 0.5
            m = threshold_metrics_for(stats, "short", 15.0)
            self.assertEqual(m["precision"], 0.5)

    def test_ok_image_dims_go_red_ok(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "OK"
            d.mkdir()
            imwrite_unicode(str(d / "ok0.png"), np.zeros((8, 8, 3), np.uint8))
            (d / "ok0.json").write_text(json.dumps({
                "shapes": [], "imagePath": "ok0.png", "imageData": None,
                "imageWidth": 200, "imageHeight": 200}), encoding="utf-8")
            results = [{"path": str(d / "ok0.png"), "detections": [
                {"label": "黑点", "score": 0.8, "bbox": [10, 10, 60, 60]}]}]
            stats = collect_match_stats(results)
            self.assertEqual(stats["n_ok_images"], 1)
            self.assertEqual(stats["dims"]["area"]["red_ok"], [2500.0])


class ThresholdGraphTest(unittest.TestCase):

    def test_registry_no_duplicate_with_legacy(self):
        """重名注册锁：ops 与 legacy_adapter 双模块导入不得冲突（f7g 抓到
        filter_bbox_area 重名——面积过滤沿用 legacy 薄委托节点）。"""
        from xccv.engine.flow import legacy_adapter  # noqa: F401  触发薄委托注册
        from xccv.engine.flow.ops import load_all
        from xccv.engine.flow.registry import get_node_class

        load_all()
        for n in ("filter_bbox_area", "filter_bbox_side",
                  "filter_foreground", "filter_gray_diff"):
            self.assertIsNotNone(get_node_class(n), n)

    def test_graph_structure_and_registry(self):
        from xccv.engine.flow import legacy_adapter  # noqa: F401  filter_bbox_area 在薄委托模块
        from xccv.engine.flow.executor import Executor
        from xccv.engine.flow.ops import load_all

        load_all()   # 注册全部算子（含新增 3 个）
        g = build_threshold_graph(
            "C:/m/best.pt", "斑点检测",
            {"conf": 0.35, "area": 500, "short": 12, "fg_ratio": 0.1})
        types = [n["type"] for n in g["nodes"]]
        self.assertEqual(types, ["xccv2/image_loader", "xccv2/model_detection",
                                 "xccv2/filter_bbox_area", "xccv2/filter_bbox_side",
                                 "xccv2/filter_foreground", "xccv2/result_output"])
        model = g["nodes"][1]["properties"]
        self.assertEqual(model["conf"], 0.35)
        self.assertEqual(model["model_path"], "C:/m/best.pt")
        # 链路双端一致 + 静态校验全过（结构/类型/环）
        for n in g["nodes"]:
            for inp in n["inputs"]:
                lid = inp.get("link")
                if lid is not None:
                    lk = g["links"][lid - 1]
                    self.assertEqual(lk[3], n["id"])
            for slot, out in enumerate(n["outputs"]):
                for lid in out["links"]:
                    lk = g["links"][lid - 1]
                    self.assertEqual((lk[1], lk[2]), (n["id"], slot))
        # Executor 校验用注册表键（与 flow_session._normalize_types 同语义：
        # "xccv2/<注册名>" → "<注册名>"）
        bare = {**g, "nodes": [{**n, "type": n["type"].split("/", 1)[1]}
                               for n in g["nodes"]]}
        Executor({"nodes": bare["nodes"], "links": g["links"]}).validate()

    def test_long_merges_into_side_node(self):
        g = build_threshold_graph("w.pt", "m", {"short": 8, "long": 200})
        types = [n["type"] for n in g["nodes"]]
        self.assertEqual(types.count("xccv2/filter_bbox_side"), 1)
        props = next(n["properties"] for n in g["nodes"]
                     if n["type"] == "xccv2/filter_bbox_side")
        self.assertEqual(props, {"short_min": 8.0, "long_min": 200.0})

    def test_filename_and_save_location(self):
        name = flow_filename("斑点/检测*模型")
        self.assertNotIn("/", name)
        self.assertTrue(name.split("_", 2)[2].endswith("模型.json"))
        with tempfile.TemporaryDirectory() as td:
            path = save_threshold_flow(td, "斑点检测",
                                       {"conf": 0.4, "area": 300}, "w.pt")
            self.assertEqual(path.parent.name, "flow")
            self.assertIn("斑点检测", path.name)
            g = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(g["schema"], 3)
            self.assertEqual(g["nodes"][1]["properties"]["conf"], 0.4)


class PanelSmokeTest(unittest.TestCase):
    """offscreen 面板冒烟：维度切换/拖线启用/生成信号。"""

    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def _stats(self):
        return {"gt_available": True, "gt_total": 2, "n_gt_images": 1,
                "n_ok_images": 1, "n_unmatched": 1,
                "green": [0.9], "red_defect": [0.4], "red_ok": [0.6],
                "dims": {"area": {"green": [10000.0], "red_defect": [400.0],
                                  "red_ok": [2500.0]},
                         "short": {"green": [100.0], "red_defect": [20.0],
                                   "red_ok": [50.0]},
                         "fg_ratio": {"green": [0.8], "red_defect": [0.01],
                                      "red_ok": []},
                         "gray_diff": {"green": [120.0], "red_defect": [3.0],
                                       "red_ok": []}},
                "per_image": []}

    def test_dims_flow(self):
        from xccv.testing.threshold_panel import ThresholdPanel

        p = ThresholdPanel()
        p.set_stats(self._stats(), 0.25)
        # 维度下拉：score + 有数据的几何/灰度维度（long 无数据不出现；
        # V0.7.1 前景占比按定稿从面板移除，引擎 dims 表仍保留该列）
        keys = [p.cmb_dim.itemData(i) for i in range(p.cmb_dim.count())]
        self.assertEqual(keys, ["score", "area", "short", "gray_diff"])
        self.assertNotIn("fg_ratio", keys)
        # 切到短边：默认 0 未启用
        idx = p.cmb_dim.findData("short")
        p.cmb_dim.setCurrentIndex(idx)
        self.assertFalse(p._dim_state["short"]["enabled"])
        # 拖线（模拟 setValue）→ 自动启用
        p._line.setValue(25.0)
        self.assertTrue(p._dim_state["short"]["enabled"])
        self.assertEqual(p._dim_state["short"]["thr"], 25.0)
        m = threshold_metrics_for(self._stats(), "short", 25.0)
        self.assertEqual(p.lbl_precision.text(), f"{m['precision']:.3f}")
        # 生成信号携带 conf+short
        got = {}
        p.generateFlowRequested.connect(lambda d: got.update(d))
        p._emit_generate_flow()
        self.assertEqual(got, {"score": 0.25, "short": 25.0})
        # 应用按钮恒用 conf 维（即使当前在 short 维）
        vals = []
        p.applyRequested.connect(lambda v: vals.append(v))
        p._emit_apply_conf()
        self.assertEqual(vals, [0.25])

# == 来源模块：test_v07_o3_o4_o5 ==
"""V0.7 O3/O4/O5/O12：Loss 判读、缩放 8×8 校验、灰度模式、划分环形图回归锁。"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from xccv.engine.io_utils import imread_unicode, imwrite_unicode
from xccv.engine.models.base import TrainConfig
from xccv.training.loss_diagnosis import diagnose
from xccv.training.scale_guard import scale_issue, scale_issue_text


def _hist(losses, metric=None, metric_key="map_loose"):
    out = []
    for i, l in enumerate(losses):
        e = {"epoch": i + 1, "train_loss": l}
        if metric and i < len(metric):
            e[metric_key] = metric[i]
        out.append(e)
    return out


class LossDiagnosisTest(unittest.TestCase):

    def test_underfit_detected(self):
        # 末段损失仍在明显线性下降（斜率/均值 < -0.05）
        h = _hist([2.0, 1.7, 1.45, 1.25, 1.05, 0.9, 0.78, 0.66, 0.56, 0.48])
        d = diagnose(h)
        self.assertEqual(d["kind"], "underfit")
        self.assertIn("欠拟合", d["text"])

    def test_overfit_detected(self):
        # 验证指标先升后明显回落（>10%），训练损失继续降
        losses = [1.0 - 0.05 * i for i in range(10)]     # 持续下降
        metric = [0.2, 0.4, 0.6, 0.75, 0.82, 0.85, 0.86, 0.85, 0.82, 0.74]
        d = diagnose(_hist(losses, metric))
        self.assertEqual(d["kind"], "overfit")
        self.assertIn("过拟合", d["text"])
        self.assertIn("best", d["text"])

    def test_nan_reported(self):
        d = diagnose(_hist([1.0, 0.8, 0.6, float("nan"), 0.4, 0.3, 0.2]))
        self.assertEqual(d["kind"], "nan")
        self.assertIn("NaN", d["text"])

    def test_insufficient_epochs(self):
        d = diagnose(_hist([1.0, 0.9, 0.8]))
        self.assertEqual(d["kind"], "insufficient")

    def test_normal_convergence(self):
        losses = [1.0, 0.6, 0.45, 0.4, 0.38, 0.37, 0.365, 0.36]
        metric = [0.5, 0.8, 0.85, 0.86, 0.865, 0.866, 0.866, 0.867]
        d = diagnose(_hist(losses, metric))
        self.assertEqual(d["kind"], "ok")
        self.assertIn("正常", d["text"])

    def test_empty_history(self):
        self.assertEqual(diagnose([])["kind"], "empty")
        self.assertEqual(diagnose(None)["kind"], "empty")


class ScaleGuardTest(unittest.TestCase):

    class _Sh:
        def __init__(self, label, box, w=1000, h=800):
            from xccv.engine.dataset.base import Shape

            self.shape = Shape(label=label, shape_type="rectangle",
                               points=[[box[0], box[1]], [box[2], box[3]]])
            self.width, self.height = w, h
            self.shapes = [self.shape]

    def test_stretch_scaling(self):
        # 1000×800 → 500×400（等比一半）：20px 短边 → 10px
        samples = [self._Sh("d", (0, 0, 100, 20))]
        stat = scale_issue(samples, "stretch", 500, 400)
        self.assertAlmostEqual(stat["median"], 10.0, places=1)

    def test_letterbox_scaling(self):
        # 1000×800 长边 1000 → imgsz 640：k=0.64，20px → 12.8
        samples = [self._Sh("d", (0, 0, 100, 20))]
        stat = scale_issue(samples, "letterbox", 640, 640)
        self.assertAlmostEqual(stat["median"], 12.8, places=1)

    def test_issue_text_threshold(self):
        # 95% 实例缩放后 <8px（1px 短边 ×100 张）
        samples = [self._Sh("d", (0, 0, 200, 1)) for _ in range(95)]
        samples += [self._Sh("d", (0, 0, 200, 200)) for _ in range(5)]
        stat = scale_issue(samples, "stretch", 640, 640)
        self.assertGreater(stat["ratio"], 0.5)
        detail, suggestion = scale_issue_text(stat)
        self.assertIn("<8px", detail)
        self.assertIn("切图训练", suggestion)

    def test_no_issue_below_threshold(self):
        samples = [self._Sh("d", (0, 0, 200, 100))]
        self.assertIsNone(scale_issue_text(scale_issue(samples, "stretch", 640, 640)))


class GrayModeTest(unittest.TestCase):

    def test_imread_gray_returns_3ch(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.png"
            img = np.zeros((16, 16, 3), np.uint8)
            img[:, :, 0] = 200          # 蓝通道有内容：灰度化应保留亮度
            imwrite_unicode(str(p), img)
            color = imread_unicode(str(p))
            gray = imread_unicode(str(p), color=False)
            self.assertEqual(color.shape, (16, 16, 3))
            self.assertEqual(gray.shape, (16, 16, 3))     # 复制回 3 通道
            # 灰度图三通道一致（亮度化后）
            self.assertTrue(np.array_equal(gray[:, :, 0], gray[:, :, 1]))
            self.assertTrue(np.array_equal(gray[:, :, 1], gray[:, :, 2]))

    def test_config_snapshot_carries_color_mode(self):
        from xccv.engine.models.base import _with_run_meta
        from datetime import datetime

        cfg = TrainConfig(task="detection", dataset_dir=Path("."),
                          output_dir=Path("."), model_name="yolo11n",
                          color_mode="gray")
        m = {}
        _with_run_meta(m, cfg, datetime.now(), datetime.now(), "completed")
        self.assertEqual(m["config"]["color_mode"], "gray")

    def test_rcnn_dataset_grayscale_flag(self):
        # RcnnDataset 接受 grayscale 并以同口径读图（行为锁：构造不崩+读图 3 通道）
        from xccv.engine.models.rcnn import RcnnDataset

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.png"
            imwrite_unicode(str(p), np.full((32, 32, 3), 128, np.uint8))
            from xccv.engine.dataset.base import Sample

            s = Sample(image_path=p, width=32, height=32)
            ds = RcnnDataset([s], ["d"], "detection", train=False, grayscale=True)
            img, target = ds[0]
            self.assertEqual(tuple(img.shape), (3, 32, 32))


class ManifestTest(unittest.TestCase):
    """run_manifest 序列化回归（asdict 含 Path → TypeError
    静默写入失败）。"""

    def test_manifest_serializes_with_path_fields(self):
        import json
        import tempfile
        from pathlib import Path as _P

        from xccv.engine.models.manifest import build_run_manifest

        cfg = TrainConfig(task="detection", dataset_dir=_P("."),
                          output_dir=_P("."), model_name="yolo11n")
        with tempfile.TemporaryDirectory() as td:
            fp = _P(td) / "ds"        # dataset_dir 为 Path：指纹函数可读目录
            fp.mkdir()
            cfg.dataset_dir = fp
            m = build_run_manifest(cfg, {"final": {"map50": 0.5}})
            raw = json.dumps(m, ensure_ascii=False)   # 不抛 TypeError 即过
            self.assertIn("xccv-run-manifest", raw)
            self.assertEqual(m["config"]["dataset_dir"], str(fp))


class SplitDonutTest(unittest.TestCase):
    """环形图随统计刷新（offscreen 冒烟——构造已在 ProjectWindow 内）。

    签名改为 (已标数, OK 数, 训练比例, 验证比例)——随划分参数实时重绘。
    """

    def test_update_split_smoke(self):
        if sys.platform == "linux" and not (os.environ.get("DISPLAY")
                                            or os.environ.get("WAYLAND_DISPLAY")):
            self.skipTest("headless Linux 无 DISPLAY：QtAgg 画布无法创建"
                          "（Windows CI / 桌面环境覆盖）")
        from xccv.training.charts import SplitDonutCanvas

        c = SplitDonutCanvas()
        c.update_split(80, 20, 0.8, 0.2)   # 四段
        c.update_split(80, 0, 0.8, 0.2)    # 仅已标
        c.update_split(0, 0, 0.8, 0.2)     # 空数据占位
        c.update_split(80, 20, 1.0, 0.2)   # 重叠划分（训练 100% + 验证 20%）
        self.assertTrue(True)

# == 来源模块：test_v07_o6_ok ==
"""V0.7 O6/O9：OK 负样本可见性回归锁（体检第 7 项 + 三处呈现）。"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from xccv.engine.dataset.reader import NEGATIVE_DIR_NAMES
from xccv.engine.io_utils import imwrite_unicode
from xccv.training.data_health import run_health_check, LEVEL_INFO
from xccv.training.dataset_stats import collect_stats


def _write_png(path: Path):
    """1×1 灰图（image_size 解析需要真图片，坑：假字节全被静默跳过）。"""
    imwrite_unicode(str(path), np.full((8, 8, 3), 200, dtype=np.uint8))


def _write_labelme_json(path: Path, label: str, box=(10, 10, 60, 60)):
    path.write_text(json.dumps({
        "version": "5.8.1", "flags": {},
        "shapes": [{"label": label, "points": [[box[0], box[1]],
                                               [box[2], box[3]]],
                    "shape_type": "rectangle"}],
        "imagePath": path.with_suffix(".png").name, "imageData": None,
        "imageHeight": 200, "imageWidth": 200,
    }), encoding="utf-8")


def _make_dataset(td: Path, n_defect: int = 2, n_ok: int = 2):
    d = td / "缺陷"
    d.mkdir()
    for i in range(n_defect):
        _write_png(d / f"def{i}.png")
        _write_labelme_json(d / f"def{i}.json", "黑点")
    if n_ok:
        ok = td / "OK"
        ok.mkdir()
        for i in range(n_ok):
            _write_png(ok / f"ok{i}.png")


class OkRatioHealthTest(unittest.TestCase):

    def test_ok_in_band_no_issue(self):
        with tempfile.TemporaryDirectory() as td:
            _make_dataset(Path(td))  # 2 缺陷 / 2 OK = 1:1
            rep = run_health_check("detection", Path(td))
            self.assertFalse([i for i in rep.issues
                              if i.code in ("ok_missing", "ok_ratio")])

    def test_no_ok_gives_hint(self):
        with tempfile.TemporaryDirectory() as td:
            _make_dataset(Path(td), n_ok=0)
            rep = run_health_check("detection", Path(td))
            issues = [i for i in rep.issues if i.code == "ok_missing"]
            self.assertEqual(len(issues), 1)
            self.assertEqual(issues[0].level, LEVEL_INFO)
            self.assertIn("OK", issues[0].suggestion)

    def test_ok_too_many_hinted(self):
        with tempfile.TemporaryDirectory() as td:
            _make_dataset(Path(td), n_defect=1, n_ok=10)  # 10:1
            rep = run_health_check("detection", Path(td))
            codes = [i.code for i in rep.issues]
            self.assertIn("ok_ratio", codes)

    def test_classification_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            _make_dataset(Path(td), n_ok=0)
            rep = run_health_check("classification", Path(td))
            self.assertFalse([i for i in rep.issues
                              if i.code.startswith("ok_")])


class OkVisibilityTest(unittest.TestCase):

    def test_collect_stats_counts_ok(self):
        with tempfile.TemporaryDirectory() as td:
            _make_dataset(Path(td), n_defect=2, n_ok=3)
            st = collect_stats("detection", Path(td))
            self.assertEqual(st.n_ok, 3)
            self.assertEqual(st.n_images, 2)

    def test_summarize_metrics_ok_line(self):
        from xccv.training.training_window import ProjectWindow

        text = ProjectWindow._summarize_metrics({
            "final": {"map_loose": 0.8, "neg_val_images": 5,
                      "neg_fp_per_img": 0.12, "neg_fp_img_rate": 0.1},
            "neg_train_images": 41,
        })
        self.assertIn("OK 负样本：训练 41 张 / 验证 5 张", text)
        self.assertIn("误检 0.12 框/张", text)

    def test_labelme_badge_and_menu_sources(self):
        src = Path("labelme/app.py").read_text(encoding="utf-8")
        self.assertIn("is_negative_dir_name", src)
        self.assertIn("[OK]", src)
        self.assertIn("_mark_dir_as_ok", src)
        self.assertIn("把所在文件夹设为 OK/良品目录", src)

# == 来源模块：test_v07_o8_ignore ==
"""V0.7 O8：`__ignore__` 忽略区域近似语义回归锁（训练剔除 + 评估同口径）。"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from xccv.engine.dataset.base import Shape, TaskType
from xccv.engine.dataset.ignore import (IGNORE_LABEL, filter_ignore_shapes,
                                        suppress_boxes)
from xccv.engine.dataset.prepare import prepare
from xccv.engine.dataset.reader import scan_labelme_dataset
from xccv.engine.io_utils import imwrite_unicode
from xccv.engine.metrics.batch_match import collect_match_stats, read_gt_shapes
from xccv.engine.models.yolo_data import build_labels_from_shapes
from xccv.training.data_health import LEVEL_INFO, run_health_check
from xccv.training.dataset_stats import collect_stats


def _write_png__v07_o8_ignore(path: Path):
    imwrite_unicode(str(path), np.full((8, 8, 3), 200, dtype=np.uint8))


def _write_json(path: Path, shapes: list, w: int = 200, h: int = 200):
    path.write_text(json.dumps({
        "version": "5.8.1", "flags": {}, "shapes": shapes,
        "imagePath": path.with_suffix(".png").name, "imageData": None,
        "imageHeight": h, "imageWidth": w,
    }), encoding="utf-8")


def _rect(label: str, box) -> dict:
    return {"label": label, "points": [[box[0], box[1]], [box[2], box[3]]],
            "shape_type": "rectangle"}


def _mk_sample_dataset(td: Path):
    """三张图：img1 正常 GT+ignore 区（互不相交）；img2 GT 全在 ignore 区内；
    img3 GT 与 ignore 区部分重叠（30%，保留）。"""
    d = td / "缺陷"
    d.mkdir()
    cases = {
        "img1": [_rect("黑点", (10, 10, 60, 60)), _rect(IGNORE_LABEL, (100, 100, 180, 180))],
        "img2": [_rect("黑点", (110, 110, 160, 160)), _rect(IGNORE_LABEL, (100, 100, 180, 180))],
        "img3": [_rect("黑点", (20, 20, 80, 80)), _rect(IGNORE_LABEL, (70, 70, 150, 150))],
    }
    for name, shapes in cases.items():
        _write_png__v07_o8_ignore(d / f"{name}.png")
        _write_json(d / f"{name}.json", shapes)
    return d, cases


class IgnoreHelperTest(unittest.TestCase):

    def test_fully_inside_dropped(self):
        shapes = [Shape("黑点", "rectangle", [[110, 110], [160, 160]]),
                  Shape(IGNORE_LABEL, "rectangle", [[100, 100], [180, 180]])]
        kept, n_reg, n_drop = filter_ignore_shapes(shapes)
        self.assertEqual((n_reg, n_drop), (1, 1))
        self.assertEqual(kept, [])   # GT 全包被剔除 + ignore 形状移除 → 空

    def test_outside_kept(self):
        shapes = [Shape("黑点", "rectangle", [[10, 10], [60, 60]]),
                  Shape(IGNORE_LABEL, "rectangle", [[100, 100], [180, 180]])]
        kept, n_reg, n_drop = filter_ignore_shapes(shapes)
        self.assertEqual((n_reg, n_drop), (1, 0))
        self.assertEqual(len(kept), 1)

    def test_partial_overlap_below_ratio_kept(self):
        # GT (20,20)-(80,80)=3600px²，与区域 (70,70)-(150,150) 交 10×10=100 → 2.8%
        shapes = [Shape("黑点", "rectangle", [[20, 20], [80, 80]]),
                  Shape(IGNORE_LABEL, "rectangle", [[70, 70], [150, 150]])]
        _, _, n_drop = filter_ignore_shapes(shapes)
        self.assertEqual(n_drop, 0)

    def test_partial_overlap_above_ratio_dropped(self):
        # GT (20,20)-(100,100)=6400px²，与区域 (40,40)-(100,100) 交 60×60=3600
        # → 56.25% > 50% → 剔除
        shapes = [Shape("黑点", "rectangle", [[20, 20], [100, 100]]),
                  Shape(IGNORE_LABEL, "rectangle", [[40, 40], [100, 100]])]
        _, _, n_drop = filter_ignore_shapes(shapes)
        self.assertEqual(n_drop, 1)

    def test_polygon_region_by_bbox(self):
        # 多边形区域按外接框近似（文档声明的口径）：GT 全在外接框内 → 剔除
        shapes = [Shape("黑点", "rectangle", [[110, 110], [160, 160]]),
                  Shape(IGNORE_LABEL, "polygon",
                        [[100, 100], [180, 100], [180, 180], [140, 130]])]
        kept, n_reg, n_drop = filter_ignore_shapes(shapes)
        self.assertEqual((n_reg, n_drop), (1, 1))
        self.assertEqual(kept, [])

    def test_no_regions_noop(self):
        shapes = [Shape("黑点", "rectangle", [[10, 10], [60, 60]])]
        kept, n_reg, n_drop = filter_ignore_shapes(shapes)
        self.assertEqual((n_reg, n_drop, len(kept)), (0, 0, 1))

    def test_suppress_boxes(self):
        regions = [(100, 100, 180, 180)]
        boxes = [[110, 110, 160, 160], [10, 10, 60, 60],
                 [60, 60, 110, 110]]   # 与区域仅 10×10 交（4%），保留
        labels = ["a", "b", "c"]
        kb, kl = suppress_boxes(boxes, labels, regions)
        self.assertEqual(kl, ["b", "c"])


class ReaderClassesTest(unittest.TestCase):

    def test_ignore_label_not_a_class(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write_png__v07_o8_ignore(d / "a.png")
            _write_json(d / "a.json", [_rect("黑点", (10, 10, 60, 60)),
                                       _rect(IGNORE_LABEL, (100, 100, 180, 180))])
            samples, classes, _ = scan_labelme_dataset(d)
            self.assertEqual(classes, ["黑点"])
            # ignore 形状保留在样本上（评估层/网格角标还要用）
            labels = [s.label for s in samples[0].shapes]
            self.assertIn(IGNORE_LABEL, labels)

    def test_classes_txt_ignore_filtered(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write_png__v07_o8_ignore(d / "a.png")
            _write_json(d / "a.json", [_rect("黑点", (10, 10, 60, 60))])
            (d / "classes.txt").write_text("黑点\n__ignore__\n", encoding="utf-8")
            _, classes, _ = scan_labelme_dataset(d)
            self.assertEqual(classes, ["黑点"])


class PrepareFilterTest(unittest.TestCase):

    def test_prepare_drops_overlapping_gt(self):
        with tempfile.TemporaryDirectory() as td:
            d, cases = _mk_sample_dataset(Path(td))
            data = prepare(TaskType.DETECTION, "yolo", d, val_ratio=0.0)
            self.assertEqual(data.classes, ["黑点"])
            self.assertEqual(data.ignore_regions, 3)
            self.assertEqual(data.ignore_dropped, 1)   # 仅 img2 全包被剔除
            # val_ratio=0 时稀有类兜底仍会把一张图移入验证集 → 按 train+val 全集断言
            by_stem = {Path(s.image_path).stem: s
                       for s in data.train_samples + data.val_samples}
            self.assertEqual(len(by_stem), 3)
            self.assertEqual([s.label for s in by_stem["img1"].shapes], ["黑点"])
            self.assertEqual(by_stem["img2"].shapes, [])   # 全包 → 剔光
            self.assertEqual([s.label for s in by_stem["img3"].shapes], ["黑点"])

    def test_yolo_labels_skip_ignore(self):
        shapes = [Shape("黑点", "rectangle", [[10, 10], [60, 60]]),
                  Shape(IGNORE_LABEL, "rectangle", [[100, 100], [180, 180]])]
        lb = build_labels_from_shapes(shapes, ["黑点"], "detection", 200, 200)
        self.assertEqual(lb["cls"].shape[0], 1)   # ignore 不进标签


class BatchMatchIgnoreTest(unittest.TestCase):

    def test_gt_exclude_ignore_and_overlap(self):
        with tempfile.TemporaryDirectory() as td:
            d, _ = _mk_sample_dataset(Path(td))
            boxes, labels = read_gt_shapes(d / "img2.json")
            self.assertEqual((boxes, labels), ([], []))     # 全包 GT 剔光
            boxes, labels = read_gt_shapes(d / "img1.json")
            self.assertEqual(labels, ["黑点"])               # ignore 标记不是 GT

    def test_det_in_ignore_region_not_fp(self):
        with tempfile.TemporaryDirectory() as td:
            d, _ = _mk_sample_dataset(Path(td))
            results = [{"path": str(d / "img1.png"),
                        "detections": [
                            {"label": "黑点", "score": 0.9, "bbox": [10, 10, 60, 60]},
                            {"label": "黑点", "score": 0.8, "bbox": [120, 120, 170, 170]}]}]
            stats = collect_match_stats(results)
            self.assertEqual(stats["gt_total"], 1)
            self.assertEqual(len(stats["green"]), 1)
            self.assertEqual(stats["red_defect"], [])   # 区域内检出不算 FP

    def test_ok_image_ignore_zone_dets_not_counted(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "OK"
            d.mkdir()
            _write_png__v07_o8_ignore(d / "ok0.png")
            _write_json(d / "ok0.json", [_rect(IGNORE_LABEL, (100, 100, 180, 180))])
            results = [{"path": str(d / "ok0.png"),
                        "detections": [{"label": "黑点", "score": 0.9,
                                        "bbox": [120, 120, 170, 170]},
                                       {"label": "黑点", "score": 0.7,
                                        "bbox": [10, 10, 60, 60]}]}]
            stats = collect_match_stats(results)
            self.assertEqual(stats["n_ok_images"], 1)
            self.assertEqual(stats["red_ok"], [0.7])    # 区域内检出不算 OK 误检


class VisibilityTest(unittest.TestCase):

    def test_health_check_reports_ignore(self):
        with tempfile.TemporaryDirectory() as td:
            _mk_sample_dataset(Path(td))
            rep = run_health_check("detection", Path(td))
            issues = [i for i in rep.issues if i.code == "ignore_regions"]
            self.assertEqual(len(issues), 1)
            self.assertEqual(issues[0].level, LEVEL_INFO)
            self.assertIn("3 个", issues[0].detail)

    def test_stats_count_ignore(self):
        with tempfile.TemporaryDirectory() as td:
            d, _ = _mk_sample_dataset(Path(td))
            stats = collect_stats(TaskType.DETECTION, d)
            self.assertEqual(stats.n_ignore, 3)
            self.assertEqual(stats.class_counts.get(IGNORE_LABEL), None)

if __name__ == '__main__':
    unittest.main()
