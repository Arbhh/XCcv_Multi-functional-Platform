"""测试：AI 桥接算子（三个模型节点 / roi_batch_classify /
detection_to_measure / geometry_to_affine）+ DetectionPlus 包装。

含真实小模型端到端：合成分类数据 1 轮训练 → best.pt 产出 →
经分类模型节点 / roi_batch_classify 的 registry.load_model 链路加载推理
（沿用 test_augment_batch9 的实测 E2E 模式）。
"""

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

import xccv.engine.flow.ops.ai_bridge  # noqa: F401  触发注册
import xccv.engine.flow.ops.listops  # noqa: F401  逐图推理/结果合并注册
import xccv.engine.flow.ops.preprocess  # noqa: F401
import xccv.engine.flow.ops.source  # noqa: F401
from xccv.engine.flow.executor import Executor
from xccv.engine.flow.ir import Geometry, ImageData, MeasureResult
from xccv.engine.flow.node_base import RunContext
from xccv.engine.flow.ops.ai_bridge import DetectionPlus
from xccv.engine.flow.registry import get_node_class, list_nodes
from xccv.engine.models.base import Prediction
from tests.synthetic import make_classification_dataset

CTX = RunContext


def node(name: str):
    return get_node_class(name)()


def _img():
    return ImageData(mat=np.zeros((80, 100), np.uint8))


def _pred():
    return Prediction(labels=[0, 1], scores=[0.9, 0.7],
                      boxes=[[10, 10, 30, 30], [50, 40, 70, 60]])


class DetectionPlusTest(unittest.TestCase):
    def test_getattr_delegates_to_prediction(self):
        """组合式包装对下游透明：按 Prediction 消费无需感知。"""
        dp = DetectionPlus(pred=_pred(), extra={"k": 1})
        self.assertEqual(len(dp.boxes), 2)
        self.assertEqual(dp.labels, [0, 1])
        self.assertEqual(dp.plus_extra["k"], 1)

    def test_filters_consume_detection_plus(self):
        from xccv.engine.flow.legacy_adapter import chain_to_xcv_graph

        dp = DetectionPlus(pred=_pred(), extra={"roi_classifications": []})
        ctx = CTX()
        ctx.set("inject_detection", dp)
        ex = Executor(chain_to_xcv_graph(
            {"nodes": [{"op": "confidence", "params": {"min": 0.8}}]}), ctx)
        res = ex.run()
        got = res[max(res)].outputs["result"]
        self.assertEqual(len(got.boxes), 1)  # 0.7 被滤掉


class AiBridgeRegistryTest(unittest.TestCase):
    def test_ai_category_nodes(self):
        names = {n["name"] for n in list_nodes(category="AI")}
        # detection_to_measure/geometry_to_affine 归「测量」，
        # merge_results 归「通用」（端口统一+分组归位）
        self.assertEqual(names, {"model_detection", "model_instance_seg",
                                 "model_classification", "roi_batch_classify",
                                 "model_on_list"})

    def test_model_requires_path(self):
        with self.assertRaises(ValueError):
            node("model_detection").run({"image": _img()},
                                        {"model_path": "", "conf": 0.25,
                                         "iou": 0.45, "device": "cpu"}, CTX())

    def test_roi_classify_requires_model(self):
        with self.assertRaises(ValueError):
            node("roi_batch_classify").run({"image": _img(), "result": _pred()},
                                           {"model_path": "", "conf": 0.5,
                                            "margin": 0.1, "device": "cpu"}, CTX())


class DetectionToMeasureTest(unittest.TestCase):
    def test_area_with_tolerance(self):
        r = node("detection_to_measure").run(
            {"result": _pred()},
            {"metric": "area", "tol_min": 100, "tol_max": 1000}, CTX())
        m = r["result"]
        self.assertEqual(m.values, [400.0, 400.0])
        self.assertTrue(m.passed)

    def test_score_metric_without_tolerance(self):
        r = node("detection_to_measure").run(
            {"result": _pred()},
            {"metric": "score", "tol_min": 0, "tol_max": 0}, CTX())
        self.assertIsNone(r["result"].passed)
        self.assertEqual(r["count"].value, 2.0)

    def test_tolerance_fail(self):
        r = node("detection_to_measure").run(
            {"result": _pred()},
            {"metric": "width", "tol_min": 0, "tol_max": 10}, CTX())
        self.assertFalse(r["result"].passed)  # 宽 20 > 10

    def test_consumes_detection_plus(self):
        dp = DetectionPlus(pred=_pred())
        r = node("detection_to_measure").run(
            {"result": dp},
            {"metric": "area", "tol_min": 0, "tol_max": 0}, CTX())
        self.assertEqual(len(r["result"].values), 2)


class GeometryToAffineTest(unittest.TestCase):
    def test_rotate_level_line(self):
        """水平线 → 单位阵；绕图中心（无旋转量时矩阵为单位阵）。"""
        line = Geometry(kind="line", data=np.array([[10.0, 40.0], [90.0, 40.0]]))
        r = node("geometry_to_affine").run(
            {"result": line, "image": ImageData(mat=np.zeros((80, 100), np.uint8))},
            {"mode": "rotate_level"}, CTX())
        vals = r["result"].values
        self.assertEqual(len(vals), 6)
        self.assertAlmostEqual(r["angle"].value, 0.0, places=3)
        np.testing.assert_allclose(vals, [1, 0, 0, 0, 1, 0], atol=1e-6)

    def test_rotate_level_tilted_line(self):
        """斜线角度 → 矩阵把图转平：输出经 affine_transform 消费。"""
        line = Geometry(kind="line", data=np.array([[0.0, 0.0], [100.0, 100.0]]),
                        subpixel=True)
        r = node("geometry_to_affine").run(
            {"result": line, "image": ImageData(mat=np.zeros((200, 200), np.uint8))},
            {"mode": "rotate_level"}, CTX())
        self.assertAlmostEqual(r["angle"].value, 45.0, places=2)

    def test_translate_point_to_center(self):
        pt = Geometry(kind="point", data=np.array([[20.0, 10.0]]))
        r = node("geometry_to_affine").run(
            {"result": pt, "image": ImageData(mat=np.zeros((80, 100), np.uint8))},
            {"mode": "translate_point"}, CTX())
        vals = r["result"].values
        self.assertAlmostEqual(vals[2], 30.0)  # 100/2 - 20
        self.assertAlmostEqual(vals[5], 30.0)  # 80/2 - 10

    def test_wrong_geometry_kind_raises(self):
        pt = Geometry(kind="point", data=np.array([[1.0, 2.0]]))
        with self.assertRaises(ValueError):
            node("geometry_to_affine").run({"result": pt}, {"mode": "rotate_level"},
                                           CTX())

    def test_pipeline_with_affine_transform(self):
        """定位归正链：geometry_to_affine → affine_transform.matrix_in。"""
        from xccv.engine.flow.ops.preprocess import AffineTransform

        line = Geometry(kind="line", data=np.array([[0.0, 10.0], [100.0, 10.0]]))
        aff = AffineTransform()
        out = aff.run(
            {"image": ImageData(mat=np.full((40, 40, 3), 200, np.uint8)),
             "matrix_in": node("geometry_to_affine").run(
                 {"result": line}, {"mode": "rotate_level"}, CTX())["result"]},
            {"matrix": "broken,text", "interp": "linear"}, CTX())
        self.assertEqual(out["image"].mat.shape, (40, 40, 3))  # 矩阵端口优先于坏文本


class RealModelE2ETest(unittest.TestCase):
    """真实小模型端到端：1 轮训练 → best.pt → 桥接节点加载推理。"""

    @classmethod
    def setUpClass(cls):
        from xccv.engine.dataset.base import TaskType
        from xccv.engine.models import create_model
        from xccv.engine.models.base import TrainConfig

        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        make_classification_dataset(root / "ds", per_class=4, size=(32, 32))
        model = create_model("resnet18", TaskType.CLASSIFICATION, 2,
                             device="cpu", use_pretrained=False)
        cfg = TrainConfig(
            task=TaskType.CLASSIFICATION, dataset_dir=root / "ds",
            output_dir=root / "run", model_name="resnet18",
            epochs=1, batch_size=2, workers=0, seed=0,
            image_width=32, image_height=32,
            augment_ops={"resize": {"width": 32, "height": 32}},
        )
        model.train(cfg)
        weights = list((root / "run").glob("best.*"))
        cls.weight_path = str(weights[0])
        cls.model_classes = model.classes

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_task_mismatch_rejected(self):
        """Q5 拆分后节点校验任务类型：分类权重放进检测节点要被拒绝。"""
        with tempfile.TemporaryDirectory() as tmp:
            weight = next(iter(Path(self.weight_path).parent.glob("best.*")))
            with self.assertRaises(ValueError) as cm:
                node("model_detection").run(
                    {"image": _img()},
                    {"model_path": str(weight), "conf": 0.25, "iou": 0.45,
                     "device": "cpu"}, CTX())
            self.assertIn("目标检测", str(cm.exception))

    def test_classification_model_end_to_end(self):
        nd = node("model_classification")
        ctx = CTX()
        r = nd.run({"image": ImageData(mat=np.full((64, 64, 3), 127, np.uint8))},
                   {"model_path": self.weight_path, "conf": 0.25, "iou": 0.45,
                    "device": "cpu"}, ctx)
        self.assertIsInstance(r["result"], Prediction)
        self.assertIn("model_classes", ctx._shared)  # 类别名表已写入会话

    def test_model_session_cache(self):
        ctx = CTX()
        kw = {"model_path": self.weight_path, "conf": 0.25, "iou": 0.45,
              "device": "cpu"}
        img = {"image": ImageData(mat=np.full((64, 64, 3), 127, np.uint8))}
        node("model_classification").run(img, kw, ctx)
        node("model_classification").run(img, kw, ctx)
        self.assertEqual(len([k for k in ctx._shared if k.startswith("ai_model:")]), 1)

    def test_roi_batch_classify_end_to_end(self):
        """检出框 → 裁剪 → 分类复判 → DetectionPlus.extra。"""
        scene = np.full((80, 120, 3), 30, np.uint8)
        pred = Prediction(labels=[0, 0], scores=[0.9, 0.9],
                          boxes=[[10, 10, 40, 40], [60, 30, 100, 70]])
        r = node("roi_batch_classify").run(
            {"image": ImageData(mat=scene), "result": pred},
            {"model_path": self.weight_path, "conf": 0.1, "margin": 0.1,
             "device": "cpu"}, CTX())
        dp = r["result"]
        self.assertIsInstance(dp, DetectionPlus)
        rois = dp.extra["roi_classifications"]
        self.assertEqual(len(rois), 2)
        self.assertEqual([x["index"] for x in rois], [0, 1])
        self.assertTrue(all(set(x) >= {"label", "label_id", "score"} for x in rois))

    def test_roi_classify_skips_degenerate_box(self):
        scene = np.full((40, 40, 3), 30, np.uint8)
        pred = Prediction(labels=[0], scores=[0.9], boxes=[[45, 45, 50, 50]])
        ctx = CTX()
        r = node("roi_batch_classify").run(
            {"image": ImageData(mat=scene), "result": pred},
            {"model_path": self.weight_path, "conf": 0.1, "margin": 0.1,
             "device": "cpu"}, ctx)
        self.assertEqual(r["result"].extra["roi_classifications"], [])
        self.assertTrue(any("退化" in w for w in ctx.warnings))


if __name__ == "__main__":
    unittest.main()

