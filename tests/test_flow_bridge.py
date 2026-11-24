"""平台测试：3 个历史测试模块的合并模块。

由以下模块按域合并而成，用例逐段原样迁移、断言未改：
- test_flow_ai_bridge
- test_v06_external_ops
- test_v06_custom_ops
"""


# == 来源模块：test_flow_ai_bridge ==
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

# == 来源模块：test_v06_external_ops ==
"""外部算子回归：插件后端接口与内置 OpenCV 展示后端。

覆盖：内置 mock 引擎的注册与可用性、算子语义映射（图像/数值输出）、
未知算子的演示回退、坏 JSON 报错、不可用后端的节点报错路径、以及
自定义后端经 register_engine 注册后的端到端调用。
"""

import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from xccv.engine.flow.ir import ImageData
from xccv.engine.flow.node_base import RunContext
from xccv.engine.flow.registry import get_node_class
from xccv.modeltools.flow_session import ensure_ops_registered

ensure_ops_registered()


def _run(**params):
    cls = get_node_class("external_op")
    full = {k: s.default for k, s in cls.PARAMS.items()}
    full.update(params)
    errs = cls().validate(full)
    assert not errs, errs
    ctx = RunContext()
    img = ImageData(mat=np.full((4, 6), 10, np.uint8))
    return cls().run({"image": img}, full, ctx), ctx


class RegistryTest(unittest.TestCase):
    def test_builtin_mock_engine_available(self):
        from xccv.engine.flow.external import list_engines

        engines = {e["name"]: e for e in list_engines()}
        self.assertIn("mock", engines)
        ok, _msg = engines["mock"]["available"], engines["mock"]["message"]
        self.assertTrue(ok)

    def test_custom_engine_registration(self):
        from xccv.engine.flow.external import (
            ExternalEngine, ExternalResult, get_engine, register_engine,
        )

        class _Mine(ExternalEngine):
            name, title = "mine_test", "测试引擎"

            def available(self):
                return True, "ok"

            def run(self, op, image, params):
                return ExternalResult(values=[1.0])

        register_engine(_Mine())
        self.assertEqual(get_engine("mine_test").run("x", None, {}).values, [1.0])


class MockEngineTest(unittest.TestCase):
    """内置 OpenCV 展示后端的算子语义与节点集成。"""

    def test_invert_image(self):
        out, _ = _run(engine="mock", op="invert_image")
        self.assertEqual(int(out["image"].mat[0, 0]), 245)
        self.assertEqual(out["image"].meta["external"], "mock")

    def test_numeric_output_and_passthrough_image(self):
        out, _ = _run(engine="mock", op="threshold", args="[128]")
        self.assertEqual(out["value"].value, 128.0)
        self.assertEqual(out["image"].mat.shape, (4, 6))  # 二值图输出

        out, _ = _run(engine="mock", op="count_regions", args="[]")
        self.assertIsInstance(out["value"].value, float)

    def test_mean_image_blur(self):
        out, _ = _run(engine="mock", op="mean_image", args="[3, 3]")
        self.assertEqual(out["image"].mat.shape, (4, 6))
        # 均值滤波不改变均匀灰度图的灰度值
        self.assertEqual(int(out["image"].mat[0, 0]), 10)

    def test_unknown_operator_demo_fallback(self):
        out, _ = _run(engine="mock", op="no_such_op")
        self.assertEqual(out["image"].mat.shape, (4, 6))  # 演示平滑输出
        self.assertIsInstance(out["value"].value, float)

    def test_procedure_file_demo_output(self):
        out, _ = _run(engine="mock", op="D:/demo/measure_part.hdvp",
                      args='{"Thresh": 128}', image_param="ImageIn",
                      outputs="Area", image_output="ImageOut")
        self.assertEqual(out["image"].mat.shape, (4, 6))
        self.assertIsInstance(out["value"].value, float)

    def test_bad_json_reported(self):
        with self.assertRaises(ValueError) as ctx:
            _run(engine="mock", op="invert_image", args="[1,")
        self.assertIn("JSON", str(ctx.exception))

    def test_unavailable_backend_node_error(self):
        from xccv.engine.flow.external import (
            ExternalEngine, register_engine,
        )

        class _Down(ExternalEngine):
            name, title = "down_test", "不可用引擎"

            def available(self):
                return False, "示例后端未安装（安装指引占位）"

            def run(self, op, image, params):
                raise AssertionError("不可用后端不应被调用")

        register_engine(_Down())
        try:
            cls = get_node_class("external_op")
            full = {k: s.default for k, s in cls.PARAMS.items()}
            full["engine"] = "down_test"  # 绕过 enum 校验，直测 run 的不可用路径
            with self.assertRaises(ValueError) as ctx:
                cls().run({"image": ImageData(mat=np.zeros((4, 6), np.uint8))},
                          full, RunContext())
            self.assertIn("外部算子不可用", str(ctx.exception))
        finally:
            from xccv.engine.flow import external as ext

            ext._ENGINES.pop("down_test", None)


class I18nKeysTest(unittest.TestCase):
    def test_node_title_and_tooltips_in_table(self):
        from xccv.i18n_en import TABLE

        cls = get_node_class("external_op")
        self.assertIn(cls.TITLE, TABLE)
        for spec in cls.PARAMS.values():
            if spec.tooltip:
                self.assertIn(spec.tooltip, TABLE)

# == 来源模块：test_v06_custom_ops ==
"""自定义算子运行时（复合 / 脚本 / .xcvop / 热重载 / 安检）。

全部用临时算子目录（operators_dir(root)），不触碰真实 platform_data。
"""

import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

from xccv.engine.flow import custom_ops as co
from xccv.engine.flow import ops
from xccv.engine.flow.executor import Executor
from xccv.engine.flow.ir import ImageData
from xccv.engine.flow.node_base import RunContext
from xccv.engine.flow.registry import get_node_class, list_nodes

ops.load_all()


def _script(op_id="blur_s", code=None, **over):
    d = co.new_op_template("script")
    d["id"] = op_id
    if code is not None:
        d["script"]["code"] = code
    d.update(over)
    return d


def _composite(op_id="blur_c"):
    d = co.new_op_template("composite")
    d["id"] = op_id
    return d


class _TmpRootCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "算子"
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        co.load_custom_ops(self.root / "__none__")   # 注销本用例注册的 custom/ 节点
        self._tmp.cleanup()

    def _img(self):
        rng = np.random.default_rng(0)
        return ImageData(mat=rng.integers(0, 255, size=(40, 50, 3), dtype=np.uint8))


class ScriptOpTest(_TmpRootCase):
    def test_save_registers_and_runs(self):
        co.save_op(_script(), self.root)
        cls = get_node_class("custom/blur_s")
        self.assertIsNotNone(cls)
        self.assertEqual(cls.CATEGORY, "自定义")
        img = self._img()
        out = cls().run({"image": img}, {"k": 5}, RunContext())["image"]
        np.testing.assert_array_equal(out.mat, cv2.GaussianBlur(img.mat, (5, 5), 0))
        names = {n["name"] for n in list_nodes(category="自定义")}
        self.assertIn("custom/blur_s", names)

    def test_hot_reload_after_edit(self):
        co.save_op(_script(), self.root)
        code = ("def run(inputs, params, ctx):\n"
                "    img = inputs['image']\n"
                "    return {'image': ImageData(mat=255 - img.mat, meta=dict(img.meta))}\n")
        co.save_op(_script(code=code), self.root)          # 保存即热重载
        img = self._img()
        out = get_node_class("custom/blur_s")().run({"image": img}, {"k": 5}, RunContext())
        np.testing.assert_array_equal(out["image"].mat, 255 - img.mat)

    def test_executor_graph_with_custom_node(self):
        from xccv.engine.io_utils import imwrite_unicode

        co.save_op(_script(), self.root)
        p = Path(self._tmp.name) / "图.png"
        imwrite_unicode(p, self._img().mat)
        graph = {"nodes": [{"id": 1, "type": "image_loader", "properties": {"path": str(p)}},
                           {"id": 2, "type": "custom/blur_s", "properties": {"k": 3}}],
                 "links": [[1, 1, 0, 2, 0]]}
        res = Executor(graph, RunContext()).run()
        self.assertTrue(res[2].ok, res[2].error)

    def test_script_rejections(self):
        cases = {
            "import os\ndef run(inputs, params, ctx):\n    return {}\n": "不允许 import os",
            "def run(inputs, params, ctx):\n    open('x')\n    return {}\n": "禁止使用 open",
            "def run(inputs, params, ctx):\n    return eval('1')\n": "禁止使用 eval",
            "def run(inputs, params, ctx):\n    return ().__class__\n": "双下划线",
            "print(1)\ndef run(inputs, params, ctx):\n    return {}\n": "顶层只允许",
            "def helper(x):\n    return x\n": "def run",
            "def run(inputs, params):\n    return {}\n": "恰好 3 个参数",
            "def run(inputs, params, ctx)\n    return {}\n": "语法错误",
            "from subprocess import run as r\ndef run(inputs, params, ctx):\n    return {}\n":
                "不允许 from subprocess",
        }
        for code, expect in cases.items():
            errs = co.check_script(code)
            self.assertTrue(any(expect in e for e in errs), (code, errs))
            with self.assertRaises(ValueError):
                co.save_op(_script(code=code), self.root)

    def test_allowed_imports_and_restricted_builtins(self):
        code = ("import math\nimport numpy as np\n"
                "def run(inputs, params, ctx):\n"
                "    v = math.sqrt(16) + float(np.mean([1, 3]))\n"
                "    return {'image': inputs['image']}\n")
        self.assertEqual(co.check_script(code), [])
        fn = co._compile_script("t", code)
        self.assertNotIn("open", fn.__globals__["__builtins__"])
        with self.assertRaises(ImportError):
            fn.__globals__["__builtins__"]["__import__"]("os")


class CompositeOpTest(_TmpRootCase):
    def test_param_injection_equals_builtin(self):
        co.save_op(_composite(), self.root)
        img = self._img()
        out = get_node_class("custom/blur_c")().run({"image": img}, {"k": 7}, RunContext())
        np.testing.assert_array_equal(out["image"].mat, cv2.GaussianBlur(img.mat, (7, 7), 0))

    def test_text_interpolation(self):
        self.assertEqual(co._subst("a_${k}_b", {"k": 3}), "a_3_b")
        self.assertEqual(co._subst("${k}", {"k": 3}), 3)            # 整串保留类型

    def test_composite_validation_errors(self):
        d = _composite()
        d["composite"]["graph"]["nodes"][0]["properties"]["sigma"] = "${missing}"
        self.assertTrue(any("未定义参数" in e for e in co.validate_op_def(d)))
        d2 = _composite()
        d2["composite"]["graph"]["nodes"][0]["type"] = "xccv2/custom/blur_c"
        self.assertTrue(any("不允许嵌套" in e for e in co.validate_op_def(d2)))
        d3 = _composite()
        d3["inputs"][0]["type"] = "number"   # detection 类型已并入 result；用 number 构造不一致
        self.assertTrue(any("类型不一致" in e for e in co.validate_op_def(d3)))
        d4 = _composite()
        d4["composite"]["output_map"] = {}
        self.assertTrue(any("未映射" in e for e in co.validate_op_def(d4)))

    def test_child_context_isolation(self):
        parent = RunContext()
        child = parent.child()
        rev = parent.revision
        child.set("model_classes", ["a"])
        self.assertEqual(parent.get("model_classes"), ["a"])        # 共享键共用
        self.assertGreater(parent.revision, rev)                    # 主图缓存同步失效
        child.cache_set("k", 1)
        self.assertIsNone(parent.cache_get("k"))                    # 计算缓存独立


class DefinitionValidationTest(unittest.TestCase):
    def test_bad_definitions(self):
        base = _script()
        cases = [
            (dict(base, id="Bad-Id"), "id 须以小写字母"),
            (dict(base, kind="plugin"), "kind 须为"),
            (dict(base, outputs=[]), "至少需要一个输出"),
            (dict(base, inputs=[{"name": "x", "type": "video"}]), "类型「video」非法"),
            (dict(base, params={"k": {"label": "k", "type": "int", "default": 99,
                                      "range": [1, 31]}}), "默认值非法"),
            (dict(base, schema=2), "schema 应为"),
        ]
        for d, expect in cases:
            errs = co.validate_op_def(d)
            self.assertTrue(any(expect in e for e in errs), (expect, errs))


class XcvopAndReloadTest(_TmpRootCase):
    def test_export_import_roundtrip(self):
        co.save_op(_script(), self.root)
        pkg = co.export_xcvop("blur_s", Path(self._tmp.name) / "导出" / "blur", self.root)
        self.assertEqual(pkg.suffix, ".xcvop")
        other = Path(self._tmp.name) / "另一台"
        op_id = co.import_xcvop(pkg, other)
        self.assertEqual(op_id, "blur_s")
        self.assertTrue((other / "blur_s.op.json").is_file())
        with self.assertRaises(FileExistsError):
            co.import_xcvop(pkg, other, overwrite=False)

    def test_malicious_package_rejected(self):
        bad = Path(self._tmp.name) / "bad.xcvop"
        with zipfile.ZipFile(bad, "w") as z:
            z.writestr("op.json", json.dumps(_script()))
            z.writestr("../evil.py", "print('x')")
        with self.assertRaises(ValueError):
            co.import_xcvop(bad, self.root)
        big = Path(self._tmp.name) / "big.xcvop"
        d = _script(description="x" * (3 * 1024 * 1024))
        with zipfile.ZipFile(big, "w") as z:
            z.writestr("op.json", json.dumps(d))
        with self.assertRaises(ValueError):
            co.import_xcvop(big, self.root)

    def test_reload_reports_bad_files_and_delete(self):
        co.save_op(_script(), self.root)
        (self.root / "broken.op.json").write_text("{not json", encoding="utf-8")
        (self.root / "invalid.op.json").write_text(
            json.dumps(dict(_script(), id="x")), encoding="utf-8")   # id 太短
        loaded, errors = co.load_custom_ops(self.root)
        self.assertEqual(loaded, ["custom/blur_s"])
        self.assertEqual(len(errors), 2)
        self.assertTrue(co.delete_op("blur_s", self.root))
        self.assertIsNone(get_node_class("custom/blur_s"))

    def test_node_library_payload_includes_custom(self):
        from xccv.modeltools.flow_session import node_library_payload

        co.save_op(_composite(), self.root)
        names = {n["name"] for n in node_library_payload()["nodes"]}
        self.assertIn("custom/blur_c", names)

if __name__ == '__main__':
    unittest.main()
