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


if __name__ == "__main__":
    unittest.main()
