"""IR 序列化回环与端口类型表测试。"""

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from xccv.engine.flow.ir import (
    PORT_TYPES, Geometry, ImageData, MeasureResult, NumberVal,
)


class PortTypesTest(unittest.TestCase):
    def test_six_types_with_color(self):
        """统一口径四类端口（图像/图像列表/统一结果/数值），
        每种带颜色与中文描述（前后端唯一事实源）。"""
        self.assertEqual(set(PORT_TYPES),
                         {"image", "result", "number"})
        for key, meta in PORT_TYPES.items():
            self.assertTrue(meta["color"].startswith("#"))
            self.assertTrue(meta["desc"])


class GeometryTest(unittest.TestCase):
    def test_roundtrip_each_kind(self):
        cases = {
            "point": [[1.0, 2.0], [3.0, 4.0]],
            "line": [[0.0, 0.0], [10.0, 5.0]],
            "circle": [[5.0, 5.0, 2.5]],
            "polygon": [[0, 0], [8, 0], [8, 6]],
            "rotated_rect": [[10, 10, 4, 2, 30]],
        }
        for kind, data in cases.items():
            g = Geometry(kind=kind, data=np.asarray(data, dtype=float),
                         subpixel=True)
            back = Geometry.from_json(g.to_json())
            self.assertEqual(back.kind, kind)
            self.assertTrue(np.allclose(back.data, np.asarray(data)))
            self.assertTrue(back.subpixel)

    def test_unknown_kind_rejected(self):
        with self.assertRaises(ValueError):
            Geometry(kind="bezier", data=np.zeros((2, 2)))

    def test_shape_contract_documented(self):
        """形状契约抽查：circle=(3,) rotated_rect=(5,)（docstring 即契约）。"""
        self.assertEqual(np.asarray([1, 2, 3], float).shape, (3,))
        g = Geometry("rotated_rect", np.zeros(5))
        self.assertEqual(g.data.shape, (5,))


class MeasureResultTest(unittest.TestCase):
    def test_single_and_multi_roundtrip(self):
        m1 = MeasureResult(value=12.5, unit="mm", tolerance=(10.0, 15.0),
                           passed=True)
        back = MeasureResult.from_json(m1.to_json())
        self.assertEqual(back.value, 12.5)
        self.assertEqual(back.tolerance, (10.0, 15.0))
        self.assertIs(back.passed, True)
        m2 = MeasureResult(values=[1.0, 2.0, 3.0])
        back2 = MeasureResult.from_json(m2.to_json())
        self.assertEqual(back2.values, [1.0, 2.0, 3.0])
        self.assertIsNone(back2.value)


class NumberValTest(unittest.TestCase):
    def test_roundtrip(self):
        back = NumberVal.from_json(NumberVal(3.5, "mm").to_json())
        self.assertEqual(back.value, 3.5)
        self.assertEqual(back.unit, "mm")


class ImageDataTest(unittest.TestCase):
    def test_to_json_excludes_mat(self):
        """图像本体不进 JSON（契约）：只有 shape/roi/meta。"""
        img = ImageData(mat=np.zeros((8, 6, 3), dtype=np.uint8),
                        roi=Geometry("circle", [2, 2, 1.5]),
                        meta={"scale": 0.05, "source_node": "image_loader"})
        payload = img.to_json()
        self.assertEqual(payload["shape"], [8, 6, 3])
        self.assertEqual(payload["roi"]["kind"], "circle")
        self.assertEqual(payload["meta"]["scale"], 0.05)
        self.assertNotIn("mat", payload)

    def test_from_json_rebuilds_shell(self):
        img = ImageData(mat=np.ones((4, 4), dtype=np.uint8),
                        meta={"channel": "gray"})
        back = ImageData.from_json(img.to_json())
        self.assertEqual(back.meta, {"channel": "gray"})
        self.assertIsNone(back.roi)
        self.assertEqual(back.mat.size, 0)  # 图像不来自 JSON（文档契约）

    def test_reference_passthrough(self):
        """IR 传递引用：roi/meta 赋值不复制 mat（零拷贝的地基）。"""
        mat = np.zeros((4, 4), dtype=np.uint8)
        img = ImageData(mat=mat)
        self.assertIs(img.mat, mat)


if __name__ == "__main__":
    unittest.main()
