"""节点注册表测试（注册唯一性 / 分类枚举 / schema 默认值合法性 / 无 Qt 导入）。"""

import os
import subprocess
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from xccv.engine.flow.ir import Geometry, ImageData
from xccv.engine.flow.node_base import ParamSpec, Port, VisionNode
from xccv.engine.flow.registry import (
    all_categories, get_node_class, list_nodes, register_node,
)

# 本模块注册的测试节点用 rg_ 前缀，避免与其他测试模块的全局注册冲突


@register_node
class _RgSource(VisionNode):
    NAME = "rg_source"
    CATEGORY = "图像源"
    INPUTS = ()
    OUTPUTS = (Port("image", "image", label="图像"),)
    PARAMS = {"gray": ParamSpec("gray", "灰度", "bool", False),
              "size": ParamSpec("size", "边长", "int", 32, range=(8, 512))}

    def run(self, inputs, params, ctx):
        n = int(params["size"])
        mat = np.full((n, n), 128, np.uint8) if params["gray"] \
            else np.zeros((n, n, 3), np.uint8)
        return {"image": ImageData(mat=mat)}


class RegistryTest(unittest.TestCase):
    def test_register_and_get(self):
        cls = get_node_class("rg_source")
        self.assertIs(cls, _RgSource)

    def test_duplicate_name_rejected(self):
        with self.assertRaises(ValueError, msg="注册名重复"):
            @register_node
            class _Dup(VisionNode):
                NAME = "rg_source"
                CATEGORY = "逻辑"

                def run(self, inputs, params, ctx):
                    return {}

    def test_missing_category_rejected(self):
        with self.assertRaises(ValueError):
            @register_node
            class _NoCat(VisionNode):
                NAME = "rg_nocat"

                def run(self, inputs, params, ctx):
                    return {}

    def test_illegal_default_rejected(self):
        """注册时校验参数默认值：默认值必须能通过自身 schema。"""
        with self.assertRaises(ValueError, msg="默认值非法"):
            @register_node
            class _BadDefault(VisionNode):
                NAME = "rg_bad_default2"
                CATEGORY = "逻辑"
                PARAMS = {"k": ParamSpec("k", "系数", "float", 5.0, range=(0.0, 1.0))}

                def run(self, inputs, params, ctx):
                    return {}

    def test_list_nodes_schema_shape(self):
        items = list_nodes("图像源")
        names = [it["name"] for it in items]
        self.assertIn("rg_source", names)
        it = next(it for it in items if it["name"] == "rg_source")
        self.assertEqual(it["category"], "图像源")
        self.assertEqual(it["outputs"][0]["name"], "image")   # 端口带 label
        self.assertEqual(it["params"]["size"]["range"], [8, 512])
        self.assertEqual(it["params"]["gray"]["default"], False)

    def test_all_categories_contains_registered(self):
        cats = all_categories()
        self.assertIn("图像源", cats)

    def test_unknown_get_returns_none(self):
        self.assertIsNone(get_node_class("no_such_node_xyz"))


class NoQtImportTest(unittest.TestCase):
    def test_engine_flow_imports_no_qt(self):
        """xccv.engine.flow 包禁止引入 Qt（子进程全新解释器验证）。"""
        repo = str(__import__("pathlib").Path(__file__).resolve().parents[1])
        code = ("import sys; import xccv.engine.flow; "
                "bad=[m for m in sys.modules if m.startswith('PyQt5')]; "
                "assert not bad, bad; print('NOQT-OK')")
        r = subprocess.run([sys.executable, "-c", code], cwd=repo,
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("NOQT-OK", r.stdout)


if __name__ == "__main__":
    unittest.main()

