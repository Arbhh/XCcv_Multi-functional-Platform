"""V0.6 T4-4 回归：自定义算子运行时（复合 / 脚本 / .xcvop / 热重载 / 安检）。

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


if __name__ == "__main__":
    unittest.main()
