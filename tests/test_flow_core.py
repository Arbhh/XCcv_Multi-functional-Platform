"""平台测试：3 个历史测试模块的合并模块。

由以下模块按域合并而成，用例逐段原样迁移、断言未改：
- test_flow_ir
- test_flow_executor
- test_flow_registry
"""


# == 来源模块：test_flow_ir ==
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

# == 来源模块：test_flow_executor ==
"""拓扑执行引擎测试（排序/环检测/类型检查/缓存/单节点重跑/错误隔离/零拷贝）。"""

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from xccv.engine.flow.executor import Executor, FlowError
from xccv.engine.flow.ir import ImageData
from xccv.engine.flow.node_base import ParamSpec, Port, RunContext, VisionNode
from xccv.engine.flow.registry import register_node

# 本模块节点用 ex_ 前缀（注册表进程级全局，避免跨测试模块重名）

_EXEC_COUNTS: dict = {}


@register_node
class _ExSrc(VisionNode):
    NAME = "ex_src"
    CATEGORY = "图像源"
    INPUTS = ()
    OUTPUTS = (Port("image", "image", label="图像"),)

    def run(self, inputs, params, ctx):
        _EXEC_COUNTS["ex_src"] = _EXEC_COUNTS.get("ex_src", 0) + 1
        return {"image": ImageData(mat=np.zeros((8, 8), np.uint8))}


@register_node
class _ExPass(VisionNode):
    NAME = "ex_pass"
    CATEGORY = "逻辑"
    INPUTS = (Port("image", "image", label="图像"),)
    OUTPUTS = (Port("image", "image", label="图像"),)

    def run(self, inputs, params, ctx):
        return {"image": inputs["image"]}  # 直通：引用原样返回（零拷贝断言对象）


@register_node
class _ExMerge(VisionNode):
    NAME = "ex_merge"
    CATEGORY = "逻辑"
    INPUTS = (Port("a", "image", label="图像a"), Port("b", "image", label="图像b"))
    OUTPUTS = (Port("same", "result", label="结果"),)

    def run(self, inputs, params, ctx):
        return {"same": inputs["a"].mat is inputs["b"].mat}


@register_node
class _ExFail(VisionNode):
    NAME = "ex_fail"
    CATEGORY = "逻辑"
    INPUTS = (Port("image", "image", label="图像"),)
    OUTPUTS = (Port("image", "image", label="图像"),)

    def run(self, inputs, params, ctx):
        raise RuntimeError("注入失败")


@register_node
class _ExRanged(VisionNode):
    NAME = "ex_ranged"
    CATEGORY = "逻辑"
    INPUTS = ()
    OUTPUTS = (Port("image", "image", label="图像"),)
    PARAMS = {"side": ParamSpec("side", "边长", "int", 8, range=(4, 64))}

    def run(self, inputs, params, ctx):
        _EXEC_COUNTS["ex_ranged"] = _EXEC_COUNTS.get("ex_ranged", 0) + 1
        return {"image": ImageData(mat=np.zeros((params["side"],) * 2, np.uint8))}


def _node(nid, ntype, **props):
    d = {"id": nid, "type": ntype, "properties": props}
    return d


def _graph(nodes, links):
    return {"nodes": nodes, "links": links}


class TopologyTest(unittest.TestCase):
    def test_linear_chain_runs(self):
        g = _graph([_node(1, "ex_src"), _node(2, "ex_pass")],
                   [[1, 1, 0, 2, 0]])
        res = Executor(g).run()
        self.assertTrue(res[1].ok and res[2].ok)
        self.assertFalse(res[2].cached)

    def test_diamond_merge_sees_same_reference(self):
        """分支/汇合：两个分支共享源输出引用（执行器零拷贝传递）。"""
        g = _graph([_node(1, "ex_src"), _node(2, "ex_pass"),
                    _node(3, "ex_pass"), _node(4, "ex_merge")],
                   [[1, 1, 0, 2, 0], [1, 1, 0, 3, 0],
                    [2, 2, 0, 4, 0], [3, 3, 0, 4, 1]])
        res = Executor(g).run()
        self.assertTrue(res[4].ok)
        self.assertTrue(res[4].outputs["same"])  # a is b（同一引用）

    def test_cycle_detected_with_path(self):
        g = _graph([_node(1, "ex_pass"), _node(2, "ex_pass")],
                   [[1, 1, 0, 2, 0], [2, 2, 0, 1, 0]])
        with self.assertRaises(FlowError) as cm:
            Executor(g).validate()
        self.assertIn("存在环", str(cm.exception))

    def test_type_mismatch_rejected(self):
        g = _graph([_node(1, "ex_src"), _node(2, "ex_merge")],
                   [[1, 1, 0, 2, 0]])  # image → 需要两个输入且类型表不齐
        # 1 的输出 image 连到 2 的输入槽0(image) 合法，但槽1 缺输入——运行期报缺输入
        ex = Executor(g)
        res = ex.run()
        self.assertFalse(res[2].ok)
        self.assertIn("缺少必需输入", res[2].error)
        # 真·类型不匹配：number 口径节点间连线（构造 bool→image 的连线）
        g2 = _graph([_node(1, "ex_merge"), _node(2, "ex_pass")],
                    [[1, 1, 0, 2, 0]])  # bool → image 不匹配
        with self.assertRaises(FlowError, msg="端口类型不匹配"):
            Executor(g2).run()

    def test_duplicate_input_link_rejected(self):
        g = _graph([_node(1, "ex_src"), _node(2, "ex_pass")],
                   [[1, 1, 0, 2, 0], [1, 1, 0, 2, 0]])
        with self.assertRaises(FlowError, msg="多条输入连线"):
            Executor(g).validate()

    def test_unknown_type_rejected(self):
        g = _graph([_node(1, "no_such_node")], [])
        with self.assertRaises(FlowError, msg="未注册"):
            Executor(g).validate()


class CacheAndRerunTest(unittest.TestCase):
    def test_session_cache_and_rerun(self):
        """会话缓存：同图重跑命中记忆化（执行计数不变）；rerun_node O(1)。"""
        _EXEC_COUNTS.clear()
        ctx = RunContext()
        # ex_src 与 ex_ranged 均无输入端口（两棵独立单节点树）
        g = _graph([_node(1, "ex_src"), _node(2, "ex_ranged", side=16)], [])
        ex = Executor(g, ctx)
        res = ex.run()
        self.assertTrue(all(r.ok for r in res.values()))
        first = _EXEC_COUNTS["ex_src"]
        # 同一会话再跑全图：命中缓存（第二次也是 16×16，键相同）
        res2 = ex.run()
        self.assertTrue(res2[2].cached)
        self.assertEqual(_EXEC_COUNTS["ex_src"], first)  # 未重新执行
        # 单节点重跑：直接命中缓存
        r3 = ex.rerun_node(2)
        self.assertTrue(r3.cached)
        # 改参数 → params_hash 变化 → 缓存失效重新执行
        g2 = _graph([_node(1, "ex_src"), _node(2, "ex_ranged", side=32)], [])
        res3 = Executor(g2, ctx).run()
        self.assertFalse(res3[2].cached)
        self.assertEqual(_EXEC_COUNTS["ex_ranged"], 2)

    def test_prefix_run_to_node(self):
        _EXEC_COUNTS.clear()
        g = _graph([_node(1, "ex_src"), _node(2, "ex_pass"), _node(3, "ex_pass")],
                   [[1, 1, 0, 2, 0], [2, 2, 0, 3, 0]])
        ex = Executor(g)
        res = ex.run(to_node=2)
        self.assertEqual(set(res), {1, 2})  # 只执行上游前缀
        self.assertIn("ex_src", _EXEC_COUNTS)


class ErrorIsolationTest(unittest.TestCase):
    def test_failed_branch_does_not_kill_sibling(self):
        g = _graph([_node(1, "ex_src"), _node(2, "ex_fail"), _node(3, "ex_pass")],
                   [[1, 1, 0, 2, 0], [1, 1, 0, 3, 0]])
        res = Executor(g).run()
        self.assertFalse(res[2].ok)
        self.assertIn("注入失败", res[2].error)
        self.assertTrue(res[3].ok)  # 独立分支继续

    def test_downstream_of_failure_is_skipped(self):
        g = _graph([_node(1, "ex_src"), _node(2, "ex_fail"), _node(3, "ex_pass")],
                   [[1, 1, 0, 2, 0], [2, 2, 0, 3, 0]])
        res = Executor(g).run()
        self.assertFalse(res[2].ok)
        self.assertFalse(res[3].ok)
        self.assertTrue(res[3].skipped)

    def test_param_validate_error_surfaces(self):
        g = _graph([_node(1, "ex_ranged", side=999)], [])
        res = Executor(g).run()
        self.assertFalse(res[1].ok)
        self.assertIn("边长", res[1].error)

    def test_undeclared_output_rejected(self):
        @register_node
        class _ExExtra(VisionNode):
            NAME = "ex_extra"
            CATEGORY = "逻辑"
            OUTPUTS = (Port("image", "image", label="图像"),)

            def run(self, inputs, params, ctx):
                return {"image": ImageData(mat=np.zeros((2, 2), np.uint8)),
                        "oops": 1}

        res = Executor(_graph([_node(9, "ex_extra")], [])).run()
        self.assertFalse(res[9].ok)
        self.assertIn("未声明的输出端口", res[9].error)

# == 来源模块：test_flow_registry ==
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

if __name__ == '__main__':
    unittest.main()
