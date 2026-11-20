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


if __name__ == "__main__":
    unittest.main()
