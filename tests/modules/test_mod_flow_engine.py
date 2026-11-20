"""模块行为锁：流程引擎（算子注册表 / 执行器 / IR）。

行为级口径：load_all() 导入即注册、执行器拓扑语义、IR 往返。
自定义节点用 modfx_ 前缀（注册表进程级全局，避免跨模块重名）。
"""

import unittest

import numpy as np

from xccv.engine.flow.executor import Executor, FlowError
from xccv.engine.flow.ir import ImageData, PORT_TYPES, Geometry
from xccv.engine.flow.node_base import Port, VisionNode
from xccv.engine.flow.registry import register_node, registered_names

from xccv.engine.flow.ops import load_all

load_all()   # 导入即注册（幂等）；画布节点库/工作台同款入口


def _node(nid, ntype, **props):
    return {"id": nid, "type": ntype, "properties": props}


@register_node
class _ModFxSrc(VisionNode):
    NAME = "modfx_src"
    CATEGORY = "图像源"
    INPUTS = ()
    OUTPUTS = (Port("image", "image", label="图像"),)

    def run(self, inputs, params, ctx):
        return {"image": ImageData(mat=np.zeros((8, 8), np.uint8))}


@register_node
class _ModFxPass(VisionNode):
    NAME = "modfx_pass"
    CATEGORY = "逻辑"
    INPUTS = (Port("image", "image", label="图像"),)
    OUTPUTS = (Port("image", "image", label="图像"),)

    def run(self, inputs, params, ctx):
        return {"image": inputs["image"]}


class BuiltinOpsRegistryTest(unittest.TestCase):
    """内置算子注册表（README 对账脚本的数据源，P-3 联动）。"""

    def test_builtin_ops_registered(self):
        names = set(registered_names())
        self.assertGreaterEqual(len(names), 50, "内置算子数不应缩水")
        for core in ("image_loader", "threshold", "blob_analysis",
                     "model_detection", "model_classification",
                     "result_output", "crop_by_detection"):
            self.assertIn(core, names, f"核心算子缺失：{core}")

    def test_duplicate_name_rejected(self):
        with self.assertRaises(ValueError):
            @register_node
            class _Dup(VisionNode):
                NAME = "modfx_src"   # 与上方重复
                CATEGORY = "逻辑"
                INPUTS = ()
                OUTPUTS = ()

                def run(self, inputs, params, ctx):
                    return {}


class ExecutorBehaviorTest(unittest.TestCase):
    """拓扑执行：线性链通过 / 环拒绝 / 未注册节点拒绝。"""

    def test_linear_chain_runs(self):
        g = {"nodes": [_node(1, "modfx_src"), _node(2, "modfx_pass")],
             "links": [[1, 1, 0, 2, 0]]}
        res = Executor(g).run()
        self.assertTrue(res[1].ok and res[2].ok)

    def test_cycle_and_unknown_rejected(self):
        g_cycle = {"nodes": [_node(1, "modfx_pass"), _node(2, "modfx_pass")],
                   "links": [[1, 1, 0, 2, 0], [2, 2, 0, 1, 0]]}
        with self.assertRaises(FlowError) as cm:
            Executor(g_cycle).validate()
        self.assertIn("存在环", str(cm.exception))
        g_unknown = {"nodes": [_node(1, "no_such_node")], "links": []}
        with self.assertRaises(FlowError) as cm2:
            Executor(g_unknown).validate()
        self.assertIn("未注册", str(cm2.exception))


class IrRoundtripTest(unittest.TestCase):
    """IR 数据契约：端口类型表 + 几何对象 JSON 往返。"""

    def test_port_types_table(self):
        self.assertEqual(set(PORT_TYPES), {"image", "result", "number"})
        for meta in PORT_TYPES.values():
            self.assertTrue(meta["color"].startswith("#"))
            self.assertTrue(meta["desc"])

    def test_geometry_roundtrip(self):
        g = Geometry(kind="polygon",
                     data=np.asarray([[0, 0], [8, 0], [8, 6]], dtype=float),
                     subpixel=True)
        back = Geometry.from_json(g.to_json())
        self.assertEqual(back.kind, "polygon")
        self.assertTrue(np.allclose(back.data, g.data))
        self.assertTrue(back.subpixel)


if __name__ == "__main__":
    unittest.main()
