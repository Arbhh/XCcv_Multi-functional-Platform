"""逻辑 4 算子测试（每算子 ≥2 例）。"""

import math
import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import xccv.engine.flow.ops.logic  # noqa: F401  触发注册
from xccv.engine.flow.ir import ImageData, MeasureResult, NumberVal
from xccv.engine.flow.node_base import RunContext

CTX = lambda: RunContext()  # noqa: E731


class CompareValueTest(unittest.TestCase):
    def test_gt_pass_and_fail(self):
        node = pp_compare()
        self.assertTrue(node.run({"number": NumberVal(5.0)},
                                 {"op": "gt", "threshold": 1.0, "hysteresis": 0},
                                 CTX())["result"])
        self.assertFalse(node.run({"number": NumberVal(-1.0)},
                                  {"op": "gt", "threshold": 0.0, "hysteresis": 0},
                                  CTX())["result"])

    def test_nan_fails_with_warning(self):
        """除零链路的 NaN：恒判 False 并在人话警告中说明（防静默通过）。"""
        ctx = CTX()
        passed = pp_compare().run({"number": NumberVal(float("nan"))},
                                  {"op": "ge", "threshold": 0.0, "hysteresis": 0},
                                  ctx)["result"]
        self.assertFalse(passed)
        self.assertTrue(any("NaN" in w for w in ctx.warnings))

    def test_hysteresis_holds_state(self):
        """回差：触发需超阈+回差，保持只需不跌破阈-回差（会话状态）。"""
        ctx = CTX()
        node = pp_compare()
        kw = {"op": "gt", "threshold": 10.0, "hysteresis": 2.0}
        self.assertFalse(node.run({"number": NumberVal(11.0)}, kw, ctx)["result"])
        self.assertTrue(node.run({"number": NumberVal(13.0)}, kw, ctx)["result"])
        # 已触发态下 9.0：未跌破 10-2=8 → 保持通过（无回差时 9<10 会翻为 False）
        self.assertTrue(node.run({"number": NumberVal(9.0)}, kw, ctx)["result"])


def pp_compare():
    from xccv.engine.flow.registry import get_node_class

    return get_node_class("compare_value")()


class MathOpTest(unittest.TestCase):
    def test_add_and_sub(self):
        node = pp_math()
        out = node.run({"a": NumberVal(7, "mm"), "b": NumberVal(2, "mm")},
                       {"op": "add"}, CTX())["number"]
        self.assertEqual(out.value, 9.0)
        self.assertEqual(out.unit, "mm")
        out2 = node.run({"a": NumberVal(7), "b": NumberVal(2)},
                        {"op": "sub"}, CTX())["number"]
        self.assertEqual(out2.value, 5.0)

    def test_div_by_zero_outputs_nan_with_warning(self):
        """除零不静默：输出 NaN + ctx 警告（下游 compare 恒 False）。"""
        ctx = CTX()
        out = pp_math().run({"a": NumberVal(3), "b": NumberVal(0)},
                            {"op": "div"}, ctx)["number"]
        self.assertTrue(math.isnan(out.value))
        self.assertTrue(any("除零" in w for w in ctx.warnings))

    def test_abs_without_b(self):
        """abs 只需 a（b 端口可选）：缺 b 不报错。"""
        out = pp_math().run({"a": NumberVal(-4)}, {"op": "abs"}, CTX())["number"]
        self.assertEqual(out.value, 4.0)


def pp_math():
    from xccv.engine.flow.registry import get_node_class

    return get_node_class("math_op")()


class ConditionBranchTest(unittest.TestCase):
    def _imgs(self):
        a = ImageData(mat=np.zeros((4, 4), np.uint8))
        b = ImageData(mat=np.full((4, 4), 9, np.uint8))
        return a, b

    def test_selects_true_input(self):
        a, b = self._imgs()
        out = pp_branch().run({"cond": True, "true_input": a, "false_input": b},
                              {}, CTX())["image"]
        self.assertIs(out, a)  # 零拷贝直通

    def test_selects_false_input(self):
        a, b = self._imgs()
        out = pp_branch().run({"cond": False, "true_input": a, "false_input": b},
                              {}, CTX())["image"]
        self.assertIs(out, b)


def pp_branch():
    from xccv.engine.flow.registry import get_node_class

    return get_node_class("condition_branch")()


class ResultMergeTest(unittest.TestCase):
    def _m(self, value=None, values=None, passed=None):
        return MeasureResult(value=value, values=values, passed=passed)

    def test_allpass_and_anypass(self):
        node = pp_merge()
        ok, ng = self._m(value=1.0, passed=True), self._m(value=0.5, passed=False)
        r1 = node.run({"m1": ok, "m2": ok}, {"op": "allpass"}, CTX())
        self.assertTrue(r1["result"])
        r2 = node.run({"m1": ok, "m2": ng}, {"op": "allpass"}, CTX())
        self.assertFalse(r2["result"])
        r3 = node.run({"m1": ok, "m2": ng}, {"op": "anypass"}, CTX())
        self.assertTrue(r3["result"])

    def test_count_ng(self):
        node = pp_merge()
        r = node.run({"m1": self._m(passed=True),
                      "m2": self._m(passed=False),
                      "m3": self._m(passed=False)}, {"op": "count_ng"}, CTX())
        self.assertEqual(r["number"].value, 2.0)
        self.assertFalse(r["result"])

    def test_mean_over_multi_point_measures(self):
        """多点测量（values 序列）参与聚合。"""
        node = pp_merge()
        r = node.run({"m1": self._m(values=[2.0, 4.0]),
                      "m2": self._m(value=6.0)}, {"op": "mean"}, CTX())
        self.assertEqual(r["number"].value, 4.0)

    def test_empty_inputs_nan_with_warning(self):
        ctx = CTX()
        r = node_run_empty_merge(ctx)
        self.assertTrue(math.isnan(r["number"].value))
        self.assertTrue(any("未收集" in w for w in ctx.warnings))


def node_run_empty_merge(ctx):
    return pp_merge().run({"m1": None}, {"op": "mean"}, ctx)


def pp_merge():
    from xccv.engine.flow.registry import get_node_class

    return get_node_class("result_merge")()


if __name__ == "__main__":
    unittest.main()

