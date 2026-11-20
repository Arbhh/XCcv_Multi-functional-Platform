"""槽自动包装测试：Cython 编译态信号槽参数裁剪修复（xccv/qt_slots.py）。

编译态下 PyQt5 无法裁剪 clicked(bool) 等信号的多余参数，所有无参槽方法
报 TypeError。apply_auto_slot_module 用纯 Python 包装函数替换编译方法。
本测试在 dev 态（源码函数）下验证包装行为的等价性；编译态由打包冒烟的
"槽包装检查"项兜底（断言类方法为纯 Python 函数）。
"""

import os
import sys
import types
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QPushButton

from xccv.qt_slots import apply_auto_slot, apply_auto_slot_module


class _Dummy(QObject):
    clicked_log = []

    text_changed = pyqtSignal(str)

    def ping(self):
        self.clicked_log.append("ping")

    def with_arg(self, value):
        self.clicked_log.append(f"arg={value}")

    @pyqtSlot()
    def explicit(self):
        self.clicked_log.append("explicit")


class QtSlotsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        _Dummy.clicked_log = []
        apply_auto_slot(_Dummy)

    def test_zero_arg_method_wrapped_as_plain_function(self):
        """零参方法被替换为纯 Python 函数（编译态修复的判定条件）。"""
        self.assertIsInstance(_Dummy.ping, types.FunctionType)
        self.assertIsNotNone(getattr(_Dummy.ping, "__wrapped__", None))

    def test_click_executes_compiled_style_slot(self):
        """连接 clicked 并点击：包装后正常执行、无 TypeError。"""
        d = _Dummy()
        b = QPushButton()
        b.clicked.connect(d.ping)
        b.click()
        self.assertEqual(_Dummy.clicked_log, ["ping"])

    def test_extra_signal_args_trimmed(self):
        """带多余参数的信号（textChanged(str) → 无参槽）：包装层完成裁剪。"""
        d = _Dummy()
        d.text_changed.connect(d.ping)
        d.text_changed.emit("hello")
        self.assertEqual(_Dummy.clicked_log, ["ping"])

    def test_args_passthrough_for_typed_slots(self):
        """带参槽：参数透传不受包装影响。"""
        d = _Dummy()
        d.text_changed.connect(d.with_arg)
        d.text_changed.emit("abc")
        self.assertEqual(_Dummy.clicked_log, ["arg=abc"])

    def test_explicit_pyqt_slot_untouched(self):
        """已显式 @pyqtSlot 的方法不被重复包装。"""
        self.assertIsNone(getattr(_Dummy.explicit, "__wrapped__", None))
        d = _Dummy()
        b = QPushButton()
        b.clicked.connect(d.explicit)
        b.click()
        self.assertEqual(_Dummy.clicked_log, ["explicit"])

    def test_apply_module_scan_and_idempotent(self):
        """模块扫描：只处理本模块定义的 QObject 子类；重复应用幂等。"""
        import importlib

        mod = importlib.import_module("tests._slot_sample")
        apply_auto_slot_module(vars(mod))
        first = mod.Sample.do_it
        apply_auto_slot_module(vars(mod))
        self.assertIs(mod.Sample.do_it, first)
        self.assertIsInstance(first, types.FunctionType)
        d = mod.Sample()
        b = QPushButton()
        b.clicked.connect(d.do_it)
        b.click()
        self.assertEqual(mod.SAMPLE_LOG, ["done"])

    def test_non_qobject_ignored(self):
        """非 QObject 类不处理。"""

        class Plain:
            def method(self):
                return 1

        apply_auto_slot(Plain)
        self.assertIsNone(getattr(Plain.method, "__wrapped__", None))
        self.assertEqual(Plain().method(), 1)


if __name__ == "__main__":
    unittest.main()
