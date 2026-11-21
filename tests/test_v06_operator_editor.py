"""QtTest E2E：工作台「编辑算子」新建→Ctrl+S 保存→列表出现→试跑→改代码 Ctrl+S 覆盖→再试跑。

自定义算子目录用临时目录（替换 custom_ops.operators_dir），不触碰真实 platform_data。
"""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from xccv.engine.flow import custom_ops as co

_INVERT = ("def run(inputs, params, ctx):\n"
           "    img = inputs['image']\n"
           "    return {'image': ImageData(mat=255 - img.mat, meta=dict(img.meta))}\n")
_IDENTITY = ("def run(inputs, params, ctx):\n"
             "    return {'image': inputs['image']}\n")


def _ctrl_s(widget):
    """真实用户按 Ctrl+S 时编辑器是活动窗口（QShortcut 为窗口级快捷键）。"""
    QApplication.setActiveWindow(widget.window())
    QApplication.processEvents()
    QTest.keyClick(widget, Qt.Key_S, Qt.ControlModifier)
    QApplication.processEvents()


def _pump_until(cond, timeout=20.0):
    app = QApplication.instance()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


class OperatorEditorE2ETest(unittest.TestCase):
    def setUp(self):
        from xccv.engine.io_utils import imwrite_unicode
        from xccv.workbench import OperatorWorkbench

        self.app = QApplication.instance() or QApplication([])
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "operators"
        real = co.operators_dir
        patcher = mock.patch.object(co, "operators_dir",
                                    side_effect=lambda root=None: real(root or self.root))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.img_path = Path(self._tmp.name) / "试跑图.png"
        rng = np.random.default_rng(1)
        self.mat = rng.integers(0, 255, size=(32, 40, 3), dtype=np.uint8)
        imwrite_unicode(self.img_path, self.mat)
        self.wb = OperatorWorkbench()
        self.wb.show()
        self.wb.run_image_edit.setText(str(self.img_path))

    def tearDown(self):
        if self.wb._editor is not None:
            self.wb._editor.close()
        _pump_until(lambda: self.wb._run_thread is None, 10)
        self.wb.close()
        self.wb.deleteLater()
        self.app.processEvents()
        co.load_custom_ops(self.root / "__empty__")   # 注销本用例注册的 custom/ 节点
        self._tmp.cleanup()

    def _run_and_get(self):
        QTest.mouseClick(self.wb.btn_run, Qt.LeftButton)
        self.assertTrue(_pump_until(lambda: self.wb._run_thread is None and
                                    self.wb._session.executor is not None and
                                    2 in self.wb._session.executor.results))
        res = self.wb._session.executor.results[2]
        self.assertTrue(res.ok, res.error)
        return res.outputs["image"].mat

    def test_new_save_list_run_overwrite(self):
        self.wb.op_list.setCurrentRow(-1)
        QTest.mouseClick(self.wb.btn_edit_op, Qt.LeftButton)
        ed = self.wb._editor
        self.assertIsNotNone(ed)
        ed.id_edit.setText("e2e_invert")
        ed.title_edit.setText("E2E 反色")
        ed.impl_edit.setPlainText(_INVERT)
        _ctrl_s(ed.impl_edit)                                      # Ctrl+S 保存（焦点在代码区）
        self.assertTrue((self.root / "e2e_invert.op.json").is_file())
        self.assertIn("已保存并注册", ed.status.text())
        cur = self.wb.op_list.currentItem()
        self.assertIsNotNone(cur)
        self.assertEqual(cur.data(Qt.UserRole), "xccv2/custom/e2e_invert")   # 列表出现且选中
        out1 = self._run_and_get()
        np.testing.assert_array_equal(out1, 255 - self.mat)           # 试跑：反色

        ed.impl_edit.setPlainText(_IDENTITY)                          # 改实现 → Ctrl+S 覆盖
        _ctrl_s(ed)
        saved = json.loads((self.root / "e2e_invert.op.json").read_text(encoding="utf-8"))
        self.assertIn("inputs['image']}", saved["script"]["code"])
        out2 = self._run_and_get()
        np.testing.assert_array_equal(out2, self.mat)                 # 新实现生效（无旧缓存）

    def test_invalid_definition_not_saved(self):
        QTest.mouseClick(self.wb.btn_edit_op, Qt.LeftButton)
        ed = self.wb._editor
        ed.id_edit.setText("Bad-Id")
        _ctrl_s(ed)
        self.assertIn("未保存", ed.status.text())
        self.assertFalse(self.root.exists() and any(self.root.glob("*.op.json")))

    def test_reload_and_import_buttons(self):
        d = co.new_op_template("composite")
        d["id"] = "dropped_in"
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "dropped_in.op.json").write_text(json.dumps(d, ensure_ascii=False),
                                                       encoding="utf-8")
        QTest.mouseClick(self.wb.btn_reload_op, Qt.LeftButton)        # 刷新：热重载
        self.assertIn("xccv2/custom/dropped_in", self.wb._lib)
        pkg = co.export_xcvop("dropped_in", Path(self._tmp.name) / "包.xcvop")
        (self.root / "dropped_in.op.json").unlink()
        with mock.patch("xccv.workbench.QFileDialog.getOpenFileName",
                        return_value=(str(pkg), "")):
            QTest.mouseClick(self.wb.btn_import_op, Qt.LeftButton)    # 导入 .xcvop
        self.assertTrue((self.root / "dropped_in.op.json").is_file())
        self.assertEqual(self.wb.op_list.currentItem().data(Qt.UserRole),
                         "xccv2/custom/dropped_in")


if __name__ == "__main__":
    unittest.main()

