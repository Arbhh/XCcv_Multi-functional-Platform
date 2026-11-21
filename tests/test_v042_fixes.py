"""v0.4.2 修复测试：标注平台亮度对比度对话框、叠加蒙版、日志保留数、
模型节点拆分后的打包校验（其余见 test_v041_fixes / test_flow_ai_bridge）。"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import PIL.Image

_APP = None
_WINDOW_HOLDER = []


def _app():
    global _APP
    if _APP is None:
        from PyQt5.QtWidgets import QApplication

        _APP = QApplication.instance() or QApplication([])
    return _APP


class BrightnessContrastDialogTest(unittest.TestCase):
    """Q6 修复：RGBA 图不再内存错位崩溃；滑条驱动回调正常。"""

    def _dialog(self, mode):
        from labelme.widgets.brightness_contrast_dialog import (
            BrightnessContrastDialog,
        )

        _app()
        img = PIL.Image.new(mode, (24, 16), (120, 120, 120, 255)[:len(mode)])
        got = {}
        dlg = BrightnessContrastDialog(img, lambda qimage: got.update(qimage=qimage))
        _WINDOW_HOLDER.append(dlg)
        self.addCleanup(dlg.close)
        return dlg, got

    def test_rgba_image_callback(self):
        """RGBA 源图（PNG 常见）：不再错位崩溃，回调收到正确尺寸的 QImage。"""
        dlg, got = self._dialog("RGBA")
        dlg.slider_brightness.setValue(80)     # 1.6 倍亮度
        dlg.onNewValue(None)
        self.assertIn("qimage", got)
        self.assertEqual(got["qimage"].width(), 24)
        self.assertEqual(got["qimage"].height(), 16)

    def test_rgb_image_and_neutral(self):
        dlg, got = self._dialog("RGB")
        dlg.onNewValue(None)                    # 1.0/1.0 中性值也走通
        self.assertIn("qimage", got)

    def test_buffer_held_for_qimage(self):
        """QImage 不拷贝字节：缓冲必须被对话框持有（否则释放后绘制即崩）。"""
        from labelme.widgets.brightness_contrast_dialog import (
            BrightnessContrastDialog,
        )

        dlg, _ = self._dialog("RGB")
        dlg.onNewValue(None)
        self.assertTrue(len(dlg._buf) >= 24 * 16 * 3)


class OverlayMaskTest(unittest.TestCase):
    """ToC4/ToE2 修复：几何/检出叠加为半透明蒙版（大面积变化，非细线）。"""

    def test_circle_mask_fills_area(self):
        from xccv.engine.flow.snapshot import overlay_result
        from xccv.engine.flow.ir import Geometry

        img = np.zeros((80, 80, 3), np.uint8)
        circle = Geometry(kind="circle", data=np.array([40.0, 40.0, 15.0]))
        vis = overlay_result(img, {"circle": circle})
        changed = int((vis.sum(axis=2) > 0).sum())
        fill_area = int(3.1416 * 15 * 15)
        self.assertGreater(changed, fill_area * 0.5)   # 填充蒙版生效（远超细线）
        # 圆心附近像素明显被蒙版染色
        self.assertGreater(int(vis[40, 40, 1]), 30)    # G 通道抬升

    def test_detection_boxes_mask(self):
        from xccv.engine.flow.snapshot import overlay_result
        from xccv.engine.models.base import Prediction

        img = np.zeros((60, 60, 3), np.uint8)
        pred = Prediction(labels=[0], scores=[0.9], boxes=[[10, 10, 40, 40]])
        vis = overlay_result(img, {"detection": pred})
        self.assertGreater(int(vis[25, 25, 2]), 30)    # 框内橙色蒙版

    def test_line_thick_with_endpoints(self):
        from xccv.engine.flow.snapshot import overlay_result
        from xccv.engine.flow.ir import Geometry

        img = np.zeros((50, 50, 3), np.uint8)
        line = Geometry(kind="line", data=np.array([[5.0, 25.0], [45.0, 25.0]]))
        vis = overlay_result(img, {"line": line})
        self.assertGreater(int(vis[25, 25, 1]), 30)    # 线身上色
        self.assertGreater(int(vis[25, 5, 1]), 30)     # 端点圆点


class LogRetentionTest(unittest.TestCase):
    def test_keep_is_50(self):
        from xccv.modeltools import flow_session as fs

        self.assertEqual(fs._RUN_LOG_KEEP, 50)


if __name__ == "__main__":
    unittest.main()

