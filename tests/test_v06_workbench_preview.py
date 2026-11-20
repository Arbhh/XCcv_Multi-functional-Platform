"""V0.6 F2-4 回归：传统平台工作台「运行预览」与「运行日志」布局。

现象：传统平台现在没有图片预览功能了（疑似被日志部分盖住）……日志部分
应该在右下角，中间的下部分是图片运行预览。

- 布局：日志区在右下（组合测试下方），运行预览在中列下部（试跑按钮下方）；
- 有图像输出的算子（gaussian_blur）预览=输出图，尺寸与输入一致；
- 无图像输出的算子（find_circle）预览=「输出+结果叠加」，且叠加确有绘制；
- 同参数重跑（整链命中会话缓存、快照仓为空）预览仍在；
- 组合链预览末节点；点选链中算子切换预览该步；改链后点选不再误映射；
- 运行失败：预览清空并提示，不抛异常；单算子/组合链日志共用一处。
"""

import os
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

from xccv.modeltools import flow_session as _fs

_fs._RUN_LOG_DIR = Path(tempfile.gettempdir()) / "xccv_test_flow_logs"

_APP = None


def _app():
    global _APP
    if _APP is None:
        from PyQt5.QtWidgets import QApplication

        _APP = QApplication.instance() or QApplication([])
    return _APP


def _disc_image(path: Path, w: int = 200, h: int = 160) -> str:
    img = np.full((h, w), 20, np.uint8)
    cv2.circle(img, (w // 2, h // 2), 40, 230, -1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)
    return str(path)


class WorkbenchPreviewTest(unittest.TestCase):
    def setUp(self):
        _app()
        from PyQt5.QtCore import Qt

        from xccv.workbench import OperatorWorkbench

        self._role = Qt.UserRole
        self.w = OperatorWorkbench()
        self.w.resize(1280, 800)
        self.w.show()
        _app().processEvents()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.img = _disc_image(Path(self._tmp.name) / "disc.png")
        self.w.run_image_edit.setText(self.img)

    def tearDown(self):
        self.w.close()
        self.w.deleteLater()
        _app().processEvents()

    def _select(self, name: str):
        for i in range(self.w.op_list.count()):
            if self.w.op_list.item(i).data(self._role) == f"xccv2/{name}":
                self.w.op_list.setCurrentRow(i)
                return
        self.fail(f"算子库中没有 {name}")

    def _wait(self, timeout_s: float = 30.0):
        from PyQt5.QtCore import QEventLoop, QTimer

        deadline = time.monotonic() + timeout_s
        while self.w._run_thread is not None and time.monotonic() < deadline:
            loop = QEventLoop()
            QTimer.singleShot(20, loop.quit)
            loop.exec_()
        _app().processEvents()
        self.assertIsNone(self.w._run_thread, "试跑工作线程未在超时内结束")

    def _pos(self, widget):
        from PyQt5.QtCore import QPoint

        return widget.mapTo(self.w, QPoint(0, 0))

    # ---- 布局 ----
    def test_layout_log_bottom_right_preview_mid_bottom(self):
        log_p, chain_p = self._pos(self.w.log_out), self._pos(self.w.chain_list)
        prev_p, run_p = self._pos(self.w.preview), self._pos(self.w.btn_run)
        self.assertGreater(log_p.y(), chain_p.y(), "日志应在组合测试列表下方")
        self.assertGreater(log_p.x(), prev_p.x(), "日志应在预览右侧（右列）")
        self.assertGreater(prev_p.y(), run_p.y(), "预览应在试跑按钮下方（中列下部）")
        self.assertLess(abs(prev_p.x() - run_p.x()), 40, "预览与试跑按钮同列")
        self.assertGreater(self.w.preview.height(), 200, "预览应为大尺寸区域")
        # 兼容别名：单算子/组合链日志共用一处
        self.assertIs(self.w.single_out, self.w.log_out)
        self.assertIs(self.w.chain_out, self.w.log_out)
        self.assertIs(self.w.single_preview, self.w.preview)

    # ---- 有图像输出 ----
    def test_image_output_previewed(self):
        self._select("gaussian_blur")
        self.w._run_single()
        self._wait()
        self.assertTrue(self.w.preview.has_image())
        self.assertEqual(self.w.preview.image_size(), (200, 160))
        self.assertIn("输出图像", self.w.preview_caption.text())
        text = self.w.log_out.toPlainText()
        self.assertIn("2/2 节点成功", text)
        self.assertIn("gaussian_blur(#2)", text)
        self.assertIn("单算子试跑：gaussian_blur", text)

    # ---- 无图像输出：输出+结果叠加 ----
    def test_geometry_output_overlay_previewed(self):
        self._select("find_circle")
        self.w._run_single()
        self._wait()
        self.assertTrue(self.w.preview.has_image(), self.w.preview_caption.text())
        self.assertIn("输出+结果叠加", self.w.preview_caption.text())
        self.assertIn("[result] 几何 circle", self.w.log_out.toPlainText())
        # 叠加确有绘制：预览图与原图（灰转 BGR）不同，圆心处偏绿（半透明蒙版）
        from xccv.workbench import build_preview

        vis, note = build_preview(self.w._session.executor, 2)
        base = cv2.cvtColor(cv2.imread(self.img, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
        self.assertEqual(vis.shape, base.shape)
        self.assertGreater(int(np.abs(vis.astype(int) - base.astype(int)).sum()), 0)
        b, g, r = (int(c) for c in vis[80, 100])
        self.assertGreater(g, b)
        self.assertGreater(g, r)

    def test_repeat_run_cache_hit_still_previewed(self):
        """同参数重跑：节点全命中会话缓存、快照仓为空——预览仍须存在。"""
        self._select("find_circle")
        self.w._run_single()
        self._wait()
        self.w._run_single()
        self._wait()
        self.assertIn("命中会话缓存", self.w.log_out.toPlainText())
        self.assertEqual(self.w._session.ctx.snapshot_store.node_ids(), [])
        self.assertTrue(self.w.preview.has_image())
        self.assertIn("输出+结果叠加", self.w.preview_caption.text())

    # ---- 组合链 ----
    def test_chain_preview_last_and_click_step(self):
        self._select("blob_analysis")
        self.w._chain_add()
        self._select("filter_count_check")
        self.w._chain_add()
        self.w._run_chain()
        self._wait()
        cap = self.w.preview_caption.text()
        self.assertIn("filter_count_check(#3)", cap)
        self.assertIn("个目标", cap)
        self.assertTrue(self.w.preview.has_image())
        self.assertIn("3/3 节点成功", self.w.log_out.toPlainText())
        # 点选链中第 1 个算子 → 预览切到 blob_analysis(#2)
        self.w.chain_list.setCurrentRow(0)
        _app().processEvents()
        self.assertIn("blob_analysis(#2)", self.w.preview_caption.text())
        self.assertTrue(self.w.preview.has_image())
        # 改链（移除）后点选不再映射到旧结果
        self.w.chain_list.setCurrentRow(1)
        self.w._chain_remove()
        before = self.w.preview_caption.text()
        self.w.chain_list.setCurrentRow(0)
        _app().processEvents()
        self.assertEqual(self.w.preview_caption.text(), before)

    # ---- 失败路径 ----
    def test_failed_run_clears_preview_without_raise(self):
        self._select("gaussian_blur")
        self.w.run_image_edit.setText(str(Path(self._tmp.name) / "不存在.png"))
        self.w._run_single()
        self._wait()
        self.assertFalse(self.w.preview.has_image())
        self.assertIn("未执行", self.w.preview_caption.text())
        self.assertIn("个失败", self.w.log_out.toPlainText())


class BuildPreviewUnitTest(unittest.TestCase):
    def test_no_executor(self):
        from xccv.workbench import build_preview

        img, note = build_preview(None, 2)
        self.assertIsNone(img)
        self.assertIn("尚未运行", note)

    def test_to_bgr_defense(self):
        from xccv.workbench import _to_bgr_u8

        self.assertEqual(_to_bgr_u8(np.zeros((4, 5), np.uint8)).shape, (4, 5, 3))
        self.assertEqual(_to_bgr_u8(np.zeros((4, 5, 4), np.uint8)).shape, (4, 5, 3))
        self.assertEqual(_to_bgr_u8(np.zeros((4, 5, 1), np.uint8)).shape, (4, 5, 3))
        out = _to_bgr_u8(np.linspace(0, 1, 20, dtype=np.float32).reshape(4, 5))
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(int(out.max()), 255)
        self.assertEqual(int(_to_bgr_u8(np.ones((2, 2), bool)).max()), 255)

    def test_overlay_small_image_pixels_unchanged_by_scaling(self):
        """overlay_result 线宽缩放只作用于长边 >1000 的大图：小图与原 2px 描边逐位一致。"""
        from xccv.engine.flow.ir import Geometry
        from xccv.engine.flow.snapshot import overlay_result

        img = np.zeros((100, 120), np.uint8)
        geo = Geometry(kind="circle", data=np.array([60.0, 50.0, 20.0]))
        vis = overlay_result(img, {"c": geo})
        ref = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        yy, xx = np.ogrid[:100, :120]
        mask = (xx - 60.0) ** 2 + (yy - 50.0) ** 2 <= 400.0
        ref[mask] = (ref[mask].astype(np.float32) * (1 - 0.38)
                     + np.array([0, 220, 60], np.float32) * 0.38).astype(np.uint8)
        cv2.circle(ref, (60, 50), 20, (0, 220, 60), 2, cv2.LINE_AA)
        self.assertTrue(np.array_equal(vis, ref))

    def test_overlay_box_inplace_equals_mask_blend(self):
        """检出框就地 ROI 混合与原整图掩码混合逐位一致。"""
        from xccv.engine.flow.snapshot import overlay_result
        from xccv.engine.models.base import Prediction

        rng = np.random.default_rng(5)
        img = rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)
        pred = Prediction(labels=[0], scores=[0.9], boxes=[[10.2, 5.7, 40.4, 30.1]])
        vis = overlay_result(img, {"d": pred})
        ref = img.copy()
        x1, y1, x2, y2 = 10, 6, 40, 30
        m = np.zeros((60, 80), bool)
        m[y1:y2 + 1, x1:x2 + 1] = True
        region = ref[m]
        fill = np.empty_like(region)
        fill[:] = (0, 140, 255)
        ref[m] = (region.astype(np.float32) * (1 - 0.38)
                  + fill.astype(np.float32) * 0.38).astype(np.uint8)
        cv2.rectangle(ref, (x1, y1), (x2, y2), (0, 140, 255), 2, cv2.LINE_AA)
        self.assertTrue(np.array_equal(vis, ref))


if __name__ == "__main__":
    unittest.main()
