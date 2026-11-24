"""标注平台测试：3 个历史测试模块的合并模块。

由以下模块按域合并而成，用例逐段原样迁移、断言未改：
- test_batch20_workbench
- test_v06_workbench_preview
- test_v06_fluency
"""


# == 来源模块：test_batch20_workbench ==
"""测试：5 官方模板可加载可执行、传统平台算子工作台。"""

import os
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

import cv2
import numpy as np

from xccv.modeltools.flow_session import (
    FlowSession, _normalize_types, list_templates, node_library_payload,
)

node_library_payload()  # 注册全部算子（模板图与工作台都依赖注册表）

# FlowSession 运行日志重定向到临时目录（避免测试污染仓库 platform_data）
import tempfile as _tempfile

from xccv.modeltools import flow_session as _fs
from pathlib import Path as _Path

_fs._RUN_LOG_DIR = _Path(_tempfile.gettempdir()) / "xccv_test_flow_logs"

_APP = None


def _app():
    global _APP
    if _APP is None:
        from PyQt5.QtWidgets import QApplication

        _APP = QApplication.instance() or QApplication([])
    return _APP


def _scene(tmp: Path) -> str:
    rng = np.random.default_rng(3)
    img = rng.integers(0, 35, size=(160, 200), dtype=np.uint8)
    yy, xx = np.mgrid[:160, :200]
    disc = (np.clip(25.5 - np.hypot(xx - 140.0, yy - 50.0), 0, 1) * 255)
    img = np.maximum(img, disc.astype(np.uint8))
    cv2.line(img, (5, 120), (195, 100), 230, 2, cv2.LINE_AA)
    p = tmp / "scene.png"
    cv2.imwrite(str(p), img)
    return str(p)


class TemplatesTest(unittest.TestCase):
    def test_five_templates_with_meta(self):
        ts = list_templates()
        ids = {t["template"] for t in ts}
        self.assertLessEqual({"tmpl_ai_filter", "tmpl_blob_prefilter", "tmpl_circle_locate",
                              "tmpl_line_level", "tmpl_template_match",
                              "tmpl_cascade_det_seg", "tmpl_crop_export"}, ids)  # V0.6 +2
        titles = {t["title"] for t in ts}
        self.assertIn("AI 检测 + 过滤 + 数量判定", titles)
        for t in ts:
            self.assertTrue(t["description"])
            self.assertTrue(t["template"])

    def test_light_templates_run_end_to_end(self):
        """全部无 AI 模板在合成图上全链跑通并出快照（AI 模板由冒烟实测覆盖）。"""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            scene = _scene(tmp)
            for t in list_templates():
                if any(n["type"].split("/")[-1].startswith("model_")
                       for n in t["flow"]["nodes"]):
                    continue   # 含 AI 模型节点的模板由冒烟实测覆盖（无模型文件可跑）
                flow = _normalize_types(t["flow"])
                for n in flow["nodes"]:
                    if n["type"].endswith("image_loader"):
                        n["properties"]["path"] = scene
                    # 落盘类算子指向临时目录（默认会写到 cwd 的「流程结果/」）
                    if n["type"].endswith(("image_sink", "result_output")):
                        n["properties"]["out_dir"] = str(tmp / "out")
                    if n["type"].endswith("csv_sink"):
                        n["properties"]["out_path"] = str(tmp / "out" / "r.csv")
                    if n["type"].endswith(("crop_by_detection", "grid_tile")):
                        n["properties"]["save_dir"] = str(tmp / "crops")
                    if n["type"].endswith("template_match"):
                        import xccv.engine.flow.ops.locate  # noqa: F401
                        from xccv.engine.flow.registry import get_node_class

                        piece = cv2.imread(scene, cv2.IMREAD_GRAYSCALE)[40:80, 20:76]
                        p = get_node_class("template_match")().record(
                            np.ascontiguousarray(piece),
                            (0, 0, piece.shape[1], piece.shape[0]), tmp / "tmpl")
                        n["properties"]["template_path"] = p
                        n["properties"]["threshold"] = 0.6
                        n["properties"]["angle_range"] = 0.0
                s = FlowSession()
                summary = s.run(flow)
                self.assertEqual(summary["errors"], {},
                                 f"{t['title']}: {summary['errors']}")
                self.assertGreater(len(s.ctx.snapshot_store.node_ids()), 0,
                                   f"{t['title']} 未产出快照")

    def test_ai_template_structure_loads(self):
        """AI 模板图结构与注册表自洽。"""
        from xccv.engine.flow.executor import Executor
        from xccv.engine.flow.node_base import RunContext

        t = next(t for t in list_templates()
                 if t["template"] == "tmpl_ai_filter")
        flow = _normalize_types(t["flow"])
        Executor(flow, RunContext()).validate()  # 不抛即结构/类型/无环


class WorkbenchTest(unittest.TestCase):
    def setUp(self):
        _app()
        from PyQt5.QtCore import Qt

        self._user_role = Qt.UserRole
        from xccv.workbench import OperatorWorkbench

        self.w = OperatorWorkbench()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _select_op(self, name: str):
        for i in range(self.w.op_list.count()):
            if self.w.op_list.item(i).data(self._user_role) == f"xccv2/{name}":
                self.w.op_list.setCurrentRow(i)
                return True
        return False

    def _wait_run_done(self, timeout_s: float = 30.0):
        """试跑已后台化——泵事件直到工作线程收尾（结果区出现汇总）。"""
        from PyQt5.QtCore import QEventLoop, QTimer

        deadline = time.monotonic() + timeout_s
        while self.w._run_thread is not None and time.monotonic() < deadline:
            loop = QEventLoop()
            QTimer.singleShot(20, loop.quit)
            loop.exec_()
        _app().processEvents()
        self.assertIsNone(self.w._run_thread, "试跑工作线程未在超时内结束")

    def test_library_and_detail(self):
        _app()
        self.assertGreaterEqual(self.w.op_list.count(), 25)
        self.assertTrue(self._select_op("gaussian_blur"))
        self.assertIn("高斯模糊", self.w.detail_box.toPlainText())
        self.assertIn("核数", self.w.detail_box.toPlainText())

    def test_param_form_roundtrip(self):
        self.assertTrue(self._select_op("threshold"))
        form = self.w._param_edits
        self.assertIn("mode", form)
        spec, edit = form["mode"]
        edit.setCurrentText("otsu")
        params = self.w._collect_params(
            self.w._lib["xccv2/threshold"])
        self.assertEqual(params["mode"], "otsu")

    def test_single_run_with_logs(self):
        """单算子试跑：真实执行 + 结果区含节点级日志（日志捕捉可见）。"""
        p = Path(self._tmp.name) / "in.png"
        cv2.imwrite(str(p), np.full((20, 20), 90, np.uint8))
        self.w.run_image_edit.setText(str(p))
        self.assertTrue(self._select_op("gaussian_blur"))
        self.w._run_single()
        self._wait_run_done()
        text = self.w.single_out.toPlainText()
        self.assertIn("2/2 节点成功", text)
        self.assertIn("gaussian_blur(#2)", text)

    def test_chain_run_blob_to_count_check(self):
        """组合链：blob_analysis → filter_count_check（检测端口自动匹配）。"""
        rng = np.random.default_rng(1)
        img = np.zeros((80, 100), np.uint8)
        cv2.circle(img, (40, 40), 12, 255, -1)
        p = Path(self._tmp.name) / "chain.png"
        cv2.imwrite(str(p), img)
        self.w.run_image_edit.setText(str(p))
        self.assertTrue(self._select_op("blob_analysis"))
        self.w._chain_add()
        self.assertTrue(self._select_op("filter_count_check"))
        self.w._chain_add()
        self.w._run_chain()
        self._wait_run_done()
        text = self.w.chain_out.toPlainText()
        self.assertIn("3/3 节点成功", text)
        self.assertIn("[count]", text)   # 数量判定：count 数值输出
        self.assertIn("[result] 检出 1 个目标", text)

    def test_chain_type_mismatch_rejected(self):
        """类型不匹配的组合在搭链时报错（不允许生成跑不过的图）。"""
        p = Path(self._tmp.name) / "x.png"
        cv2.imwrite(str(p), np.zeros((10, 10), np.uint8))
        self.w.run_image_edit.setText(str(p))
        self.assertTrue(self._select_op("gaussian_blur"))
        self.w._chain_add()
        self.assertTrue(self._select_op("filter_confidence"))
        self.w._chain_add()   # filter_confidence 需要 detection，blur 只有 image
        graph, error = self.w._build_chain_graph(str(p))
        self.assertIsNone(graph)
        self.assertIn("无法连接", error)

    def test_single_run_requires_image(self):
        self.assertTrue(self._select_op("gaussian_blur"))
        self.w.run_image_edit.setText("")
        from unittest import mock

        with mock.patch("xccv.workbench.QMessageBox.information") as m:
            self.w._run_single()
            self.assertTrue(m.called)

# == 来源模块：test_v06_workbench_preview ==
"""传统平台工作台「运行预览」与「运行日志」布局回归。

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

# == 来源模块：test_v06_fluency ==
"""全局流畅度治理（后台化/分桶/防抖/缓存）。"""

import os
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

from PyQt5.QtCore import Qt as _Qt

_USER_ROLE = _Qt.UserRole

from xccv.training.data_health import _dhash, _duplicate_groups

_ROOT = Path(__file__).resolve().parents[1]




def _pump(ms: int):
    from PyQt5.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec_()


def _flush_deferred():
    """立即执行 deleteLater 队列（对齐 test_ui_upgrade 模式，防 offscreen 退出段错误）。"""
    from PyQt5.QtCore import QEvent

    _app().processEvents()
    _app().sendPostedEvents(None, QEvent.DeferredDelete)
    _app().processEvents()


class DataHealthBucketTest(unittest.TestCase):
    """分桶聚类与全量两两比对结果等价（同一判定谓词）。"""

    @staticmethod
    def _make_hashes(n_unique: int = 60, dup_clusters: int = 4) -> dict:
        rng = np.random.default_rng(7)
        hashes, i = {}, 0
        for c in range(dup_clusters):
            base = int(rng.integers(0, 2**64, dtype=np.uint64))
            for _k in range(3):  # 每簇 3 张近似图（距离 ≤4）
                noisy = base
                for _b in range(2):
                    noisy ^= 1 << int(rng.integers(0, 64))
                hashes[f"dup{c}_{i}.png"] = noisy
                i += 1
        for u in range(n_unique):
            hashes[f"u{u}.png"] = int(rng.integers(0, 2**64, dtype=np.uint64))
        return hashes

    def _groups_as_sets(self, groups):
        return sorted(frozenset(g) for g in groups)

    def test_bucket_equivalent_to_bruteforce(self):
        hashes = self._make_hashes()
        for threshold in (0, 2, 4, 6, 7):   # 分桶路径（<8）
            fast = _duplicate_groups(hashes, threshold)
            # 暴力参考实现（O(n²)，强制走 fallback 分支逻辑）
            paths = list(hashes)
            parent = {p: p for p in paths}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    if bin(hashes[paths[i]] ^ hashes[paths[j]]).count("1") <= threshold:
                        parent[find(paths[i])] = find(paths[j])
            ref = {}
            for p in paths:
                ref.setdefault(find(p), []).append(p)
            ref_groups = [g for g in ref.values() if len(g) > 1]
            self.assertEqual(self._groups_as_sets(fast),
                             self._groups_as_sets(ref_groups),
                             f"threshold={threshold} 分桶与全量结果不一致")

    def test_high_threshold_fallback_bruteforce(self):
        hashes = self._make_hashes()
        fast = _duplicate_groups(hashes, 8)     # ≥8 走回退分支
        brute = _duplicate_groups(hashes, 12)   # 必然全对（更大阈值更多组）
        self.assertIsInstance(fast, list)
        self.assertIsInstance(brute, list)
        self.assertLessEqual(len(fast), len(brute))

    def test_bucket_performance_on_thousand(self):
        """千级哈希分桶耗时显著低于全量两两（量级 sanity，不卡阈值）。"""
        hashes = {f"k{i}.png": (i * 2654435761) % (1 << 64)
                  for i in range(1200)}
        t0 = time.perf_counter()
        _duplicate_groups(hashes, 4)
        fast = time.perf_counter() - t0
        self.assertLess(fast, 20.0)  # 全量两两在千级要 >5s，分桶应远小于此


class WorkbenchAsyncTest(unittest.TestCase):
    """试跑后台化——提交后主线程立即可继续（线程在跑），完成后结果回填。"""

    def test_run_single_async_with_wait(self):
        _app()
        from xccv.workbench import OperatorWorkbench

        w = OperatorWorkbench()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "in.png"
            cv2.imwrite(str(p), np.full((64, 64), 90, np.uint8))
            w.run_image_edit.setText(str(p))
            for i in range(w.op_list.count()):
                if w.op_list.item(i).data(_USER_ROLE) == "xccv2/gaussian_blur":
                    w.op_list.setCurrentRow(i)
                    break
            w._run_single()
            self.assertIsNotNone(w._run_thread)   # 已在后台线程
            self.assertFalse(w.btn_run.isEnabled())  # 忙态禁用
            deadline = time.monotonic() + 30
            while w._run_thread is not None and time.monotonic() < deadline:
                _pump(20)
            _app().processEvents()
            self.assertIsNone(w._run_thread)
            self.assertTrue(w.btn_run.isEnabled())
            self.assertIn("2/2 节点成功", w.single_out.toPlainText())
            w.close()
            _flush_deferred()


class SearchDebounceTest(unittest.TestCase):
    """textChanged 走 300ms 防抖（一次输入风暴只触发一次过滤）。"""

    def test_debounce_batches_typing(self):
        from unittest import mock

        _app()
        from xccv.app import HomeWindow

        win = HomeWindow()
        self.addCleanup(win.close)
        self.addCleanup(_flush_deferred)
        with mock.patch.object(win, "_apply_search") as spied:
            for ch in "训练平台":
                win.search_edit.setText(win.search_edit.text() + ch)
                _pump(30)  # 每次输入间隔 30ms（<300ms 防抖窗口）
            self.assertEqual(spied.call_count, 0)  # 防抖窗口内不触发
            _pump(400)  # 越过防抖窗口
            self.assertEqual(spied.call_count, 1)  # 落定后恰好一次
        win._apply_search("")  # 清理界面态
        _flush_deferred()


class FlowTreeAsyncTest(unittest.TestCase):
    """目录扫描后台化 + 预览缓存命中。"""

    def _make_page(self, images: int = 120):
        _app()
        from xccv.flow_platform import FlowPage

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name) / "ds"
        root.mkdir()
        for i in range(images):
            cv2.imwrite(str(root / f"im{i:04d}.png"),
                        np.full((24, 32, 3), i % 255, np.uint8))
        page = FlowPage()
        self.addCleanup(page.close)
        self.addCleanup(_flush_deferred)
        page.data_edit.setText(str(root))
        page._load_data_root()
        deadline = time.monotonic() + 60
        while page._scan_thread is not None and time.monotonic() < deadline:
            _pump(20)
        _app().processEvents()
        return page, root

    def test_tree_filled_async(self):
        page, _root = self._make_page(120)
        self.assertIn("120/120", page.count_label.text())
        self.assertGreater(page.tree.topLevelItemCount(), 0)

    def test_preview_cache_hit(self):
        from unittest import mock

        page, root = self._make_page(3)
        # 找第一张图片项（可能直接在隐形根下，也可能在目录项下）
        img_item = None
        roots = [page.tree.topLevelItem(i)
                 for i in range(page.tree.topLevelItemCount())]
        for it in roots:
            kind, _val = it.data(0, _USER_ROLE) or ("", "")
            if kind == "img":
                img_item = it
                break
            for i in range(it.childCount()):
                kind, _val = it.child(i).data(0, _USER_ROLE) or ("", "")
                if kind == "img":
                    img_item = it.child(i)
                    break
            if img_item is not None:
                break
        self.assertIsNotNone(img_item)
        page.tree.setCurrentItem(img_item)
        with mock.patch("xccv.flow_platform.imread_unicode", wraps=__import__(
                "xccv.flow_platform", fromlist=["imread_unicode"]).imread_unicode) as spied:
            page._on_tree_click(img_item, 0)
            first_calls = spied.call_count
            page._on_tree_click(img_item, 0)  # 第二次点同一张：缓存命中
            self.assertEqual(spied.call_count, first_calls)  # 不再解码
        _app().processEvents()

if __name__ == '__main__':
    unittest.main()
