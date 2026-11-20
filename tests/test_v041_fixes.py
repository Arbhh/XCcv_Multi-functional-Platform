"""v0.4.1 修复回归测试。

覆盖：工作台 Qt5 导航兼容、算子中文标题、xccv2 前缀嗅探、混图友好报错、
运行日志落盘、无图像输出节点快照叠加、纯传统流程打包/运行、ai_detection
旧链兼容、批量期间禁并发推理。
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2

from xccv.engine.io_utils import imwrite_unicode
import numpy as np

from xccv.modeltools.flow_session import (
    FlowSession, _normalize_types, node_library_payload,
)
from xccv.modeltools import flow_session as _fs

node_library_payload()  # 注册全部算子

# 运行日志重定向到临时目录（避免测试污染仓库 platform_data）
import tempfile as _tempfile

_fs._RUN_LOG_DIR = Path(_tempfile.gettempdir()) / "xccv_test_v041_logs"


def _app():
    from PyQt5.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


_WINDOW_HOLDER = []   # 测试窗口持引用：防测试进程尾部析构原生崩溃（坑 11 同族）


class WorkbenchNavTest(unittest.TestCase):
    """A1/G1 修复回归：主界面「传统平台」页挂载必须是 Qt5 兼容写法。

    根因：栈替换的 Qt6 专属 API 在 Qt 5.15 没有，实测首点「传统平台」
    即 AttributeError 崩溃。修复后为 removeWidget + setParent(None) + insertWidget。
    注：不构造真实 HomeWindow 做集成导航——重模块导入顺序会改变 sip 对
    顶层窗口的托管（测试进程内/退出期原生崩溃），改用逻辑级 + 源码级双重锁定。
    """

    def test_no_qt6_replace_widget_in_app(self):
        """源码级锁定：app.py 不得出现 Qt6-only 的 replaceWidget。"""
        src = (Path(__file__).resolve().parents[1] / "xccv" / "app.py").read_text(
            encoding="utf-8")
        self.assertNotIn("replaceWidget", src)

    def test_workbench_constructs_with_registry(self):
        """工作台本体可构建（等价于导航后首次挂载的实质内容）。

        注：本用例若与 testing_window 导入同进程会触发退出期原生崩溃
        （sip 清理顺序），工作台构建的完整覆盖在 test_batch20_workbench，
        这里仅以分类过滤下拉的选项完整性做纯逻辑校验。
        """
        from xccv.engine.flow.registry import all_categories, list_nodes

        cats = set(all_categories())
        self.assertLessEqual({"图像源", "预处理", "定位", "逻辑", "检出过滤", "AI"},
                             cats)
        self.assertGreaterEqual(len(list_nodes()), 30)


class ChineseTitleTest(unittest.TestCase):
    """B1/Q9：节点库/画布暴露中文名称，而非注册函数名。"""

    def test_titles_are_chinese(self):
        from xccv.engine.flow.registry import list_nodes

        titles = {n["name"]: n["title"] for n in list_nodes()}
        self.assertEqual(titles["gaussian_blur"], "高斯模糊")
        self.assertEqual(titles["threshold"], "二值化")
        self.assertEqual(titles["blob_analysis"], "斑点分析")
        self.assertEqual(titles["find_circle"], "圆定位")
        self.assertEqual(titles["template_match"], "模板匹配")
        self.assertEqual(titles["model_detection"], "目标检测模型")
        self.assertEqual(titles["model_instance_seg"], "实例分割模型")
        self.assertEqual(titles["model_classification"], "分类模型")
        self.assertEqual(titles["filter_confidence"], "置信度过滤")
        self.assertEqual(titles["filter_count_check"], "数量判定")
        self.assertEqual(titles["image_loader"], "读入图片")
        # 生产算子（34 个）title 全部不含下划线（排除漏翻译的函数名）；
        # 注册表是全局的，套件中测试注册的节点（ex_/rg_ 前缀等）不在检查范围
        prod = [n for n in list_nodes() if not n["name"].startswith(("rg_", "ex_",
                                                                     "lr_"))]
        for n in prod:
            self.assertNotIn("_", n["title"],
                             f"{n['name']} 的标题仍是函数名: {n['title']}")


class SniffXccv2Test(unittest.TestCase):
    """修复：画布保存的新格式（xccv2/ 前缀）被正确识别并拒绝误跑。"""

    def test_sniff_xccv2_prefix(self):
        from xccv.engine.flow.legacy_adapter import sniff_format

        self.assertEqual(sniff_format({"nodes": [
            {"id": 1, "type": "xccv2/gaussian_blur", "properties": {}}]}), "xcv_graph")

    def test_load_flow_rejects_canvas_saved_xcv(self):
        from xccv.engine import detection_flow as df

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "检出流程.json"
            p.write_text(json.dumps({"last_node_id": 1, "last_link_id": 0,
                                     "nodes": [{"id": 1, "type": "xccv2/threshold",
                                                "properties": {}}],
                                     "links": []}, ensure_ascii=False),
                         encoding="utf-8")
            with self.assertLogs("xccv.engine.flow.legacy", level="WARNING"):
                self.assertIsNone(df.load_flow(p))


class MixedGraphTest(unittest.TestCase):
    """混图（旧版节点 + 新算子）给中文人话报错，而非"类型未注册"。"""

    def test_mixed_graph_friendly_error(self):
        from xccv.modeltools.flow_session import FlowErrorMixed

        graph = {"nodes": [
            {"id": 1, "type": "xccv2/image_loader", "properties": {"path": ""}},
            {"id": 2, "type": "xccv/post_confidence", "properties": {"min": 0.3}}],
            "links": []}
        with self.assertRaises(FlowErrorMixed) as cm:
            FlowSession().run(graph)
        self.assertIn("xccv/post_confidence", str(cm.exception))
        self.assertIn("混用", str(cm.exception))


class RunLogTest(unittest.TestCase):
    """Q4：每次运行落一份日志到 platform_data/logs/流程运行日志。"""

    def test_run_writes_log_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "in.png"
            imwrite_unicode(Path(p), np.zeros((8, 8, 3), np.uint8))
            s = FlowSession()
            s.run({"nodes": [{"id": 1, "type": "image_loader",
                              "properties": {"path": str(p)}}], "links": []},
                  label="单测")
            logs = list(_fs._RUN_LOG_DIR.glob("flow_*.txt"))
            self.assertTrue(logs)
            text = logs[-1].read_text(encoding="utf-8")
            self.assertIn("来源：单测", text)
            self.assertIn("image_loader(#1)", text)

    def test_log_dir_kept_under_cap(self):
        _fs._RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
        for i in range(_fs._RUN_LOG_KEEP + 6):
            (_fs._RUN_LOG_DIR / f"flow_20260101_{i:06d}_000.txt").write_text("x",
                                                                             encoding="utf-8")
        FlowSession.write_run_log({"ok_count": 1, "total": 1, "errors": {},
                                   "logs": "", "warnings": []}, label="清理")
        remaining = list(_fs._RUN_LOG_DIR.glob("flow_*.txt"))
        self.assertLessEqual(len(remaining), _fs._RUN_LOG_KEEP)


class SnapshotOverlayTest(unittest.TestCase):
    """C2/C3 修复：无图像输出节点（blob/find_circle 等）有"输入+结果叠加"快照。"""

    def _run_blob(self, tmp: Path):
        import xccv.engine.flow.ops.locate  # noqa: F401
        from xccv.engine.flow.executor import Executor
        from xccv.engine.flow.node_base import RunContext

        img = np.zeros((60, 80), np.uint8)
        cv2.circle(img, (40, 30), 12, 255, -1)
        p = tmp / "blob.png"
        imwrite_unicode(Path(p), img)
        g = {"nodes": [
            {"id": 1, "type": "image_loader", "properties": {"path": str(p)}},
            {"id": 2, "type": "blob_analysis",
             "properties": {"threshold_mode": "otsu", "thresh": 128, "invert": False,
                            "min_area": 10, "max_area": 0, "circularity": 0.0}}],
            "links": [[1, 1, 0, 2, 0]]}
        ctx = RunContext(snapshot_store=_fs.SnapshotStore())
        Executor(g, ctx).run()
        return ctx.snapshot_store.get(2)

    def test_blob_snapshot_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            snap = self._run_blob(Path(tmp))
            self.assertIsNotNone(snap)
            # 斑点分析带图像直通输出 → 快照存输出图（键 image）
            self.assertIn("image", snap["images"])
            vis = snap["images"]["image"]
            self.assertEqual(vis.ndim, 3)   # 快照图统一转 BGR 三通道

    def test_overlay_draws_boxes_for_prediction(self):
        from xccv.engine.flow.snapshot import overlay_result
        from xccv.engine.models.base import Prediction

        img = np.zeros((40, 40, 3), np.uint8)
        pred = Prediction(labels=[0], scores=[0.9], boxes=[[5, 5, 30, 30]])
        vis = overlay_result(img, {"detection": pred})
        self.assertTrue((vis != img).any())   # 框画上去了


class PackRelaxationTest(unittest.TestCase):
    """F2/Q7：纯传统流程可打包（不含模型）；旧链无模型仍拦截。"""

    def _xcv_flow(self):
        return {"nodes": [
            {"id": 1, "type": "xccv2/image_loader", "properties": {"path": ""}},
            {"id": 2, "type": "xccv2/gaussian_blur", "properties": {}}],
            "links": [[1, 1, 0, 2, 0]]}

    def test_pure_traditional_passes_validation(self):
        from xccv.modeltools.flow_shared import validate_flow_for_pack

        self.assertEqual(validate_flow_for_pack(self._xcv_flow()), [])

    def test_legacy_graph_without_model_still_blocked(self):
        from xccv.modeltools.flow_shared import validate_flow_for_pack

        flow = {"nodes": [{"id": 1, "type": "xccv/post_confidence",
                           "properties": {"min": 0.3}}], "links": []}
        problems = validate_flow_for_pack(flow)
        self.assertTrue(any("没有模型节点" in p for p in problems))

    def test_xcv_model_node_without_weight_blocked(self):
        """新模型节点未选权重 → 打包校验拦截（C1：校验作用到新节点）。"""
        from xccv.modeltools.flow_shared import validate_flow_for_pack

        flow = {"nodes": [{"id": 1, "type": "xccv2/model_detection",
                           "properties": {}}], "links": []}
        problems = validate_flow_for_pack(flow)
        self.assertTrue(any("尚未选择权重文件" in p for p in problems))

    def test_xcv_bad_count_check_blocked(self):
        from xccv.modeltools.flow_shared import validate_flow_for_pack

        flow = {"nodes": [
            {"id": 1, "type": "xccv2/filter_count_check",
             "properties": {"min": 5, "max": 2}}],
            "links": []}
        problems = validate_flow_for_pack(flow)
        self.assertTrue(any("下限 5 大于上限 2" in p for p in problems))

    def test_xcv_bad_region_blocked(self):
        from xccv.modeltools.flow_shared import validate_flow_for_pack

        flow = {"nodes": [{"id": 1, "type": "xccv2/filter_region_keep",
                           "properties": {"regions": "1,2,3"}}],
                "links": []}
        problems = validate_flow_for_pack(flow)
        self.assertTrue(any("不是 4 个数字" in p for p in problems))

    def test_xcv_disconnected_blocked(self):
        from xccv.modeltools.flow_shared import validate_flow_for_pack

        flow = {"nodes": [
            {"id": 1, "type": "xccv2/image_loader", "properties": {"path": ""}},
            {"id": 2, "type": "xccv2/gaussian_blur", "properties": {}}],
            "links": []}   # 无连线：blur 的必需输入缺失 → 结构问题
        problems = validate_flow_for_pack(flow)
        self.assertTrue(any("结构问题" in p or "缺少" in p for p in problems))

    def test_xvap_roundtrip_without_models(self):
        from xccv.engine import flow_package as fp

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pure.xvap"
            fp.save_xvap(path, self._xcv_flow(), {})
            flow, models = fp.load_xvap(path)
            self.assertEqual(len(flow["nodes"]), 2)
            self.assertEqual(models, {})


class FlowPageXcvRunTest(unittest.TestCase):
    """流程图平台页的新算子流程运行路由（Q7 修复）。"""

    def test_flow_page_runs_xcv_flow(self):
        """Q7：流程图平台页「运行流程」对纯传统流程走新引擎并回写状态栏。

        用 MagicMock 作 self 调未绑定方法（覆盖路径注入/运行/状态消息逻辑）；
        不构造真实 FlowPage——重 Qt 窗口在测试进程的析构顺序会触发原生崩溃。
        """
        _app()
        from xccv.flow_platform import FlowPage

        with tempfile.TemporaryDirectory() as tmp:
            img = np.zeros((60, 80, 3), np.uint8)
            img[20:40, 30:60] = 255
            img_path = Path(tmp) / "目标.png"
            from xccv.engine.io_utils import imwrite_unicode

            imwrite_unicode(img_path, img)   # cv2.imwrite 中文文件名静默失败（P265 复验）
            flow = {"last_node_id": 2, "last_link_id": 1, "nodes": [
                {"id": 1, "type": "xccv2/image_loader", "pos": [0, 0],
                 "properties": {"path": "", "is_grayscale": False}},
                {"id": 2, "type": "xccv2/gaussian_blur", "pos": [1, 0],
                 "properties": {"ksize": 3, "sigma": 0.0}}],
                "links": [[1, 1, 0, 2, 0]]}
            flow_path = Path(tmp) / "flow.json"
            flow_path.write_text(json.dumps(flow, ensure_ascii=False),
                                 encoding="utf-8")

            page = mock.MagicMock()
            FlowPage._run_flow_xcv(page, flow, img_path)
            msg = page.window.return_value.statusBar.return_value \
                .showMessage.call_args[0][0]
            self.assertIn("2/2 节点成功", msg)
            self.assertIn("流程运行日志", msg)
            page._show_mat.assert_called_once()   # 可视化已回显


class BatchConcurrencyGuardTest(unittest.TestCase):
    """防护：批量运行期间禁用 UI 线程单图推理（同模型跨线程并发）。

    用 MagicMock 作 self 调未绑定方法——不实例化 PyQt 对象（未初始化的
    PyQt 子类桩被 GC 时会触发 sip 原生崩溃，且崩在后续用例里极难定位）。
    """

    def test_infer_blocked_during_batch(self):
        from xccv.testing.testing_window import TestingWindow

        fake = mock.MagicMock()
        fake._xcv_flow = None                    # 非传统流程模式
        fake._current_image_path.return_value = None   # 非批量分支取不到图即返回
        fake.model = None
        TestingWindow._infer_current(fake)          # 无模型 → 早退
        fake.result_stats.setText.assert_called_once_with("未加载模型")
        fake._current_image_path.assert_not_called()

        fake.model = object()
        fake._batch_thread = object()               # 批量运行中 → 守卫拦截
        TestingWindow._infer_current(fake)
        fake._current_image_path.assert_not_called()

        fake._batch_thread = None
        TestingWindow._infer_current(fake)          # 非批量 → 正常走到取图
        fake._current_image_path.assert_called_once()


if __name__ == "__main__":
    unittest.main()
