"""平台测试：2 个历史测试模块的合并模块。

由以下模块按域合并而成，用例逐段原样迁移、断言未改：
- test_v06_yolo_predict
- test_v06_defectgen
"""


# == 来源模块：test_v06_yolo_predict ==
"""YOLO-seg 预测掩码必须还原原图尺寸 + 可视化防御双保险。

背景：ultralytics YOLO-seg 返回的 masks
停在网络输入分辨率（letterbox 后 imgsz，如 1216），未还原原图（如 2144），
`render.draw_predictions` 内 `overlay[mask] = color` 抛 IndexError 整窗崩溃。

修复三层：
1. engine/models/yolo.py `_masks_to_original`：LetterBox 逆变换还原原图；
   返回 **list[(H,W) bool]**（Prediction.masks 契约；返回 3 维 ndarray 会让
   消费方 `if pred.masks:` 真值歧义抛 ValueError——实测反馈"YOLO 无法正常
   推理"；旧测试只断言 ndarray 形状且随机图零检出，把错契约锁死了）；
2. testing/render.py `_align_mask`：形状不符先尝试对齐，仍失败跳过+留痕；
3. testing_window._show_result：可视化异常降级为原图+状态提示，不再未捕获。
"""

import os
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import torch

from xccv.engine.models.base import Prediction
from xccv.engine.models.yolo import _masks_to_original
from xccv.testing.render import _align_mask, draw_predictions

_ROOT = Path(__file__).resolve().parents[1]
_SEG_WEIGHTS = _ROOT / "xccv" / "engine" / "weights" / "ultralytics" / "yolo11n-seg.pt"


class _FakeRes:
    """模拟 ultralytics Results：只暴露 masks.data（网络输入分辨率）。"""

    def __init__(self, masks):
        self.masks = mock.Mock()
        self.masks.data = masks


class MasksToOriginalTest(unittest.TestCase):
    def test_same_size_passthrough(self):
        data = (torch.rand(2, 32, 48) > 0.5).float()
        res = _FakeRes(data)
        out = _masks_to_original(res, (32, 48))
        self.assertIsInstance(out, list)  # Prediction.masks 契约
        self.assertEqual(len(out), 2)
        for m in out:
            self.assertEqual(m.shape, (32, 48))
            self.assertEqual(m.dtype, np.dtype(bool))

    def test_letterbox_restore(self):
        """网络输入 64x80，原图 32x64（纵向 letterbox）：输出必须等于原图尺寸。"""
        data = torch.zeros(1, 64, 80)
        data[0, 16:48, 8:72] = 1.0  # 中心区域（去 padding 后恰为原图范围）
        res = _FakeRes(data)
        out = _masks_to_original(res, (32, 64))
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].shape, (32, 64))
        # 中心区域映射回原图内部：掩码应非空且不超过原图范围
        self.assertTrue(out[0].any())

    def test_output_dtype_bool_2d(self):
        data = torch.rand(3, 20, 20)
        out = _masks_to_original(_FakeRes(data), (40, 40))
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 3)
        for m in out:
            self.assertEqual(m.ndim, 2)
            self.assertEqual(m.dtype, np.dtype(bool))


class AlignMaskDefenseTest(unittest.TestCase):
    """render._align_mask：双保险防御。"""

    def test_matching_shape_passthrough(self):
        mask = np.zeros((10, 12), bool)
        mask[2:5, 3:8] = True
        out = _align_mask(mask, (10, 12))
        self.assertTrue(out.dtype == np.dtype(bool))
        self.assertEqual(int(out.sum()), 15)

    def test_mismatch_aligned_by_resize(self):
        mask = np.zeros((16, 16), np.uint8)
        mask[4:12, 4:12] = 1
        out = _align_mask(mask, (32, 32))
        self.assertEqual(out.shape, (32, 32))
        self.assertTrue(out.any())

    def test_invalid_mask_returns_none(self):
        self.assertIsNone(_align_mask(np.zeros((0,), bool), (8, 8)))
        self.assertIsNone(_align_mask(np.zeros((2, 3, 4)), (8, 8)))


class DrawPredictionsDefenseTest(unittest.TestCase):
    """draw_predictions 对任何形状不符的掩码不得抛未捕获异常（双保险）。"""

    def test_wrong_size_mask_no_crash(self):
        img = np.zeros((64, 64, 3), np.uint8)
        mask = np.zeros((32, 32), bool)   # 与图像不符（模拟未还原的旧链路）
        mask[8:24, 8:24] = True
        pred = Prediction(labels=[0], scores=[0.9], boxes=[[10, 10, 30, 30]],
                          masks=[mask])
        out = draw_predictions(img, pred, ["a"], "segmentation")
        self.assertEqual(out.shape, img.shape)

    def test_garbage_mask_skipped(self):
        img = np.zeros((48, 48, 3), np.uint8)
        pred = Prediction(labels=[0, 1], scores=[0.9, 0.8],
                          boxes=[[5, 5, 20, 20], [22, 22, 40, 40]],
                          masks=[np.zeros((0,), bool), np.zeros((2, 3, 4))])
        out = draw_predictions(img, pred, ["a", "b"], "segmentation")
        self.assertEqual(out.shape, img.shape)

    def test_ndarray_masks_no_truth_ambiguity(self):
        """上游误给 3 维 ndarray（N>1）时不得因 `if masks:` 真值歧义抛错。"""
        img = np.zeros((40, 40, 3), np.uint8)
        masks = np.zeros((2, 40, 40), bool)
        masks[0, 2:10, 2:10] = True
        masks[1, 20:30, 20:30] = True
        pred = Prediction(labels=[0, 1], scores=[0.9, 0.8],
                          boxes=[[2, 2, 10, 10], [20, 20, 30, 30]], masks=masks)
        out = draw_predictions(img, pred, ["a", "b"], "segmentation")
        self.assertGreater(float(out[22:28, 22:28].mean()), 0)

    def test_legacy_adapter_image_size_ndarray_masks(self):
        from xccv.engine.flow.legacy_adapter import _image_size

        pred = Prediction(labels=[0, 1], scores=[0.9, 0.8],
                          boxes=[[0, 0, 4, 4], [1, 1, 5, 5]],
                          masks=np.zeros((2, 30, 50), bool))
        self.assertEqual(_image_size(pred, None), (50, 30))

    def test_correct_mask_still_rendered(self):
        img = np.zeros((64, 64, 3), np.uint8)
        mask = np.zeros((64, 64), bool)
        mask[10:30, 10:30] = True
        pred = Prediction(labels=[1], scores=[0.8], boxes=[[10, 10, 30, 30]],
                          masks=[mask])
        out = draw_predictions(img, pred, ["a", "b"], "segmentation")
        self.assertGreater(float(out[15:25, 15:25].mean()), 0)


@unittest.skipUnless(_SEG_WEIGHTS.is_file(), "缺少 yolo11n-seg 预训练权重")
class RealYoloSegSmokeTest(unittest.TestCase):
    """真实权重 CPU 冒烟：加载→推理→掩码形状==原图→渲染不抛错。"""

    def test_predict_masks_match_original(self):
        from xccv.engine.models.yolo import YoloModel
        from xccv.engine.dataset.base import TaskType

        model = YoloModel("yolo11n", TaskType.SEGMENTATION, 1,
                          device="cpu", use_pretrained=False)
        model.load_weights(_SEG_WEIGHTS)
        rng = np.random.default_rng(3)
        img = rng.integers(0, 255, size=(2144, 1216 // 2, 3), dtype=np.uint8)
        pred = model.predict(img, conf=0.2)
        if pred.masks is not None:  # 随机图可能零检出（真检出见 RealTrainedYoloSegTest）
            for m in pred.masks:
                self.assertEqual(m.shape, img.shape[:2])
        from xccv.testing.render import draw_predictions

        out = draw_predictions(img, pred, ["a"], "segmentation")
        self.assertEqual(out.shape, img.shape)


_TRAINED_SEG = (_ROOT / "traintest" / "seg" / "模型" / "20260916_140728_yolo11n-seg"
                / "best.pt")
_D1 = _ROOT / "traintest" / "seg" / "数据集" / "D1"


@unittest.skipUnless(_TRAINED_SEG.is_file() and _D1.is_dir(),
                     "缺少用户训练的 yolo11n-seg 模型或 D1 数据集")
class RealTrainedYoloSegTest(unittest.TestCase):
    """实测反馈场景复现：自训 yolo11n-seg + D1 真图，**N>0 真检出**
    → masks 为 list 且逐个等于原图尺寸 → 渲染与测试平台结果展示均不报错。"""

    @classmethod
    def setUpClass(cls):
        from xccv.engine.io_utils import imread_unicode
        from xccv.engine.models import load_model

        cls.model = load_model(_TRAINED_SEG)
        img_path = sorted(_D1.rglob("*.jpg"))[0]
        cls.img = imread_unicode(str(img_path))
        cls.pred = cls.model.predict(cls.img, conf=0.25, iou=0.5)

    def test_real_detections_masks_contract(self):
        self.assertGreater(len(self.pred.boxes), 0, "应有真检出（复现前提）")
        self.assertIsInstance(self.pred.masks, list)
        self.assertEqual(len(self.pred.masks), len(self.pred.boxes))
        for m in self.pred.masks:
            self.assertEqual(m.shape, self.img.shape[:2])
            self.assertEqual(m.dtype, np.dtype(bool))

    def test_real_detections_render_and_show(self):
        out = draw_predictions(self.img, self.pred, list(self.model.classes),
                               "segmentation")
        self.assertEqual(out.shape, self.img.shape)
        from PyQt5.QtWidgets import QApplication

        from xccv.testing.testing_window import TestingWindow

        _ = QApplication.instance() or QApplication([])
        win = TestingWindow()
        try:
            win.classes = list(self.model.classes)
            win.task = "segmentation"
            win._show_result("D1.jpg", self.img, self.pred)
            self.assertNotIn("可视化失败", win.result_stats.text())
        finally:
            win.close()
            win.deleteLater()


class ShowResultNarrowingTest(unittest.TestCase):
    """_show_result 渲染异常降级：状态栏提示+不抛未捕获（crash 面收窄）。"""

    def test_render_failure_degrades(self):
        from PyQt5.QtWidgets import QApplication

        from xccv.testing.testing_window import TestingWindow

        _ = QApplication.instance() or QApplication([])
        win = TestingWindow()
        try:
            img = np.zeros((32, 32, 3), np.uint8)
            pred = Prediction(labels=[0], scores=[0.9], boxes=[[1, 1, 8, 8]])
            with mock.patch("xccv.testing.render.draw_predictions",
                            side_effect=RuntimeError("boom")):
                win._show_result("t.png", img, pred)  # 不应抛
            self.assertIn("可视化失败", win.result_stats.text())
        finally:
            win.close()
            win.deleteLater()

# == 来源模块：test_v06_defectgen ==
# -*- coding: utf-8 -*-
"""缺陷生成（DefectAugmentStudio 移植）测试。

三层结构：
- 后端契约（FastAPI 经真实 uvicorn port=0 起服）——任何平台都跑；
- 入口契约（launcher 映射 / 模型工具第四卡 / 闪屏图标 / 窗口模块纪律文本）
  ——offscreen 可跑（全程不导入 QtWebEngine）；
- Qt 壳生命周期（构建→动态端口→loadFinished→关窗停服→端口释放、双实例
  端口互异）——仅 QT_QPA_PLATFORM=windows 时跑294offscreen 无 GL
  上下文，QtWebEngine 不能 offscreen 自动化）。
"""
import contextlib
import json
import os
import socket
import threading
import time
import unittest
import urllib.request
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "xccv" / "defectgen" / "web"

import uvicorn  # noqa: E402

from xccv.defectgen.backend.app import create_app  # noqa: E402


_APP_HOLD = None  # QApplication 必须持引用，否则被 GC


def _ensure_app():
    """创建/复用 QApplication。AA_ShareOpenGLContexts 仅在非 offscreen 平台
    设置：offscreen 插件无 GL 上下文，带该属性构造 QApplication 会直接
    fail-fast（实测 0xC0000409）；offscreen 下也不会导入 QtWebEngine，
    不需要该属性。"""
    global _APP_HOLD
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            from PyQt5.QtCore import Qt

            QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
        app = QApplication([])
        _APP_HOLD = app
    return app


@contextlib.contextmanager
def _serve():
    """真实 uvicorn 起服（port=0 动态端口）；退出即 should_exit 停服。"""
    server = uvicorn.Server(uvicorn.Config(
        create_app(), host="127.0.0.1", port=0, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True, name="t-defectgen-uvicorn")
    th.start()
    port = None
    deadline = time.time() + 20
    while time.time() < deadline and port is None:
        try:
            if server.servers:
                port = int(server.servers[0].sockets[0].getsockname()[1])
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.05)
    if port is None:
        server.should_exit = True
        th.join(timeout=5)
        raise AssertionError("uvicorn 动态端口未就绪")
    try:
        yield port
    finally:
        server.should_exit = True
        th.join(timeout=10)


def _get(port, path, timeout=8):
    # 返回 HTTPMessage 本体（.get 大小写不敏感）；dict() 会固化 wire 上的
    # 小写键 "content-type"，导致 MIME 断言误判
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}",
                                timeout=timeout) as r:
        return r.status, r.headers, r.read()


class DefectgenBackendContractTest(unittest.TestCase):
    """后端契约：health / 首页 / 静态 MIME/ 路由清单。"""

    def test_health_and_index(self):
        with _serve() as port:
            st, _, body = _get(port, "/api/health")
            self.assertEqual(st, 200)
            self.assertTrue(json.loads(body.decode("utf-8"))["ok"])
            st, _, body = _get(port, "/")
            self.assertEqual(st, 200)
            self.assertIn(b"<html", body[:400].lower())

    def test_static_mime_lock_p187(self):
        """Windows 注册表把 .js 记成 text/plain 会让 ES module 白屏——锁定修复。"""
        js = sorted(p.name for p in (WEB_DIR / "js").glob("*.js"))
        self.assertGreaterEqual(len(js), 10)
        with _serve() as port:
            for name in js:
                st, headers, _ = _get(port, f"/js/{name}")
                self.assertEqual(st, 200, name)
                self.assertIn("javascript",
                              headers.get("Content-Type", "").lower(), name)
            for name in sorted(p.name for p in (WEB_DIR / "css").glob("*.css")):
                _, headers, _ = _get(port, f"/css/{name}")
                self.assertIn("css",
                              headers.get("Content-Type", "").lower(), name)

    def test_api_routes_registered(self):
        paths = {getattr(r, "path", "") for r in create_app().routes}
        for p in ("/api/health", "/api/extract", "/api/sam_status",
                  "/api/poisson_preview", "/api/paint_defect",
                  "/api/placement_map", "/api/composite",
                  "/api/batch/start", "/api/batch/status/{jid}",
                  "/api/batch/cancel/{jid}"):
            self.assertIn(p, paths, p)

    def test_sam_status_contract(self):
        with _serve() as port:
            st, _, body = _get(port, "/api/sam_status")
            self.assertEqual(st, 200)
            self.assertTrue(json.loads(body.decode("utf-8"))["grabcut"])


class DefectgenEntryContractTest(unittest.TestCase):
    """入口三件套 + 窗口模块纪律文本（不导入 QtWebEngine，offscreen 安全）。"""

    def test_launcher_mapping(self):
        from xccv.launcher import PLATFORMS

        exe, module = PLATFORMS["defectgen"]
        self.assertEqual(exe, "XCcv缺陷生成.exe")
        self.assertEqual(module, "xccv.defectgen.defectgen_window")

    def test_modeltools_card_exists(self):
        _ensure_app()
        from xccv.modeltools.modeltools_window import ModelToolsWindow, _ToolCard

        win = ModelToolsWindow()
        self.addCleanup(win.close)
        cards = [c for c in win.findChildren(_ToolCard)
                 if getattr(c, "tool_key", None) == "defectgen"]
        self.assertEqual(len(cards), 1)
        self.assertTrue(cards[0].isEnabled())

    def test_splash_icon_mapping(self):
        from xccv.platform_entry import _SPLASH_ICONS

        self.assertEqual(_SPLASH_ICONS["defectgen"], "defectgen.png")
        assets = ROOT / "xccv" / "assets"
        for f in ("defectgen.png", "defectgen.ico"):
            self.assertTrue((assets / f).is_file(), f)

    def test_window_module_discipline_text(self):
        """不导入模块，锁定 设计纪律点（动态端口/停服/页面销毁/GL 共享）。"""
        src = (ROOT / "xccv" / "defectgen" / "defectgen_window.py"
               ).read_text(encoding="utf-8")
        for token in ("AA_ShareOpenGLContexts", "port=0", "should_exit",
                      "deleteLater", "QTWEBENGINE_CHROMIUM_FLAGS",
                      "actual_port"):
            self.assertIn(token, src, token)


def _shell_ok():
    """壳测试运行前提（装饰期判断）：
    - 平台非 offscreen6052offscreen 无 GL 上下文）；
    - QApplication 尚未创建6102必须先导入窗口模块再建 app——gate 单进程
      discovery 里 runner/兄弟测试早已建 app 且无 AA 标志，此时导入
      QtWebEngine 必炸；兄弟模块 test_flowgraph.py 还会在 import 期把
      QT_QPA_PLATFORM 硬改成 "windows"，env 判断在装饰期不可靠）。"""
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        return False
    from PyQt5.QtWidgets import QApplication

    return QApplication.instance() is None


@unittest.skipUnless(_shell_ok(),
                     "QtWebEngine 壳需独立进程/独立平台（当前进程"
                     "已有 QApplication 或平台为 offscreen，均不能安全导入）")
class DefectgenWindowShellTest(unittest.TestCase):
    """Qt 壳完整生命周期（真实窗口，windows 平台专项）。"""

    @classmethod
    def setUpClass(cls):
        cls.app = _ensure_app()
        cls.app.setQuitOnLastWindowClosed(False)
        from xccv.defectgen.defectgen_window import DefectGenWindow

        cls.DGW = DefectGenWindow

    @staticmethod
    def _close_all(*wins):
        for w in wins:
            w.close()

    def _wait_port_loaded(self, win, timeout=30.0):
        state = {"loaded": False, "failed": False}

        def _on_load(ok):
            state["loaded" if ok else "failed"] = True

        win.page.loadFinished.connect(_on_load)
        port = None
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.app.processEvents()
            if port is None:
                port = win._server.actual_port()
            elif state["loaded"] or state["failed"]:
                break
            time.sleep(0.03)
        self.assertIsInstance(port, int, "动态端口未就绪")
        self.assertTrue(state["loaded"], "页面 loadFinished 失败")
        return port

    def _assert_port_closed(self, port, timeout=5.0):
        deadline = time.time() + timeout
        rc = 0
        while time.time() < deadline:
            s = socket.socket()
            s.settimeout(1.0)
            rc = s.connect_ex(("127.0.0.1", port))
            s.close()
            if rc != 0:
                return
            time.sleep(0.1)
        self.fail(f"端口 {port} 未释放（connect_ex={rc}）")

    def test_lifecycle_and_stop_on_close(self):
        win = self.DGW()
        win.show()
        self.addCleanup(self._close_all, win)
        port = self._wait_port_loaded(win)
        self.assertTrue(
            win.page.url().toString().startswith(f"http://127.0.0.1:{port}"))
        st, _, body = _get(port, "/api/health")
        self.assertEqual(st, 200)
        self.assertTrue(json.loads(body.decode("utf-8"))["ok"])
        win.close()
        self.app.processEvents()
        self._assert_port_closed(port)

    def test_two_instances_distinct_ports(self):
        """与流程图编辑器等同时打开无冲突的根保证：动态端口互异、互不影响。"""
        w1 = self.DGW()
        w1.show()
        w2 = self.DGW()
        w2.show()
        self.addCleanup(self._close_all, w1, w2)
        p1 = self._wait_port_loaded(w1)
        p2 = self._wait_port_loaded(w2)
        self.assertNotEqual(p1, p2)
        st, _, body = _get(p1, "/api/health")
        self.assertTrue(json.loads(body.decode("utf-8"))["ok"])
        st, _, body = _get(p2, "/api/health")
        self.assertTrue(json.loads(body.decode("utf-8"))["ok"])
        w1.close()
        self.app.processEvents()
        self._assert_port_closed(p1)
        st, _, body = _get(p2, "/api/health")
        self.assertEqual(st, 200, "实例 1 关闭不得影响实例 2")
        w2.close()
        self.app.processEvents()
        self._assert_port_closed(p2)

if __name__ == '__main__':
    unittest.main()
