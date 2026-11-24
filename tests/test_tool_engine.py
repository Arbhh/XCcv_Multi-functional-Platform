"""平台测试：3 个历史测试模块的合并模块。

由以下模块按域合并而成，用例逐段原样迁移、断言未改：
- test_v06_p06_engine_load
- test_v06_p07_engine_meta
- test_v06_trt_fallback
"""


# == 来源模块：test_v06_p06_engine_load ==
# -*- coding: utf-8 -*-
"""复测回归：.engine/.engine 加载崩溃。

YOLO() 加载 TensorRT 引擎后 model.model 是引擎元数据字符串（str）而非
nn.Module，load_weights 无保护地 .to(device) 崩溃
AttributeError: 'str' object has no attribute 'to'。
修复=仅对 nn.Module 做 device 迁移。
回归用例：mock YOLO 返回 model.model=str，load_weights 不再崩溃且
num_classes/classes 正常来自 model.names。
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xccv.engine.dataset.base import TaskType  # noqa: E402


class EngineLoadWeightsTest(unittest.TestCase):
    """P-06：.engine 元数据 str 不再做 .to(device)。"""

    def _load_with(self, model_attr):
        from xccv.engine.models.yolo import YoloModel

        m = YoloModel.__new__(YoloModel)
        m.device = "cpu"
        m.num_classes = 0
        m.classes = []
        m.task = TaskType.DETECTION

        fake_names = {0: "黑点", 1: "异色"}
        fake = type("FakeYOLO", (), {})()
        fake.model = model_attr
        fake.names = fake_names
        fake.task = "segment"
        with patch("xccv.engine.models.yolo.YOLO", return_value=fake):
            m.load_weights(Path("dummy.pt"))
        return m

    def test_engine_string_model_no_crash(self):
        m = self._load_with("TRT engine metadata string")
        self.assertEqual(m.num_classes, 2)
        self.assertEqual(m.classes, ["黑点", "异色"])
        self.assertEqual(m.task, TaskType.SEGMENTATION)

    def test_pt_module_still_moved(self):
        import torch

        moved = []
        net = torch.nn.Linear(2, 2)
        orig_to = net.to

        def spy(*a, **k):
            moved.append(a)
            return orig_to(*a, **k)

        net.to = spy
        m = self._load_with(net)
        self.assertEqual(len(moved), 1)  # nn.Module 仍执行 device 迁移
        self.assertEqual(m.num_classes, 2)

# == 来源模块：test_v06_p07_engine_meta ==
"""P-07 回归：矩形/分割模型 .engine 引擎链路（容器头 task/classes/imgsz 回填）。

背景：ultralytics 直接加载 .engine 丢 task（分割引擎按检测解析→大量异常框），
且导出请求尺寸被 stride 取整后未记录。修复：容器头记录引擎真实输入形状，
加载侧 apply_engine_meta 回填 task/classes/imgsz，predict 按引擎形状 letterbox。
"""

import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _fake_yolo_model(task="detect", names=None):
    """ ultralytics YOLO 加载结果的最小替身（overrides 在 predict 时生效）。"""
    return types.SimpleNamespace(task=task,
                                 overrides={},
                                 names=names or {0: "all"},
                                 model="engine-metadata-str")


def _fake_result():
    return types.SimpleNamespace(boxes=None, masks=None)


class ApplyEngineMetaTest(unittest.TestCase):
    """apply_engine_meta：容器头 → YoloModel 状态回填。"""

    def _make_model(self):
        from xccv.engine.dataset.base import TaskType
        from xccv.engine.models.yolo import YoloModel

        m = YoloModel.__new__(YoloModel)  # 跳过权重构建，仅测元数据回填
        m.device = "cpu"
        m.model = _fake_yolo_model()
        m.num_classes = 1
        m.classes = ["all"]
        m.task = TaskType.DETECTION
        return m

    def test_segment_meta_backfills_task_and_names(self):
        from xccv.engine.dataset.base import TaskType

        m = self._make_model()
        m.apply_engine_meta({"kind": "yolo", "task": "segmentation",
                             "classes": ["黑点", "异色"], "imgsz": [608, 1952]})
        self.assertIs(m.task, TaskType.SEGMENTATION)
        self.assertEqual(m.model.task, "segment")
        self.assertEqual(m.model.overrides["task"], "segment")
        self.assertEqual(m.num_classes, 2)
        self.assertEqual(m.classes, ["黑点", "异色"])
        self.assertEqual(m.model.names, {0: "黑点", 1: "异色"})
        self.assertEqual(m._engine_imgsz, [608, 1952])
        self.assertEqual(m.model.overrides["imgsz"], [608, 1952])

    def test_predict_passes_engine_imgsz(self):
        from xccv.engine.dataset.base import TaskType
        from xccv.engine.models.yolo import YoloModel

        m = self._make_model()
        calls = {}

        def fake_predict(img, **kw):
            calls.update(kw)
            return [_fake_result()]

        m.model.predict = fake_predict
        m.apply_engine_meta({"task": "segmentation", "classes": ["a"],
                             "imgsz": [640, 640]})
        import numpy as np

        m.predict(np.zeros((32, 32, 3), dtype=np.uint8))
        self.assertEqual(calls.get("imgsz"), [640, 640])
        # 未回填引擎几何的 .pt 直载路径：不传 imgsz（走 ultralytics 默认）
        m2 = self._make_model()
        m2._engine_imgsz = None
        calls2 = {}

        def fake_predict2(img, **kw):
            calls2.update(kw)
            return [_fake_result()]

        m2.model.predict = fake_predict2
        m2.predict(np.zeros((32, 32, 3), dtype=np.uint8))
        self.assertNotIn("imgsz", calls2)

    def test_empty_meta_noop(self):
        m = self._make_model()
        before = (m.task, m.num_classes, dict(m.model.overrides))
        m.apply_engine_meta(None)
        m.apply_engine_meta({})
        self.assertEqual((m.task, m.num_classes, dict(m.model.overrides)), before)


class ResolveYoloImgszTest(unittest.TestCase):
    """_resolve_yolo_imgsz：显式参数优先，缺省回源训练尺寸（矩形保留）。"""

    def test_explicit_passthrough(self):
        from xccv.engine.model_export import _resolve_yolo_imgsz

        y = types.SimpleNamespace(overrides={"imgsz": [586, 1920]})
        self.assertEqual(_resolve_yolo_imgsz(640, y), 640)
        self.assertEqual(_resolve_yolo_imgsz((480, 800), y), (480, 800))

    def test_fallback_to_overrides(self):
        from xccv.engine.model_export import _resolve_yolo_imgsz

        self.assertEqual(_resolve_yolo_imgsz(None, types.SimpleNamespace(
            overrides={"imgsz": [586, 1920]})), [586, 1920])
        self.assertEqual(_resolve_yolo_imgsz(None, types.SimpleNamespace(
            overrides={"imgsz": 736})), 736)
        self.assertEqual(_resolve_yolo_imgsz(None, types.SimpleNamespace(
            overrides={})), 640)


class ExportMetaImgszTest(unittest.TestCase):
    """_export_yolo_engine：引擎真实输入形状写入 meta（mock ultralytics）。"""

    def test_meta_records_resolved_imgsz(self):
        import xccv.engine.model_export as me

        exported = {}

        class FakeYOLO:
            def __init__(self, path):
                self.overrides = {"imgsz": [586, 1920]}

            def export(self, **kw):
                exported.update(kw)
                import os

                fd, p = tempfile.mkstemp(suffix=".engine")
                os.close(fd)  # 立即关句柄：Windows 下未关闭会阻塞后续 unlink
                out = Path(p)
                out.write_bytes(b"fake-engine")
                return str(out)

        orig = None
        import ultralytics

        orig = ultralytics.YOLO
        ultralytics.YOLO = FakeYOLO
        try:
            meta = {"kind": "yolo", "task": "segmentation", "classes": ["a"]}
            me._export_yolo_engine(Path("m.pt"), Path("out.engine"), None,
                                   "cuda", half=False, meta=meta)
        finally:
            ultralytics.YOLO = orig
        self.assertEqual(exported["imgsz"], [586, 1920])
        self.assertEqual(meta["imgsz"], [586, 1920])  # 反序列化失败时退回请求尺寸
        self.assertEqual(meta["task"], "segmentation")


class SuggestParseImgszTest(unittest.TestCase):
    """_suggest_imgsz 矩形返回 (高, 宽)；_parse_imgsz 双格式解析。"""

    def test_suggest_rect_from_train_args(self):
        from xccv.modeltools.modeltools_window import _suggest_imgsz

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.pt"
            torch.save({"train_args": {"imgsz": [586, 1920]}}, p)
            self.assertEqual(_suggest_imgsz(p), (586, 1920))
            p2 = Path(d) / "s.pt"
            torch.save({"train_args": {"imgsz": [640, 640]}}, p2)
            self.assertEqual(_suggest_imgsz(p2), 640)
            p3 = Path(d) / "old.pt"
            torch.save({"train_args": {"imgsz": 512}}, p3)
            self.assertEqual(_suggest_imgsz(p3), 512)

    def test_parse_imgsz(self):
        from xccv.modeltools.modeltools_window import _parse_imgsz

        self.assertEqual(_parse_imgsz("640"), 640)
        self.assertEqual(_parse_imgsz(" 586x1920 "), (586, 1920))
        self.assertEqual(_parse_imgsz("608×1952"), (608, 1952))
        self.assertEqual(_parse_imgsz("640*640"), 640)
        self.assertIsNone(_parse_imgsz("abc"))
        self.assertIsNone(_parse_imgsz("31"))
        self.assertIsNone(_parse_imgsz("999999"))
        self.assertIsNone(_parse_imgsz(""))
        self.assertIsNone(_parse_imgsz("1x2x3"))

# == 来源模块：test_v06_trt_fallback ==
""".engine（TensorRT）在无 GPU/无 TensorRT 环境友好降级。

加载 .engine 时缺 tensorrt 模块此前抛裸 ModuleNotFoundError 堆栈；现转译为
TrtUnavailableError 中文指引（测试平台对话框/日志 INFO，不裸堆栈）。
"""

import os
import tempfile
import unittest

from xccv.engine.trt_runtime import local_compute_capability
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class TrtUnavailableErrorTest(unittest.TestCase):
    def test_trt_engine_init_friendly_error(self):
        """TrtEngine 构造在 import tensorrt 失败时抛中文指引。"""
        import builtins

        from xccv.engine import trt_runtime

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "tensorrt":
                raise ModuleNotFoundError("No module named 'tensorrt'")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(trt_runtime.TrtUnavailableError) as ctx:
                trt_runtime.TrtEngine(b"\x00" * 16)
        msg = str(ctx.exception)
        self.assertIn("TensorRT", msg)
        self.assertIn("GPU", msg)
        self.assertIn(".pt", msg)  # 指引替代格式

    def test_yolo_load_weights_translates_tensorrt_error(self):
        """ultralytics 加载 .engine 缺 tensorrt 时同样转译（YOLO 型 .engine 链路）。"""
        from xccv.engine.models.yolo import YoloModel
        from xccv.engine.trt_runtime import TrtUnavailableError

        m = YoloModel("yolo11n", "detection", 1, device="cpu", use_pretrained=False)
        err = ModuleNotFoundError("No module named 'tensorrt'")
        with mock.patch("xccv.engine.models.yolo.YOLO", side_effect=err):
            with self.assertRaises(TrtUnavailableError):
                m.load_weights("fake.engine")

    def test_yolo_load_weights_reraises_other_import_errors(self):
        """与 tensorrt 无关的 ModuleNotFoundError 不应被吞掉转译。"""
        from xccv.engine.models.yolo import YoloModel

        m = YoloModel("yolo11n", "detection", 1, device="cpu", use_pretrained=False)
        err = ModuleNotFoundError("No module named 'something_else'")
        with mock.patch("xccv.engine.models.yolo.YOLO", side_effect=err):
            with self.assertRaises(ModuleNotFoundError) as ctx:
                m.load_weights("fake.pt")
        self.assertIn("something_else", str(ctx.exception))

    def test_model_load_thread_friendly_message(self):
        """测试平台加载线程对 TrtUnavailableError 只给纯文案（无异常类名前缀）。"""
        from PyQt5.QtCore import QCoreApplication

        from xccv.engine.trt_runtime import TrtUnavailableError
        from xccv.testing import testing_window

        _ = QCoreApplication.instance() or QCoreApplication([])
        captured = {}
        thread = testing_window._ModelLoadThread("whatever.pt")
        thread.failed.connect(lambda msg: captured.update(msg=msg))
        with mock.patch("xccv.engine.models.load_model",
                        side_effect=TrtUnavailableError("需要 GPU 环境")):
            thread.run()
        self.assertEqual(captured.get("msg"), "需要 GPU 环境")

    def test_model_load_thread_other_errors_keep_class_name(self):
        from PyQt5.QtCore import QCoreApplication

        from xccv.testing import testing_window

        _ = QCoreApplication.instance() or QCoreApplication([])
        captured = {}
        thread = testing_window._ModelLoadThread("whatever.pt")
        thread.failed.connect(lambda msg: captured.update(msg=msg))
        with mock.patch("xccv.engine.models.load_model",
                        side_effect=ValueError("bad file")):
            thread.run()
        self.assertEqual(captured.get("msg"), "ValueError: bad file")



_ROOT = Path(__file__).resolve().parents[1]
_REAL_XCS = (_ROOT / "traintest" / "seg" / "模型" / "20260913_150930_maskrcnn"
             / "best.engine")


class TrtNoCudaFallbackTest(unittest.TestCase):
    """实测反馈 3.1：venv 装有 tensorrt 10.x，但本机无 CUDA 设备——
    `import tensorrt` 成功、`trt.Runtime()` 构造返回 nullptr（pybind11
TypeError 裸弹窗）。此前只拦截 ModuleNotFoundError，漏掉这条路。"""

    def test_registry_xcs_without_cuda_friendly(self):
        """registry 先验 CUDA：无设备直接友好降级（不触 TensorRT）。"""
        from xccv.engine.models import registry
        from xccv.engine.trt_runtime import TrtUnavailableError

        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(TrtUnavailableError) as ctx:
                registry.load_model(Path("不存在也无妨.engine"))
        self.assertIn("GPU", str(ctx.exception))
        self.assertNotIn("pybind11", str(ctx.exception))

    def test_trt_engine_runtime_nullptr_translated(self):
        """CUDA 声称可用但 Runtime 构造失败（驱动/设备异常）同样友好降级。"""
        import sys
        import types

        from xccv.engine import trt_runtime

        fake = types.ModuleType("tensorrt")

        class _Logger:
            WARNING = 2

            def __init__(self, *a):
                pass

        def _runtime(*a):
            raise TypeError("pybind11::init(): factory function returned nullptr")

        fake.Logger = _Logger
        fake.Runtime = _runtime
        with mock.patch.dict(sys.modules, {"tensorrt": fake}),                 mock.patch("torch.cuda.is_available", return_value=True):
            with self.assertRaises(trt_runtime.TrtUnavailableError):
                trt_runtime.TrtEngine(b"\x00" * 16)

    def test_trt_engine_no_cuda_translated(self):
        from xccv.engine import trt_runtime

        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(trt_runtime.TrtUnavailableError):
                trt_runtime.TrtEngine(b"\x00" * 16)

    @unittest.skipUnless(_REAL_XCS.is_file(), "缺少实测反馈场景的 best.engine")
    def test_real_xcs_user_scenario(self):
        """复现实测操作：测试平台加载线程 + 真 best.engine（本机无 GPU）。"""
        import torch

        if torch.cuda.is_available():
            self.skipTest("本机有 CUDA，非无 GPU 场景")
        from PyQt5.QtCore import QCoreApplication

        from xccv.testing import testing_window

        _ = QCoreApplication.instance() or QCoreApplication([])
        captured = {}
        thread = testing_window._ModelLoadThread(_REAL_XCS)
        thread.failed.connect(lambda msg: captured.update(msg=msg))
        thread.done.connect(lambda m: captured.update(model=m))
        thread.run()  # 同步执行 run()，与实测加载线程同一路径
        msg = captured.get("msg", "")
        self.assertIn("TensorRT", msg)
        self.assertIn("GPU", msg)
        self.assertNotIn("pybind11", msg)
        self.assertNotIn("TypeError", msg)  # 纯文案，无异常类名前缀


_TRT_COMPUTE_LOG = ("deserializeCudaEngine: Deserialize engine quad SPD \\"
                    "incompatible device, expecting compute 7.5 got compute 8.9, "
                    "please rebuild.")


class ComputeMismatchTest(unittest.TestCase):
    """加速模型跨显卡（算力不同）→ 中文指引而非"文件损坏"。"""

    def test_parse_compute_mismatch_from_trt_log(self):
        from xccv.engine.trt_runtime import _parse_compute_mismatch

        hit = _parse_compute_mismatch([_TRT_COMPUTE_LOG])
        self.assertEqual(hit, ((7, 5), (8, 9)))
        self.assertIsNone(_parse_compute_mismatch(["一切正常", "no match"]))

    def test_mismatch_message_contains_both_cc_and_hint(self):
        from xccv.engine.trt_runtime import (
            _parse_compute_mismatch, compute_mismatch_from_text,
        )

        expect, built = _parse_compute_mismatch([_TRT_COMPUTE_LOG])
        msg = compute_mismatch_from_text(_TRT_COMPUTE_LOG)
        self.assertIn(f"算力 {built[0]}.{built[1]}", msg)
        self.assertIn(f"算力 {expect[0]}.{expect[1]}", msg)
        self.assertIn("重新生成加速模型", msg)
        self.assertIn("没有损坏", msg)
        self.assertIsNone(compute_mismatch_from_text("维度不匹配 shape mismatch"))

    def test_meta_precheck_detects_cross_gpu(self):
        """引擎元数据记录的算力与本机不同 → 加载前即拦（不必反序列化）。"""
        import torch

        from xccv.engine.trt_runtime import engine_compute_mismatch

        with mock.patch("torch.cuda.is_available", return_value=True),             mock.patch("torch.cuda.get_device_capability",
                      return_value=(7, 5)):
            msg = engine_compute_mismatch({"kind": "rcnn", "cc_build": [8, 9]})
            self.assertIn("8.9", msg)
            self.assertIn("7.5", msg)
            self.assertIsNone(engine_compute_mismatch({"kind": "rcnn",
                                                    "cc_build": [7, 5]}))
            self.assertIsNone(engine_compute_mismatch({"kind": "rcnn"}))
            self.assertIsNone(engine_compute_mismatch(None))
        with mock.patch("torch.cuda.is_available", return_value=False):
            self.assertIsNone(engine_compute_mismatch({"cc_build": [8, 9]}))

    @unittest.skipUnless(_REAL_XCS.is_file(), "缺少实测反馈场景的 best.engine")
    def test_real_xcs_cross_gpu_friendly(self):
        """原机（sm_89）生成的真 best.engine 在本机（sm_75）：加载 → 中文跨显卡
        指引（TrtComputeMismatchError，TrtUnavailableError 子类=纯文案路径）。

        严格断言子类：曾因捕获 Logger 构造报 TypeError 被 nullptr 转译吞成
        "无 GPU" 通用文案，而宽松断言（接受父类）让测试假绿。
        """
        import torch

        if not torch.cuda.is_available():
            self.skipTest("本机无 CUDA，属无 GPU 场景用例")
        from xccv.engine.models import registry
        from xccv.engine.trt_runtime import TrtComputeMismatchError, local_compute_capability

        if local_compute_capability() == (8, 9):
            self.skipTest("本机即生成机型（sm_89），不构成跨显卡场景")
        with self.assertRaises(TrtComputeMismatchError) as ctx:
            registry.load_model(_REAL_XCS)
        msg = str(ctx.exception)
        self.assertIn(".engine", msg)
        self.assertIn("8.9", msg)              # 生成机型算力（来自 TRT 日志）
        self.assertIn("重新生成加速模型", msg)
        self.assertNotIn("pybind11", msg)
        self.assertNotIn("当前环境不可用", msg)  # 不得退化为"无 GPU"通用文案

    def test_capture_logger_constructs_on_real_trt(self):
        """真 TensorRT 下捕获 Logger 可构造并能收日志（ILogger 签名随版本变）。"""
        try:
            import tensorrt as trt
        except Exception:
            self.skipTest("未安装 tensorrt")
        from xccv.engine.trt_runtime import _make_capture_logger

        lg = _make_capture_logger(trt)
        lg.log(trt.ILogger.ERROR, "boom expecting compute 7.5 got compute 8.9")
        lg.log(trt.ILogger.INFO, "verbose noise")
        self.assertEqual(len(lg.messages), 1)
        self.assertIn("expecting compute", lg.messages[0])

    @unittest.skipUnless(local_compute_capability(),
                         "需要 NVIDIA GPU（cc_build 记录依赖显卡算力）")
    def test_write_engine_records_build_cc(self):
        """导出落盘时元数据记录生成算力/TRT 版本（回读可预检）。"""
        import json

        from xccv.engine import model_export as me

        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / "out.engine"
            me._write_engine(b"fake-engine",
                             dst, meta={"kind": "classifier"})
            meta, engine, _extra = me.unpack_engine_blob(dst.read_bytes())
            self.assertEqual(engine, b"fake-engine")
            self.assertIn("cc_build", meta)
            self.assertEqual(len(meta["cc_build"]), 2)
            self.assertIn("trt_build", meta)
            json.dumps(meta)  # 必须可序列化（落盘前已 json.dumps，此处双保险）

    def test_registry_yolo_branch_translates_cross_gpu_text(self):
        """无算力记录的旧 YOLO .engine：ultralytics 抛跨显卡错误 → 转译为中文指引。"""
        import torch

        if not torch.cuda.is_available():
            self.skipTest("YOLO 引擎链路需 CUDA 设备")
        from xccv.engine.models import registry
        from xccv.engine.trt_runtime import TrtComputeMismatchError

        fd, tmp = tempfile.mkstemp(suffix=".engine")
        with os.fdopen(fd, "wb") as f:
            f.write(b"\x00" * 32)   # 无 XCVE 魔数 → YOLO 纯引擎
        eng = Path(tmp)
        self.addCleanup(eng.unlink, True)
        with mock.patch("xccv.engine.models.registry.YoloModel") as ym:
            ym.return_value.load_weights.side_effect = RuntimeError(
                "trt deserialize: expecting compute 7.5 got compute 8.9")
            with self.assertRaises(TrtComputeMismatchError) as ctx:
                registry.load_model(eng)
        self.assertIn("重新生成加速模型", str(ctx.exception))

if __name__ == '__main__':
    unittest.main()
