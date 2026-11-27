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

# == 工具脚本契约：环境安装器 / 权重下载器 / 运行时接管 / 文档对账 ==
# 双态契约：无网 / 无权重环境不挂——下载与探测全部经注入替身，不真实联网。
import contextlib
import io
import os
import subprocess as _subprocess
import tempfile as _tempfile
from unittest import mock

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))


def _fake_nvidia_smi(stdout: str, returncode: int = 0):
    """注入 detect_gpu 的 subprocess.run 替身（假 nvidia-smi 输出）。"""

    def _run(cmd, capture_output=True, text=True, timeout=15):
        return _subprocess.CompletedProcess(cmd, returncode,
                                            stdout=stdout, stderr="")

    return _run


class SetupEnvTorchSourceTest(unittest.TestCase):
    """安装器：GPU 探测与 torch 源选择（含 ultralytics torch<2.10 上界约束）。"""

    def test_detect_gpu_parses_query(self):
        import setup_env

        runner = _fake_nvidia_smi('"NVIDIA RTX 4070", "550.54", "8.9"\n')
        gpu = setup_env.detect_gpu(runner=runner)
        self.assertEqual(gpu["name"], "NVIDIA RTX 4070")
        self.assertEqual(gpu["driver"], 550)
        self.assertEqual(gpu["compute_cap"], 89)

    def test_detect_gpu_none_on_failure(self):
        import setup_env

        def _boom(*_a, **_k):
            raise OSError("nvidia-smi not executable")

        self.assertIsNone(setup_env.detect_gpu(
            runner=_fake_nvidia_smi("", returncode=1)))
        self.assertIsNone(setup_env.detect_gpu(runner=_boom))

    def test_source_cpu_without_gpu(self):
        import setup_env

        name, url, _reason = setup_env.select_torch_index(None)
        self.assertEqual(name, "cpu")
        self.assertEqual(url, setup_env.TORCH_INDEXES["cpu"])

    def test_source_cu128_for_blackwell(self):
        import setup_env

        gpu = {"name": "RTX 5070", "driver": 576, "compute_cap": 120}
        name, url, _reason = setup_env.select_torch_index(gpu)
        self.assertEqual(name, "cu128")

    def test_source_cu126_for_mid_gpu(self):
        import setup_env

        gpu = {"name": "RTX 4070", "driver": 550, "compute_cap": 89}
        name, _url, _reason = setup_env.select_torch_index(gpu)
        self.assertEqual(name, "cu126")

    def test_source_cu128_for_new_driver(self):
        import setup_env

        gpu = {"name": "RTX 4070", "driver": 576, "compute_cap": 89}
        name, _url, _reason = setup_env.select_torch_index(gpu)
        self.assertEqual(name, "cu128")

    def test_source_cpu_for_old_driver(self):
        import setup_env

        gpu = {"name": "GTX 1060", "driver": 472, "compute_cap": 61}
        name, _url, _reason = setup_env.select_torch_index(gpu)
        self.assertEqual(name, "cpu")

    def test_source_override_name_and_url(self):
        import setup_env

        name, url, _reason = setup_env.select_torch_index(None, "cu126")
        self.assertEqual((name, url), ("cu126", setup_env.TORCH_INDEXES["cu126"]))
        name, url, _reason = setup_env.select_torch_index(
            None, "https://mirror.example/whl")
        self.assertEqual((name, url), ("custom", "https://mirror.example/whl"))

    def test_torch_210_rejected_by_upper_bound(self):
        import setup_env

        with self.assertRaises(SystemExit):
            setup_env.select_torch_index(None, "", "2.10.0")


class DownloadWeightsContractTest(unittest.TestCase):
    """权重下载器：清单 / 注入下载 / 已就绪跳过 / 失败分支（不真实联网）。"""

    def _write_zip_head(self, url, dst, progress=None):
        dst.write_bytes(b"PK\x03\x04" + b"\x00" * 2048)

    def _write_junk(self, url, dst, progress=None):
        dst.write_bytes(b"junk")

    def test_list_prints_families_and_targets(self):
        import download_weights

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            download_weights._list_all()
        out = buf.getvalue()
        for fam in ("yolo", "sam", "dino", "rcnn"):
            self.assertIn(fam, out)
        self.assertIn("dino_vitbase8_pretrain.pth", out)

    def test_download_generic_writes_and_skips_existing(self):
        import download_weights

        calls = []

        def fetcher(url, dst, progress=None):
            calls.append(url)
            self._write_zip_head(url, dst, progress)

        with _tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "w" / "fake.pt"
            state = {}
            result = download_weights.download_generic(
                "fake.pt", dest, ["https://a/x", "https://b/x"], 1000,
                True, state, fetcher)
            self.assertEqual(result, "新下载完成")
            self.assertEqual(calls, ["https://a/x"])
            self.assertTrue(dest.is_file())
            # 已就绪：第二次调用不再触发下载（断点跳过口径）
            result2 = download_weights.download_generic(
                "fake.pt", dest, ["https://a/x"], 1000, True, state,
                self._fail_fetcher)
            self.assertIn("已就绪", result2)
            self.assertEqual(calls, ["https://a/x"])

    def _fail_fetcher(self, url, dst, progress=None):
        raise AssertionError("已就绪文件不应触发下载")

    def test_download_generic_fails_and_cleans_parts(self):
        import download_weights

        with _tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "w" / "fake.pt"
            state = {}
            with self.assertRaises(RuntimeError) as ctx:
                download_weights.download_generic(
                    "fake.pt", dest, ["https://a/x", "https://b/x"], 1000,
                    True, state, self._write_junk)
            self.assertIn("2 个下载源", str(ctx.exception))
            self.assertFalse(dest.exists())
            self.assertEqual(list(dest.parent.glob("*.part*")), [])

    def test_family_yolo_uses_ensure_weight_path(self):
        """yolo 族与训练链路同一 ensure_pretrained_weight（落位/校验同口径）。

        全部权重在落位目录预置且校验通过时逐个跳过下载（fetcher 注入为
        立即失败——被调用即说明误触网络），返回就绪数等于清单数。
        """
        import download_weights
        from xccv.engine.weights import ensure_weight

        with _tempfile.TemporaryDirectory() as td:
            base = Path(td)
            dest_dir = base / "ultralytics"
            dest_dir.mkdir()
            for name in download_weights.YOLO_NAMES:
                (dest_dir / name).write_bytes(
                    b"PK\x03\x04" + b"\x00" * (200 * 1024))
            old_dir = ensure_weight.WEIGHTS_DIR
            old_sha = ensure_weight._KNOWN_SHA256
            ensure_weight.WEIGHTS_DIR = base
            ensure_weight._KNOWN_SHA256 = {}
            self.addCleanup(setattr, ensure_weight, "WEIGHTS_DIR", old_dir)
            self.addCleanup(setattr, ensure_weight, "_KNOWN_SHA256", old_sha)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                done = download_weights.family_yolo(
                    {}, fetcher=self._fail_fetcher)
            self.assertEqual(done, len(download_weights.YOLO_NAMES))


class EnsureFamilyWeightsTest(unittest.TestCase):
    """运行时接管：SAM/DINO/DINOv2/R-CNN 权重缺失时统一下载器补齐。

    双态契约：无网（fetcher 必败）不挂、回退原缺失指引路径；有网替身
    （注入 fetcher 写本地假权重）完成补齐且二次调用跳过下载。
    """

    def setUp(self):
        from xccv.engine.weights import ensure_family_weights as efw

        self._efw = efw
        self._tmp = _tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._base = Path(self._tmp.name)
        # 统一落位隔离：引擎权重目录整体指向临时根（WEIGHTS_DIR 单一来源）
        self.addCleanup(setattr, efw, "WEIGHTS_DIR", efw.WEIGHTS_DIR)
        efw.WEIGHTS_DIR = self._base
        # 统一注入：进度/落点路径校验的假 fetcher + 小尺寸门槛（免大文件）
        self.addCleanup(setattr, efw, "SAM_PT_MIN", efw.SAM_PT_MIN)
        self.addCleanup(setattr, efw, "SAM_ONNX_MIN", efw.SAM_ONNX_MIN)
        self.addCleanup(setattr, efw, "DINO_PTH_MIN", efw.DINO_PTH_MIN)
        self.addCleanup(setattr, efw, "TV_PTH_MIN", efw.TV_PTH_MIN)
        self.addCleanup(setattr, efw, "HF_WEIGHT_MIN", efw.HF_WEIGHT_MIN)
        efw.SAM_PT_MIN = efw.SAM_ONNX_MIN = 1000
        efw.DINO_PTH_MIN = efw.TV_PTH_MIN = efw.HF_WEIGHT_MIN = 1000
        self._calls: list = []

    def _ok_fetcher(self, url, dst, progress=None):
        self._calls.append(url)
        dst.write_bytes(b"PK\x03\x04" + b"\x00" * 2048)

    def _boom_fetcher(self, url, dst, progress=None):
        raise OSError("no network")

    def test_sam_weight_download_skip_and_failure(self):
        efw = self._efw

        dest = efw.ensure_sam_weight("sam2.1_t.pt", fetcher=self._ok_fetcher)
        self.assertEqual(dest, self._base / "sam" / "sam2.1_t.pt")
        self.assertTrue(dest.is_file())
        # 已就绪：二次调用不再触发下载
        dest2 = efw.ensure_sam_weight("sam2.1_t.pt", fetcher=self._ok_fetcher)
        self.assertEqual(dest2, dest)
        self.assertEqual(len(self._calls), 1)
        # 无网：中文 RuntimeError + .part 清理
        with self.assertRaises(RuntimeError) as ctx:
            efw.ensure_sam_weight("sam_b.pt", fetcher=self._boom_fetcher)
        self.assertIn("下载失败", str(ctx.exception))
        self.assertIn("手动下载", str(ctx.exception))
        self.assertEqual(list((self._base / "sam").glob("*.part*")), [])

    def test_sam_missing_files_survive_no_network(self):
        efw = self._efw

        # 双态：无网不挂，返回完成数 0（回退原缺失指引）
        self.assertEqual(efw.ensure_sam_missing_files(
            ["sam_b.pt", "mobile_sam.pt"], fetcher=self._boom_fetcher), 0)

    def test_all_entries_land_in_engine_weights_dir(self):
        """落位统一契约：全部下载条目的落点必须在引擎权重目录（WEIGHTS_DIR）
        之下，禁止仓库根 weights/ 等其他位置。"""
        efw = self._efw
        for fam in ("sam", "dino", "rcnn"):
            for e in efw.entries_for(fam):
                try:
                    dest = e.dest
                except RuntimeError:
                    continue  # torchvision 枚举不可用（文件名待解析）
                self.assertTrue(
                    str(dest).lower().startswith(str(efw.WEIGHTS_DIR).lower()),
                    f"{fam}/{e.key or e.label} 落点越界：{dest}")
        self.assertEqual(efw.hf_local_dir().parents[2], efw.WEIGHTS_DIR)
        self.assertEqual(efw.hub_checkpoints_dir().parents[2], efw.WEIGHTS_DIR)

    def test_registry_matches_ai_assist_models(self):
        efw = self._efw
        from xccv.annotation import ai_assist

        sam_keys = {e.key for e in efw.entries_for("sam")}
        for m in ai_assist._MODELS:
            self.assertIn(m["file"], sam_keys)
            if m.get("file2"):
                self.assertIn(m["file2"], sam_keys)
        self.assertIn("dino_vitbase8_pretrain.pth",
                      {e.key for e in efw.entries_for("dino")})

    def test_ai_assist_hook_downloads_missing_weight(self):
        efw = self._efw
        from xccv.annotation import ai_assist

        self.addCleanup(setattr, ai_assist, "_WEIGHTS_DIR",
                        ai_assist._WEIGHTS_DIR)
        ai_assist._WEIGHTS_DIR = self._base / "sam"
        real = efw.ensure_sam_missing_files
        efw.ensure_sam_missing_files = (
            lambda files, notify=None: real(files, notify,
                                            fetcher=self._ok_fetcher))
        self.addCleanup(setattr, efw, "ensure_sam_missing_files", real)
        eng = ai_assist._engine_by_name("Sam2 (balanced)")
        self.assertIsNotNone(eng)
        self.assertTrue((self._base / "sam" / "sam2.1_t.pt").is_file())

    def test_dino_vitb8_find_hook(self):
        efw = self._efw
        import xccv.engine.offline as _offline
        from xccv.engine.models import dino_vit

        # dino_vit 默认落点经 offline.WEIGHTS_DIR 解析——隔离到临时根
        self.addCleanup(setattr, _offline, "WEIGHTS_DIR", _offline.WEIGHTS_DIR)
        _offline.WEIGHTS_DIR = self._base
        old_env = os.environ.pop("XCCV_DINO_CKPT", None)
        self.addCleanup(lambda: os.environ.__setitem__(
            "XCCV_DINO_CKPT", old_env) if old_env else None)
        real = efw.ensure_dino_vitb8
        efw.ensure_dino_vitb8 = (
            lambda notify=None: real(notify, fetcher=self._ok_fetcher))
        self.addCleanup(setattr, efw, "ensure_dino_vitb8", real)
        got = dino_vit.find_dino_checkpoint()
        self.assertEqual(got,
                         self._base / "dino" / "dino_vitbase8_pretrain.pth")
        # 清掉补齐产物再测无网：接管失败回退 None（原缺失指引路径），不挂
        got.unlink()
        efw.ensure_dino_vitb8 = (
            lambda notify=None: real(notify, fetcher=self._boom_fetcher))
        self.assertIsNone(dino_vit.find_dino_checkpoint())

    def test_dinov2_ready_and_failure_paths(self):
        efw = self._efw
        import types as _types

        # 下载路径（离线桩）：快照下载以桩替代——落盘齐备文件后返回落点
        dest = self._base / "dinov2" / efw.DINOV2_REPO_ID

        def _fake_dl(**kw):
            p = Path(kw["local_dir"])
            p.mkdir(parents=True, exist_ok=True)
            (p / "config.json").write_text("{}", encoding="utf-8")
            (p / "preprocessor_config.json").write_text("{}", encoding="utf-8")
            (p / "model.safetensors").write_bytes(b"\x00" * (10 * 1024 * 1024 + 1))

        fake = _types.ModuleType("huggingface_hub")
        fake.snapshot_download = _fake_dl
        with mock.patch.dict(sys.modules, {"huggingface_hub": fake}):
            self.assertEqual(efw.ensure_dinov2(), dest)

        # 无网路径：权重低于门槛（未就绪）且 snapshot_download 必败 → None
        (dest / "model.safetensors").write_bytes(b"\x00" * 100)
        boom = _types.ModuleType("huggingface_hub")

        def _boom(**kw):
            raise OSError("no network")

        boom.snapshot_download = _boom
        with mock.patch.dict(sys.modules, {"huggingface_hub": boom}):
            self.assertIsNone(efw.ensure_dinov2())

        # 发现路径：一层子目录 config+权重齐备 → 直接返回，不联网
        ready = self._base / "dinov2" / "local-dinov2"
        ready.mkdir(parents=True)
        (ready / "config.json").write_text("{}", encoding="utf-8")
        (ready / "model.safetensors").write_bytes(b"\x00" * (10 * 1024 * 1024 + 1))
        self.assertEqual(efw.ensure_dinov2(), ready)

    def test_dinov2_find_backbone_discovery(self):
        efw = self._efw
        import xccv.engine.offline as _offline
        from xccv.engine.models import subspacead

        # subspacead 扫描 offline.WEIGHTS_DIR/dinov2；本机 HF 缓存
        # （Path.home）命中也会早退——一并隔离到临时根
        self.addCleanup(setattr, _offline, "WEIGHTS_DIR", _offline.WEIGHTS_DIR)
        _offline.WEIGHTS_DIR = self._base
        real_home = Path.home
        Path.home = classmethod(lambda cls: self._base)
        self.addCleanup(setattr, Path, "home", real_home)
        d = self._base / "dinov2" / "local-dinov2"
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}", encoding="utf-8")
        # 发现口径要求 config 与权重文件齐备（空壳缓存不视作就绪）
        (d / "model.safetensors").write_bytes(b"\x00" * (1024 * 1024 + 1))
        src, digest = subspacead._find_backbone()
        self.assertEqual(src, str(d))
        self.assertTrue(digest)

        # 接管下载的嵌套落点（org/<model>）同样可被发现
        nested = (self._base / "dinov2" / "facebook"
                  / "dinov2-with-registers-small")
        nested.mkdir(parents=True)
        (nested / "config.json").write_text("{}", encoding="utf-8")
        (nested / "model.safetensors").write_bytes(b"\x00" * (1024 * 1024 + 1))
        src2, digest2 = subspacead._find_backbone()
        self.assertEqual(src2, str(nested))
        self.assertTrue(digest2)

    def test_dinov2_find_backbone_empty_cache_skipped(self):
        """空壳 HF 缓存（快照仅有 config 无权重）不视作就绪：
        发现期不返回、不联网，抛 FileNotFoundError 由创建方延迟接管。"""
        efw = self._efw
        import xccv.engine.offline as _offline
        from xccv.engine.models import subspacead

        self.addCleanup(setattr, _offline, "WEIGHTS_DIR", _offline.WEIGHTS_DIR)
        _offline.WEIGHTS_DIR = self._base
        real_home = Path.home
        Path.home = classmethod(lambda cls: self._base)
        self.addCleanup(setattr, Path, "home", real_home)
        # 空壳：config 有、权重无
        d = self._base / "dinov2" / efw.DINOV2_REPO_ID
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(FileNotFoundError):
            subspacead._find_backbone()

    def test_dinov2_backbone_takeover_at_ensure(self):
        """创建期缺骨干 → 首用（_ensure_backbone）经统一下载器接管，
        notify 原样透传；接管后 transformers 加载失败给中文指引。"""
        efw = self._efw
        import xccv.engine.offline as _offline
        from xccv.engine.models import subspacead

        self.addCleanup(setattr, _offline, "WEIGHTS_DIR", _offline.WEIGHTS_DIR)
        _offline.WEIGHTS_DIR = self._base
        real_home = Path.home
        Path.home = classmethod(lambda cls: self._base)
        self.addCleanup(setattr, Path, "home", real_home)
        # 无任何本地骨干：发现期只抛 FileNotFoundError，不联网
        with self.assertRaises(FileNotFoundError):
            subspacead._find_backbone()
        # 惰性接管：_ensure_backbone 调统一下载器（notify 透传），
        # 拿到落点后再走 transformers 加载（离线夹具必然失败 → RuntimeError）
        inst = subspacead.SubspaceAdModel.__new__(subspacead.SubspaceAdModel)
        inst._bb_id = None
        inst._bb_sha = ""
        inst._bb_loaded = False
        inst._bb = None
        inst._proc = None
        inst.device = "cpu"
        d = self._base / "dinov2" / efw.DINOV2_REPO_ID
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}", encoding="utf-8")
        calls = []

        def _fake_ensure(notify=None):
            calls.append(notify)
            return d

        with mock.patch.object(efw, "ensure_dinov2", _fake_ensure):
            with self.assertRaises(RuntimeError):
                inst._ensure_backbone(notify="N")
        self.assertEqual(calls, ["N"])
        self.assertEqual(inst._bb_id, str(d))

    def test_rcnn_backbones_ensure(self):
        efw = self._efw
        hub = self._base / "torch" / "hub" / "checkpoints"
        # maskrcnn18 → resnet18 V1 预取到引擎权重目录的 torch.hub 缓存
        efw.ensure_rcnn_backbones("maskrcnn18", fetcher=self._ok_fetcher)
        self.assertTrue((hub / "resnet18-f37072fd.pth").is_file())
        # fasterrcnn → torchvision 默认 backbone 枚举（落点可解析）
        efw.ensure_rcnn_backbones("fasterrcnn", fetcher=self._ok_fetcher)
        self.assertTrue(any(hub.glob("resnet50-*.pth")))
        # use_pretrained=False：不触发任何下载
        before = len(self._calls)
        efw.ensure_rcnn_backbones("maskrcnn18", use_pretrained=False,
                                  fetcher=self._ok_fetcher)
        self.assertEqual(len(self._calls), before)

    def test_rcnn_backbones_survive_no_network(self):
        efw = self._efw
        # 双态：无网不挂（仅记日志，回退 torchvision 自带下载路径）
        efw.ensure_rcnn_backbones("maskrcnn18", fetcher=self._boom_fetcher)
        self.assertFalse((self._base / "torch" / "hub" / "checkpoints"
                          / "resnet18-f37072fd.pth").exists())


class DocReconcileDependencyTest(unittest.TestCase):
    """文档对账：torch/torchvision 版本矩阵三方一致（漂移即失败）。"""

    def test_real_repo_matrix_green(self):
        import doc_reconcile

        repo = Path(__file__).resolve().parents[1]
        problems = []
        doc_reconcile.check_dependency_matrix(repo, problems)
        self.assertEqual(problems, [])

    def test_drifted_matrix_reports_problems(self):
        import doc_reconcile

        with _tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wf = root / ".github" / "workflows"
            wf.mkdir(parents=True)
            (wf / "ci.yml").write_text(
                'pip install "torch==1.8.0" "torchvision==0.9.0" '
                '--index-url https://download.pytorch.org/whl/cpu\n',
                encoding="utf-8")
            (root / "README.md").write_text("readme", encoding="utf-8")
            (root / "README_EN.md").write_text("readme", encoding="utf-8")
            (root / "requirements.txt").write_text(
                "ultralytics==8.4.7\n", encoding="utf-8")
            problems = []
            doc_reconcile.check_dependency_matrix(root, problems)
            self.assertTrue(problems)
            joined = "；".join(problems)
            self.assertIn("torch==2.9.1", joined)
            self.assertIn("1.8.0", joined)

    def test_torch_line_in_requirements_rejected(self):
        """requirements.txt 不应包含 torch 安装行（torch 单独按源安装）。"""
        import doc_reconcile

        with _tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wf = root / ".github" / "workflows"
            wf.mkdir(parents=True)
            (wf / "ci.yml").write_text(
                'pip install "torch==2.9.1" "torchvision==0.24.1" '
                '--index-url https://download.pytorch.org/whl/cpu\n'
                "# ultralytics 8.4.7 constraint torch<2.10\n",
                encoding="utf-8")
            readme = (
                "pip install torch==2.9.1 torchvision==0.24.1 "
                "--index-url https://download.pytorch.org/whl/cu128\n"
                "CI uses --index-url https://download.pytorch.org/whl/cpu; "
                "torch<2.10 applies.\n")
            (root / "README.md").write_text(readme, encoding="utf-8")
            (root / "README_EN.md").write_text(readme, encoding="utf-8")
            (root / "requirements.txt").write_text(
                "ultralytics==8.4.7\ntorch==2.9.1\n", encoding="utf-8")
            problems = []
            doc_reconcile.check_dependency_matrix(root, problems)
            self.assertTrue(any("requirements.txt 不应包含 torch" in p
                                for p in problems))

if __name__ == '__main__':
    unittest.main()
