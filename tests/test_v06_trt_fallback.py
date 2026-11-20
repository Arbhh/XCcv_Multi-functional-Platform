"""V0.6 T1-2 回归：.engine（TensorRT）在无 GPU/无 TensorRT 环境友好降级。

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
    TypeError 裸弹窗）。T1-2 只拦了 ModuleNotFoundError，漏掉这条路。"""

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


if __name__ == "__main__":
    unittest.main()
