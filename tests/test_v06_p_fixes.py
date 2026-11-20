# -*- coding: utf-8 -*-
"""V0.6.0 安装实测缺陷修复回归。

- P-01 剪枝 worker done 行崩溃：pretrain_prune 返回的 result 含 "ckpt"
  （整个网络对象），worker `_emit(event="done")` 的 json.dumps 崩溃 →
  子进程 exit 1、UI 报失败（而产物其实已落盘）。修复=emit 前剥离
  ckpt + `_emit` 对不可序列化对象降级兜底。
  回归：pre 模式端到端（真实子进程，分类小数据集）必须 returncode=0、
  done.result 不含 ckpt、dst 落盘——修复前该用例必现复现缺陷。
- P-02 加速导出签名：modeltools `_export_engine` wrapper 缺 progress
  形参而调用处已传 → 必现 TypeError。回归=签名含 progress 且透传
  到引擎层 export_engine。
"""
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

os_guard = __import__("os")
os_guard.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]

from tests.synthetic import make_classification_dataset  # noqa: E402


class EmitFallbackTest(unittest.TestCase):
    """P-01a：_emit 遇不可序列化对象降级而不是崩溃。"""

    def test_emit_tensor_result_degrades(self):
        import torch

        from xccv.engine.prune_worker import _emit

        buf = io.StringIO()
        with redirect_stdout(buf):
            _emit(event="done", result={"status": "applied",
                                        "ckpt": torch.zeros(2)})
        line = buf.getvalue().strip()
        ev = json.loads(line)  # 不抛 = 协议行可用
        self.assertEqual(ev["event"], "done")
        self.assertTrue(ev.get("stripped"))  # 降级标记

    def test_emit_plain_result_untouched(self):
        from xccv.engine.prune_worker import _emit

        buf = io.StringIO()
        with redirect_stdout(buf):
            _emit(event="done", result={"status": "applied", "params": 123})
        ev = json.loads(buf.getvalue().strip())
        self.assertNotIn("stripped", ev)
        self.assertEqual(ev["result"]["params"], 123)


class PreModeEndToEndTest(unittest.TestCase):
    """P-01b：pre 模式端到端（真实子进程）——done 行与产物落盘。"""

    @classmethod
    def setUpClass(cls):
        root = Path(tempfile.mkdtemp(prefix="pfix_pre_"))
        cls._root = root
        cls_dir, _ = make_classification_dataset(root / "cls", per_class=4)
        cls._dataset = cls_dir

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(str(cls._root), True)

    def test_pre_prune_done_line_and_artifact(self):
        dst = self._root / "out" / "resnet18_pruned.pt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            [sys.executable, "-m", "xccv.engine.prune_worker",
             "--mode", "pre", "--dataset-dir", str(self._dataset),
             "--task", "classification", "--model-name", "resnet18",
             "--width", "32", "--height", "32", "--ratio", "0.3",
             "--dst", str(dst)],
            cwd=str(ROOT), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=300)
        self.assertEqual(r.returncode, 0,
                         "stdout=%s stderr=%s" % (r.stdout[-1500:], r.stderr[-1500:]))
        evs = [json.loads(l) for l in r.stdout.splitlines()
               if l.strip().startswith("{")]
        dones = [e for e in evs if e["event"] == "done"]
        self.assertEqual(len(dones), 1)
        res = dones[0]["result"]
        self.assertEqual(res["status"], "applied")
        self.assertNotIn("ckpt", res)  # P-01：网络对象不得走 stdout
        self.assertTrue(dst.is_file())  # 产物落盘


class ExportEngineWrapperTest(unittest.TestCase):
    """P-02：加速导出 wrapper 接受并透传 progress。"""

    def test_signature_accepts_progress(self):
        import inspect

        from xccv.modeltools.modeltools_window import _export_engine

        self.assertIn("progress", inspect.signature(_export_engine).parameters)

    def test_progress_forwarded_to_engine(self):
        import xccv.modeltools.modeltools_window as mw

        calls = {}
        with patch("xccv.engine.model_export.export_engine",
                   side_effect=lambda *a, **k: calls.update(k) or {}):
            cb = lambda f, t: None  # noqa: E731
            mw._export_engine("s.pt", "d.engine", imgsz=640, half=True,
                              progress=cb)
        self.assertIs(calls.get("progress"), cb)
        self.assertEqual(calls.get("imgsz"), 640)
        self.assertEqual(calls.get("half"), True)


if __name__ == "__main__":
    unittest.main()
