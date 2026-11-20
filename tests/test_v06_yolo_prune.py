# -*- coding: utf-8 -*-
"""剪枝进程隔离与稳定性回归。

覆盖：
- 子进程协议契约（--selftest / post 模式回环 / error 事件与退出码）、
  紧急终止（手动 kill / 水位自动终止 / 硬超时）、终止后半截输出不落盘
  （原子落盘）、UI 终止按钮显隐与状态复位；
- yolo11n 剪 30% CPU 全流程成功 + 剪后模型可加载推理
  （内置权重在位才跑）。
"""
import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from tests.synthetic import make_classification_dataset  # noqa: E402
from xccv.engine.dataset.base import TaskType  # noqa: E402
from xccv.engine.models import TrainConfig, create_model, load_model  # noqa: E402
from xccv.engine.models.pruning import prune_checkpoint  # noqa: E402

_YOLO11N = ROOT / "xccv" / "engine" / "weights" / "ultralytics" / "yolo11n.pt"

# ---- 共享夹具：极小分类 ckpt（1 epoch/32px，进程内只构建一次）----
_FIXTURE_DIR: str | None = None


def _classifier_pt() -> Path:
    global _FIXTURE_DIR
    if _FIXTURE_DIR is None:
        root = Path(tempfile.mkdtemp(prefix="p7fix_"))
        atexit.register(shutil.rmtree, str(root), True)
        cls_dir, _ = make_classification_dataset(root / "cls", per_class=4)
        model = create_model("resnet18", TaskType.CLASSIFICATION, 2,
                             use_pretrained=False)
        model.train(TrainConfig(
            task=TaskType.CLASSIFICATION, dataset_dir=cls_dir,
            output_dir=root / "run", model_name="resnet18", epochs=1,
            batch_size=2, image_size=32, workers=0, use_pretrained=False,
            val_ratio=0.25))
        _FIXTURE_DIR = str(root)  # 成功后才落全局（训练失败不污染夹具）
    return Path(_FIXTURE_DIR) / "run" / "best.pt"


def _run_worker(args, timeout=240):
    return subprocess.run(
        [sys.executable, "-m", "xccv.engine.prune_worker", *args],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=timeout)


def _events(out: str) -> list:
    evs = []
    for line in (out or "").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                evs.append(json.loads(line))
            except ValueError:
                pass
    return evs


class WorkerProtocolTest(unittest.TestCase):
    """T7-1：JSON 行协议契约与退出码（真实子进程）。"""

    def test_selftest(self):
        r = _run_worker(["--selftest"], timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        evs = _events(r.stdout)
        self.assertEqual(evs[0]["event"], "phase")
        self.assertEqual(evs[0]["name"], "start")
        self.assertEqual(evs[-1]["event"], "done")
        self.assertEqual(evs[-1]["result"]["status"], "applied")

    def test_post_roundtrip(self):
        src = _classifier_pt()
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / "out.pt"
            r = _run_worker(["--mode", "post", "--src", str(src),
                             "--dst", str(dst), "--ratio", "0.2"])
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            dones = [e for e in _events(r.stdout) if e["event"] == "done"]
            self.assertEqual(len(dones), 1)
            self.assertEqual(dones[0]["result"]["status"], "applied")
            self.assertTrue(dst.is_file())
            self.assertEqual(list(Path(td).glob("*.part")), [])
            model = load_model(dst)
            pred = model.predict(np.zeros((32, 32, 3), np.uint8))
            self.assertEqual(len(pred.probs), 2)

    def test_missing_src_reports_error(self):
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / "out.pt"
            r = _run_worker(["--mode", "post", "--src", str(Path(td) / "no.pt"),
                             "--dst", str(dst), "--ratio", "0.2"])
            self.assertEqual(r.returncode, 1)
            evs = _events(r.stdout)
            self.assertEqual(evs[-1]["event"], "error")
            self.assertNotEqual(dst.exists(), True)


class _ManagerTestBase(unittest.TestCase):
    """_PruneProc 托管类基类（offscreen QApplication + 事件泵等待）。"""

    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])
        from xccv.modeltools import modeltools_window

        cls.mw = modeltools_window

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dst = Path(self._tmp.name) / "out.pt"

    def _mgr(self) -> "self.mw._PruneProc":
        mgr = self.mw._PruneProc()
        self.addCleanup(mgr.deleteLater)
        return mgr

    def _wait_until(self, cond, timeout=40.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.app.processEvents()
            if cond():
                return
            time.sleep(0.03)
        self.fail("等待超时")


class WorkerKillTest(_ManagerTestBase):
    """T7-1：手动终止 / 水位自动终止 / 硬超时 + 终止不留半截输出。"""

    def _start_sleeping(self, mgr, timeout_s=120.0):
        state = {"finished": False, "outcome": None, "phases": []}
        mgr.phase.connect(lambda n, d: state["phases"].append(n))
        mgr.finished.connect(lambda o: state.update(finished=True, outcome=o))
        mgr.start(["--mode", "post", "--src", str(_classifier_pt()),
                   "--dst", str(self.dst), "--ratio", "0.2",
                   "--sleep-after", "trace:60"], timeout_s=timeout_s)
        return state

    def test_manual_kill(self):
        mgr = self._mgr()
        state = self._start_sleeping(mgr)
        self._wait_until(lambda: "trace" in state["phases"])
        mgr.kill("单测手动终止")
        self._wait_until(lambda: state["finished"])
        self.assertEqual(state["outcome"]["status"], "killed")
        self.assertIn("手动", state["outcome"]["reason"])
        self.assertFalse(self.dst.exists(), "终止后不得留下半截输出")
        self.assertFalse(mgr.running())

    def test_watermark_auto_kill(self):
        mgr = self._mgr()
        state = self._start_sleeping(mgr)
        with patch.object(self.mw, "PRUNE_RAM_LIMIT_PCT", 0):
            self._wait_until(lambda: state["finished"], timeout=20)
        self.assertEqual(state["outcome"]["status"], "killed")
        self.assertIn("水位", state["outcome"]["reason"])

    def test_hard_timeout_kill(self):
        mgr = self._mgr()
        state = self._start_sleeping(mgr, timeout_s=1.0)
        self._wait_until(lambda: state["finished"], timeout=20)
        self.assertEqual(state["outcome"]["status"], "killed")
        self.assertIn("硬超时", state["outcome"]["reason"])


class PruneWindowUiSmokeTest(_ManagerTestBase):
    """T7-1：终止按钮随任务显隐；无任务时窗口可正常构建（offscreen）。"""

    def test_kill_button_visibility(self):
        from xccv.modeltools.modeltools_window import PruneToolWindow

        win = PruneToolWindow()
        self.addCleanup(win.close)
        self.assertFalse(win.btn_kill.isVisibleTo(win))
        win._set_running(True)
        self.assertTrue(win.btn_kill.isVisibleTo(win))
        win._set_running(False)
        self.assertFalse(win.btn_kill.isVisibleTo(win))
        self.assertIsNone(win._prune_proc)


@unittest.skipUnless(_YOLO11N.is_file(), "内置 yolo11n.pt 不在位")
class Yolo11nCpuPruneTest(unittest.TestCase):
    """YOLO 结构化剪枝**受控降级**（TP 上游对 ultralytics 图内存
    指数爆炸，1.4~1.6 全版本实测）——默认快速失败+可读原因，不留半截
    文件；XCCV_YOLO_PRUNE=1 专家模式才走真实 CPU 全流程。"""

    def test_controlled_degrade_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / "yolo11n_pruned30.pt"
            result = prune_checkpoint(_YOLO11N, dst, 0.3)
            self.assertIn("yolo-unsupported", result["status"], result)
            self.assertFalse(dst.exists(), "降级路径不得产出半截文件")

    @unittest.skipUnless(os.environ.get("XCCV_YOLO_PRUNE") == "1",
                         "专家模式默认不跑真实 YOLO 剪枝（上游 TP 内存爆炸）")
    def test_expert_mode_full_flow(self):
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / "yolo11n_pruned30.pt"
            result = prune_checkpoint(_YOLO11N, dst, 0.3)
            self.assertEqual(result["status"], "applied", result)
            self.assertLess(result["params_after"], result["params_before"])
            model = load_model(dst)
            pred = model.predict(np.zeros((160, 160, 3), np.uint8), conf=0.1)
            self.assertIsInstance(pred.boxes, list)


if __name__ == "__main__":
    unittest.main()
