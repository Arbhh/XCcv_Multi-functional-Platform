"""训练平台测试：4 个历史测试模块的合并模块。

由以下模块按域合并而成，用例逐段原样迁移、断言未改：
- test_v06_yolo_prune
- test_weight_slim
- test_v06_lazy_torch
- test_v06_slim_init
"""


# == 来源模块：test_v06_yolo_prune ==
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
    """JSON 行协议契约与退出码（真实子进程）。"""

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
    """手动终止 / 水位自动终止 / 硬超时 + 终止不留半截输出。"""

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
    """终止按钮随任务显隐；无任务时窗口可正常构建（offscreen）。"""

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

# == 来源模块：test_weight_slim ==
# -*- coding: utf-8 -*-
"""训练产物瘦身测试：YOLO ckpt 剥离 optimizer 等推理无关数据。"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import torch
from torch import nn

from xccv.engine.models.yolo import _lean_yolo_ckpt


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)


class LeanYoloCkptTest(unittest.TestCase):
    def test_weights_in_ema_key_moved_to_model(self):
        """训练期 best.pt：权重挂在 ema、model=None → 归一到 model。"""
        net = _TinyNet()
        ckpt = {"epoch": 3, "best_fitness": 0.5, "model": None, "ema": net,
                "optimizer": {"state": {"x": torch.zeros(1000)}},
                "scaler": {"scale": 1.0}, "updates": 5,
                "train_args": {"task": "detect"}, "train_metrics": {},
                "train_results": {}, "date": "d", "version": "8.4.7"}
        lean = _lean_yolo_ckpt(ckpt)
        self.assertIs(lean["model"], net)
        for absent in ("optimizer", "scaler", "ema", "updates",
                       "train_metrics", "train_results"):
            self.assertNotIn(absent, lean)
        self.assertEqual(lean["epoch"], 3)
        self.assertEqual(lean["train_args"], {"task": "detect"})

    def test_weights_in_model_key_kept(self):
        """strip 后的 ckpt：权重在 model → 原样保留。"""
        net = _TinyNet()
        lean = _lean_yolo_ckpt({"model": net, "ema": None, "optimizer": None,
                                "epoch": 9})
        self.assertIs(lean["model"], net)

    def test_no_weights_raises(self):
        with self.assertRaises(RuntimeError):
            _lean_yolo_ckpt({"model": None, "ema": None})

# == 来源模块：test_v06_lazy_torch ==
"""主界面/标注/部署/训练/测试/模型工具窗口模块导入不拉 torch。

平台窗口模块在启动闪屏显示之前导入——模块级 torch/引擎链会把加载中界面
推迟数秒。修复后窗口模块必须 torch-free，引擎 API 经
xccv.engine.__getattr__（PEP 562）与函数内局部导入按需引入。

子进程探针：与门禁进程隔离，保证 sys.modules 干净起点。
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PY = sys.executable

_MODULES = [
    "xccv.app",
    "xccv.annotation.annotator_window",
    "xccv.deploy_platform",
    "xccv.testing.testing_window",
    "xccv.training.training_window",
    "xccv.modeltools.modeltools_window",
    # 流程图入口链（V0.6 收尾核验补：首版漏测——detection_flow 模块级
    # `from .models.base import Prediction` 经 models 包拉起 torch，流程图
    # 平台首窗 2.6s、主界面首次进入流程页卡 2.6s；）
    "xccv.flow_platform",
    "xccv.modeltools.flow_shared",
    "xccv.modeltools.flowgraph_web_window",
    "xccv.engine.detection_flow",
]

_PROBE = (
    "import sys; sys.path.insert(0, r'{root}');\n"
    "mods = sys.argv[1:]\n"
    "import importlib\n"
    "for m in mods:\n"
    "    importlib.import_module(m)\n"
    "print('TORCH_LOADED' if 'torch' in sys.modules else 'CLEAN')\n"
).format(root=str(_ROOT))


class LazyTorchImportTest(unittest.TestCase):
    def test_platform_windows_import_without_torch(self):
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        proc = subprocess.run([_PY, "-c", _PROBE, *_MODULES],
                              capture_output=True, text=True, env=env,
                              cwd=str(_ROOT), timeout=300)
        out = proc.stdout.strip().splitlines()
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertEqual(out[-1] if out else "", "CLEAN",
                         f"平台窗口模块导入了 torch：\n{proc.stdout[-500:]}")

    def test_engine_lazy_api_forwarding(self):
        """PEP 562：from xccv.engine import load_model / getattr 两种写法均可用。"""
        code = (
            "import sys; sys.path.insert(0, r'%s')\n"
            "import xccv.engine as e\n"
            "from xccv.engine import Prediction\n"
            "assert callable(e.load_model) and Prediction is not None\n"
            "try:\n"
            "    e.no_such_attr\n"
            "    raise SystemExit('AttributeError expected')\n"
            "except AttributeError:\n"
            "    pass\n"
            "print('TORCH' if 'torch' in sys.modules else 'CLEAN')\n"
        ) % str(_ROOT)
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        proc = subprocess.run([_PY, "-c", code], capture_output=True,
                              text=True, env=env, cwd=str(_ROOT), timeout=120)
        out = proc.stdout.strip().splitlines()
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        # 惰性访问发生在最后一步之前已把 models 引入——只断言 API 转发正确
        self.assertIn(out[-1] if out else "", ("CLEAN", "TORCH"))

# == 来源模块：test_v06_slim_init ==
"""自建初始权重：构建器/注入优先级测试。

- tools/build_maskrcnn18_init.py：初始化配方落地 + 确定性 + 产物落盘；
- rcnn._load_slim_pretrained：**自建 init 优先**，转换产物 slim_coco
  仅最后兜底；类别头按形状
  匹配跳过（COCO 类数与项目类数不一致时新头随机初始化）。
全部 imagenet=False（离线、免下载），骨干随机但配方域可断言。
"""

import tempfile
import unittest
from pathlib import Path

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import torch

from xccv.engine.dataset.base import TaskType
from xccv.engine.models.rcnn import RCNNModel, _load_slim_pretrained
from build_maskrcnn18_init import _apply_recipe, build_init_net, save_artifact


class RecipeTest(unittest.TestCase):
    """配方落地与确定性（离线路径，骨干随机）。"""

    def test_recipe_lands_and_is_deterministic(self):
        import hashlib

        net, notes = build_init_net(num_classes=2, imagenet=False)
        self.assertEqual(notes["backbone"], "random")
        mp = net.roi_heads.mask_predictor
        self.assertLess(float(mp.mask_fcn_logits.weight.std()), 0.002)
        self.assertLess(
            float(net.roi_heads.box_predictor.cls_score.weight.std()), 0.012)
        self.assertEqual(
            float(net.roi_heads.box_predictor.cls_score.bias.abs().max()), 0.0)
        self.assertEqual(net.roi_heads.box_head.fc6.in_features, 64 * 7 * 7)
        n = sum(p.numel() for p in net.parameters())
        self.assertTrue(11.2e6 < n < 13.8e6, f"{n/1e6:.2f}M")
        net2, _ = build_init_net(num_classes=2, imagenet=False)
        keys = sorted(k for k in net.state_dict()
                      if not k.startswith("backbone.body."))

        def _h(sd):
            hh = hashlib.sha256()
            for k in keys:
                hh.update(k.encode())
                hh.update(sd[k].detach().cpu().contiguous()
                          .numpy().tobytes())
            return hh.hexdigest()

        self.assertEqual(_h(net.state_dict()), _h(net2.state_dict()))

    def test_build_uses_platform_class_semantics(self):
        """类别数口径：_build 内部 +1 背景；cls out=80+1（COCO 口径产物）。"""
        net, _ = build_init_net(num_classes=2, imagenet=False)
        self.assertEqual(
            net.roi_heads.box_predictor.cls_score.out_features, 3)


class ArtifactRoundtripTest(unittest.TestCase):
    def test_save_then_load_rebuilds_slim(self):
        net, _ = build_init_net(num_classes=2, imagenet=False)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "maskrcnn18_slim_init.pt"
            save_artifact(net, out)
            self.assertTrue(out.is_file())
            ckpt = torch.load(out, map_location="cpu", weights_only=False)
            m = RCNNModel("fasterrcnn", TaskType.DETECTION, 1, device="cpu",
                          use_pretrained=False)
            m.load_weights(ckpt)
            self.assertEqual(m.net.backbone.out_channels, 64)
            self.assertEqual(
                m.net.roi_heads.box_predictor.cls_score.out_features, 3)


class InjectionPriorityTest(unittest.TestCase):
    """_load_slim_pretrained：自建 init 优先，coco 转换产物仅最后兜底。"""

    def _make_net(self, nc):
        return RCNNModel("maskrcnn18", TaskType.SEGMENTATION, nc,
                         device="cpu", use_pretrained=False)._build(
            "maskrcnn18", nc, 320, use_pretrained=False, slim_head=True)

    def test_init_priority_then_coco_fallback(self):
        """源权重=自建 init 优先；coco 转换产物仅最后兜底。"""
        from unittest import mock

        from build_maskrcnn18_init import build_init_net as _b

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "weights").mkdir()
            net_init, _ = _b(num_classes=3, imagenet=False)
            save_artifact(net_init,
                          root / "weights" / "maskrcnn18_slim_init.pt")
            net_coco, _ = _b(num_classes=3, imagenet=False)
            sd = net_coco.state_dict()
            sd["backbone.body.conv1.weight"] = torch.full_like(
                sd["backbone.body.conv1.weight"], 1.5)
            tmp_pt = root / "coco.pt"
            torch.save({"format": "xccv-rcnn", "arch": "maskrcnn18",
                        "state_dict": sd}, tmp_pt)
            import shutil

            shutil.copy(tmp_pt, root / "weights" / "maskrcnn18_slim_coco.pt")
            tmp_pt.unlink()

            net = self._make_net(3)
            with mock.patch("xccv.engine.offline.WEIGHTS_DIR",
                            root / "weights"), \
                    mock.patch("xccv.app_paths.app_root", return_value=root):
                self.assertTrue(_load_slim_pretrained(net, 3))
            # 优先级：吃 init（非 coco 标记值）
            self.assertNotEqual(
                float(net.backbone.body.conv1.weight.flatten()[0]), 1.5)
            self.assertTrue(torch.equal(
                net.backbone.body.conv1.weight,
                net_init.backbone.body.conv1.weight))
            # 移除 init → 兜底 coco（标记值 1.5）
            (root / "weights" / "maskrcnn18_slim_init.pt").unlink()
            net2 = self._make_net(3)
            with mock.patch("xccv.engine.offline.WEIGHTS_DIR",
                            root / "weights"), \
                    mock.patch("xccv.app_paths.app_root", return_value=root):
                self.assertTrue(_load_slim_pretrained(net2, 3))
            self.assertEqual(
                float(net2.backbone.body.conv1.weight.flatten()[0]), 1.5)

    def test_class_heads_skipped_on_mismatch(self):
        """COCO 80 类产物注入 3 类项目：骨干注入、类别头保持随机（形状跳过）。"""
        from unittest import mock

        from build_maskrcnn18_init import build_init_net as _b

        net_init, _ = _b(num_classes=80, imagenet=False)
        net = self._make_net(3)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "weights").mkdir()
            save_artifact(net_init,
                          root / "weights" / "maskrcnn18_slim_init.pt")
            before = net.roi_heads.box_predictor.cls_score.weight.clone()
            with mock.patch("xccv.engine.offline.WEIGHTS_DIR",
                            root / "weights"), \
                    mock.patch("xccv.app_paths.app_root", return_value=root):
                self.assertTrue(_load_slim_pretrained(net, 3))
            self.assertTrue(torch.equal(
                net.backbone.body.conv1.weight,
                net_init.backbone.body.conv1.weight), "骨干应被注入")
            self.assertTrue(torch.equal(
                net.roi_heads.box_predictor.cls_score.weight, before),
                "类别数不匹配的头必须保持原值")


# == 预训练权重下载进度可见化 ==

class EnsureWeightTest(unittest.TestCase):
    """ensure_pretrained_weight 契约：命中即返 / 多源回退 / 失败抛出 / 直通。"""

    def setUp(self):
        # 合成测试文件不是官方权重：关闭已知 SHA-256 表，单测校验逻辑另测
        from xccv.engine.weights import ensure_weight
        self._old_sha = ensure_weight._KNOWN_SHA256
        ensure_weight._KNOWN_SHA256 = {}
        self.addCleanup(setattr, ensure_weight, "_KNOWN_SHA256", self._old_sha)

    @staticmethod
    def _fake_weight(size=200_000):
        return b"PK\x03\x04" + b"\0" * size

    def test_bundled_hit_returns_path(self):
        from xccv.engine.weights.ensure_weight import ensure_pretrained_weight
        from xccv.engine.offline import WEIGHTS_DIR
        dest = WEIGHTS_DIR / "ultralytics" / "yolo11n.pt"
        if not dest.is_file():
            self.skipTest("随包权重不在位（CI 口径）")

        def _boom(url, dst, progress):
            raise AssertionError("命中时不得发起下载")
        out = ensure_pretrained_weight("yolo11n.pt", fetcher=_boom)
        self.assertEqual(Path(out), dest)

    def test_download_reports_and_lands_bundled(self):
        import tempfile
        from xccv.engine.weights import ensure_weight
        dest_dir = Path(tempfile.mkdtemp())
        old = ensure_weight.WEIGHTS_DIR
        ensure_weight.WEIGHTS_DIR = dest_dir
        try:
            msgs = []

            def fetcher(url, dst, progress):
                self.assertIn("yolo11n.pt", url)
                dst.write_bytes(self._fake_weight())
                progress(1.0)
            out = ensure_weight.ensure_pretrained_weight(
                "yolo11n.pt", notify=lambda text, frac: msgs.append((text, frac)),
                fetcher=fetcher)
            self.assertEqual(Path(out), dest_dir / "ultralytics" / "yolo11n.pt")
            self.assertTrue((dest_dir / "ultralytics" / "yolo11n.pt").is_file())
            self.assertIsNone(msgs[0][1])
            self.assertEqual(msgs[-1][1], 1.0)
            self.assertFalse((dest_dir / "ultralytics" / "yolo11n.pt.part1").exists())
        finally:
            ensure_weight.WEIGHTS_DIR = old

    def test_failure_retries_next_mirror(self):
        import tempfile
        from xccv.engine.weights import ensure_weight
        dest_dir = Path(tempfile.mkdtemp())
        old = ensure_weight.WEIGHTS_DIR
        ensure_weight.WEIGHTS_DIR = dest_dir
        try:
            calls = []

            def fetcher(url, dst, progress):
                calls.append(url)
                if len(calls) == 1:
                    dst.write_bytes(b"PK\x03\x04partial")
                    raise ConnectionError("网络不可用")
                dst.write_bytes(self._fake_weight())
                progress(1.0)
            out = ensure_weight.ensure_pretrained_weight(
                "yolo11n.pt", fetcher=fetcher)
            self.assertEqual(Path(out), dest_dir / "ultralytics" / "yolo11n.pt")
            self.assertEqual(len(calls), 2)
            self.assertNotEqual(calls[0], calls[1])
            self.assertFalse((dest_dir / "ultralytics" / "yolo11n.pt.part1").exists())
        finally:
            ensure_weight.WEIGHTS_DIR = old

    def test_all_sources_fail_raises_without_fallback(self):
        import tempfile
        from xccv.engine.weights import ensure_weight
        dest_dir = Path(tempfile.mkdtemp())
        old = ensure_weight.WEIGHTS_DIR
        ensure_weight.WEIGHTS_DIR = dest_dir
        try:
            def fetcher(url, dst, progress):
                raise ConnectionError("网络不可用")
            msgs = []
            with self.assertRaises(RuntimeError) as ctx:
                ensure_weight.ensure_pretrained_weight(
                    "yolo11n.pt",
                    notify=lambda text, frac: msgs.append((text, frac)),
                    fetcher=fetcher)
            self.assertIn("下载失败", str(ctx.exception))
            self.assertIn("已尝试", str(ctx.exception))
            self.assertTrue(any("失败" in text for text, _ in msgs))
            self.assertFalse(list((dest_dir / "ultralytics").glob("*.part*")))
        finally:
            ensure_weight.WEIGHTS_DIR = old

    def test_incomplete_existing_file_is_redownloaded(self):
        import tempfile
        from xccv.engine.weights import ensure_weight
        dest_dir = Path(tempfile.mkdtemp())
        old = ensure_weight.WEIGHTS_DIR
        ensure_weight.WEIGHTS_DIR = dest_dir
        try:
            dest = dest_dir / "ultralytics" / "yolo11n.pt"
            dest.parent.mkdir(parents=True)
            dest.write_bytes(b"PK\x03\x04tiny")     # 截断文件不应命中
            called = []

            def fetcher(url, dst, progress):
                called.append(1)
                dst.write_bytes(self._fake_weight())
                progress(1.0)
            out = ensure_weight.ensure_pretrained_weight(
                "yolo11n.pt", fetcher=fetcher)
            self.assertTrue(called)
            self.assertEqual(Path(out), dest)
        finally:
            ensure_weight.WEIGHTS_DIR = old

    def test_known_weight_hash_mismatch_rejected(self):
        import tempfile
        from xccv.engine.weights import ensure_weight
        dest_dir = Path(tempfile.mkdtemp())
        old_dir = ensure_weight.WEIGHTS_DIR
        ensure_weight.WEIGHTS_DIR = dest_dir
        ensure_weight._KNOWN_SHA256 = {"yolo11n.pt": "0" * 64}
        try:
            def fetcher(url, dst, progress):
                dst.write_bytes(self._fake_weight())
                progress(1.0)
            with self.assertRaises(RuntimeError):
                ensure_weight.ensure_pretrained_weight(
                    "yolo11n.pt", fetcher=fetcher)
            self.assertFalse((dest_dir / "ultralytics" / "yolo11n.pt").exists())
        finally:
            ensure_weight.WEIGHTS_DIR = old_dir

    def test_explicit_path_and_yaml_pass_through(self):
        from xccv.engine.weights.ensure_weight import ensure_pretrained_weight
        marker = str(Path("D:/weights/my.pt"))
        self.assertEqual(ensure_pretrained_weight(marker), marker)
        self.assertEqual(ensure_pretrained_weight("yolo11n.yaml"), "yolo11n.yaml")


class P2TransferWeightTest(unittest.TestCase):
    """P2 迁移的基底权重也必须走平台下载器（不得回退裸名触发 ultralytics）。"""

    @staticmethod
    def _model():
        from xccv.engine.models.yolo import YoloModel
        m = YoloModel("yolo11n", TaskType.DETECTION, 2, use_pretrained=True)
        m.enable_p2()      # init_weight 变为 p2 yaml，基底权重由 _p2_transfer 取
        return m

    class _DummyModel:
        def __init__(self):
            self.ckpt = {}
            self.model = object()
            self.loaded = None

        def load(self, path):
            self.loaded = path

    def test_p2_transfer_uses_platform_downloader(self):
        from xccv.engine.models import yolo as yolo_mod
        m = self._model()
        dummy = self._DummyModel()
        calls = []
        old = yolo_mod.ensure_pretrained_weight
        yolo_mod.ensure_pretrained_weight = (
            lambda name, notify=None: (calls.append(name), "/tmp/base.pt")[1])
        self.addCleanup(setattr, yolo_mod, "ensure_pretrained_weight", old)
        m._p2_transfer(dummy, reporter=None)
        self.assertEqual(calls, ["yolo11n.pt"])
        self.assertEqual(dummy.loaded, "/tmp/base.pt")

    def test_p2_transfer_failure_keeps_random_init(self):
        from xccv.engine.models import yolo as yolo_mod
        m = self._model()
        dummy = self._DummyModel()
        old = yolo_mod.ensure_pretrained_weight

        def _boom(name, notify=None):
            raise RuntimeError("预训练权重下载失败")
        yolo_mod.ensure_pretrained_weight = _boom
        self.addCleanup(setattr, yolo_mod, "ensure_pretrained_weight", old)
        msgs = []

        class _R:
            @staticmethod
            def on_message(text, frac=None):
                msgs.append(text)

        m._p2_transfer(dummy, reporter=_R())   # 不阻塞：按随机初始化继续
        self.assertIsNone(dummy.loaded)
        self.assertTrue(any("随机初始化" in t for t in msgs))


class YoloNoNetworkDownloadTest(unittest.TestCase):
    """ultralytics 内部下载入口：AMP 自检不得再拉 yolo26n.pt。"""

    def test_amp_probe_is_disabled(self):
        from xccv.engine.models.yolo import _disable_amp_probe
        _disable_amp_probe()
        from ultralytics.engine import trainer as ult_trainer
        from ultralytics.utils import checks as ult_checks
        self.assertTrue(ult_checks.check_amp(None))
        self.assertTrue(ult_trainer.check_amp(None))


class ReporterMessageHookTest(unittest.TestCase):
    """TrainReporter.on_message 基类可实例化；_QtReporter 转发 status_message。"""

    def test_base_reporter_noop(self):
        from xccv.engine.models.base import TrainReporter

        class _R(TrainReporter):
            pass
        _R().on_message("任意消息", None)

    def test_qt_reporter_forwards_status_message(self):
        from PyQt5.QtCore import QObject, pyqtSignal
        from xccv.training.trainer_thread import _make_reporter

        class _FakeThread(QObject):
            status_message = pyqtSignal(str, float)
        th = _FakeThread()
        rep = _make_reporter(th)
        seen = []
        th.status_message.connect(lambda text, frac: seen.append((text, frac)))
        rep.on_message("正在下载预训练权重 yolo11n.pt …", None)
        rep.on_message("正在下载预训练权重 yolo11n.pt …", 0.5)
        self.assertEqual(seen, [("正在下载预训练权重 yolo11n.pt …", -1.0),
                                ("正在下载预训练权重 yolo11n.pt …", 0.5)])

    def test_train_thread_has_status_message_signal(self):
        from xccv.training.trainer_thread import TrainThread

        self.assertTrue(hasattr(TrainThread, "status_message"))

if __name__ == '__main__':
    unittest.main()
