"""V0.6 T1-4 回归：主界面/标注/部署/训练/测试/模型工具窗口模块导入不拉 torch。

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


if __name__ == "__main__":
    unittest.main()
