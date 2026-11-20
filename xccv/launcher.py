"""平台独立进程启动器。

打包后每个平台是独立 exe（与主界面同在安装根），主界面点击卡片时优先
以子进程拉起同目录 exe；exe 缺失时回退解释器直接调用平台 main()（安装
态模块已编译为 .pyd，不能用 `python -m`，必须走 importlib）。开发态
（源码 .py）用 pythonw -m。任何平台出错都不会影响主界面与其他平台。
单实例由平台侧 single_instance 守卫负责。
"""

import subprocess
import sys
from pathlib import Path

from .app_paths import app_root, is_installed

# 平台 key -> (独立 exe 名, 模块名)
PLATFORMS = {
    "annotation": ("XCcv标注平台.exe", "xccv.annotation.annotator_window"),
    "training": ("XCcv训练平台.exe", "xccv.training.training_window"),
    "testing": ("XCcv测试平台.exe", "xccv.testing.testing_window"),
    "tools": ("XCcv模型工具.exe", "xccv.modeltools.modeltools_window"),
    # 部署族（v0.5.4：同一模块按 argv 分发三种独立进程窗口）
    "deploy": ("XCcv推理服务.exe", "xccv.deploy_platform"),
    "model_load": ("XCcv模型加载.exe", "xccv.deploy_platform"),
    "csharp_test": ("XCcvCSharp测试.exe", "xccv.deploy_platform"),
    # V0.6 P6：缺陷生成（DefectAugmentStudio，独立进程）
    "defectgen": ("XCcv缺陷生成.exe", "xccv.defectgen.defectgen_window"),
    # 流程图（新版=主界面流程图页；独立窗口同 FlowPage+HTML 画布）
    # 与主界面（复盘页「前往流程图界面」跳转目标）——修复缺键 KeyError
    "flowmain": ("XCcv流程图.exe", "xccv.flow_platform"),
    "main": ("XCcv.exe", "xccv.app"),
}


def _pythonw() -> str:
    python = Path(sys.executable)
    w = python.with_name("pythonw.exe")
    return str(w if w.is_file() else python)


def launch_platform(key: str, arg: str | None = None) -> subprocess.Popen:
    """以独立进程启动平台，返回进程句柄。

    arg 为可选的命令行参数（如数据体检的「去标注平台处理」把数据集目录
    传给标注平台；annotator_window.main 读取 sys.argv[1]）。
    """
    exe_name, module = PLATFORMS[key]
    if getattr(sys, "frozen", False) or is_installed():
        # 冻结/安装态：独立 exe 在安装根（python 环境 = <根>\app，见
        # app_paths 安装态布局），缺失回退 importlib 调 main()
        exe = Path(sys.prefix).resolve().parent / exe_name
        if not exe.is_file():  # 旧布局兜底：exe 与解释器同目录
            exe = Path(sys.executable).resolve().parent / exe_name
        if exe.is_file():
            cmd = [str(exe)]
            if arg:
                cmd.append(arg)
            return subprocess.Popen(cmd, cwd=str(exe.parent), close_fds=True)
        code = (f"import importlib; importlib.import_module('{module}').main()")
        cmd = [_pythonw(), "-c", code]
    else:  # 开发态：源码模块可直接 -m
        cmd = [_pythonw(), "-m", module]
    if arg:
        cmd.append(arg)
    return subprocess.Popen(cmd, cwd=str(app_root()), close_fds=True)
