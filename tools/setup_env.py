# -*- coding: utf-8 -*-
"""XCcv 一键环境安装器（根目录 install.bat 的逻辑承载，纯 CLI、不依赖 Qt）。

需求：零基础用户双击 bat 即可完成"探测 GPU/驱动 → 选 torch 安装源 → 创建 venv →
装依赖 → 写 python_path.txt → 环境体检"全流程；失败分支（无 GPU / 无网 / 代理 /
权限 / 版本不符）每条都有明确的中文可读指引，离线机给出离线安装指引路径。

实现：bat 只负责"找到系统 python → 启动本脚本"（薄入口）；本脚本承载全部逻辑——
- GPU 探测：nvidia-smi 查询 显卡名/驱动版本/算力 → CUDA 家族；
- torch 源选择：cu128（Blackwell/新驱动）/ cu126（其余 N 卡）/ cpu（无 N 卡或
  驱动过旧）；ultralytics 8.4.7 上界约束 torch<2.10，默认固定 torch==2.9.1
  + torchvision==0.24.1（与 CI / README / requirements.txt 版本矩阵一致）；
- venv 创建于仓库根 venv/（env_check.py 已把它列为兜底候选）；
- 失败即停（exit 1），已完成的步骤不回滚，重跑可续。

使用：python tools/setup_env.py [--root DIR] [--index NAME|URL] [--offline]
      [--torch VER] [--torchvision VER] [--venv-name NAME] [--skip-check]
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

ROOT_DEFAULT = Path(__file__).resolve().parent.parent

# 版本矩阵基准（与 .github/workflows/ci.yml CPU 源口径一致；
# tools/doc_reconcile.py 对 README / requirements.txt / ci.yml 三方对账）
TORCH_VERSION = "2.9.1"
TORCHVISION_VERSION = "0.24.1"
ULTRALYTICS_TORCH_UPPER = (2, 10)      # ultralytics 8.4.7 约束 torch<2.10
# 依赖矩阵锚定的宿主 Python（requirements 固定版本按此打包验证；
# 更新的 Python 缺少对应安装包，如 onnxruntime==1.19.2 无 3.13/3.14 wheel）
HOST_PYTHON = (3, 11)

# torch 安装源：名字 → pytorch 官方 index（本地按显卡选源；CI 用 cpu 源）
TORCH_INDEXES = {
    "cu128": "https://download.pytorch.org/whl/cu128",
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cpu": "https://download.pytorch.org/whl/cpu",
}
# cu12x 轮子经 CUDA 次版本兼容机制要求的最低 NVIDIA 驱动（12.0 家族基线）
MIN_DRIVER_CU = 525
# Blackwell（RTX 50 系，算力 sm_100/sm_120）必须 cu128 源
SM_REQUIRED_CU128 = 100

OFFLINE_GUIDE = """\
离线安装指引（内网/无网机器）：
  1. 在任一可联网的 Windows x64 机器（Python 3.11）执行：
       pip download -d wheels/ -r requirements.txt
       pip download -d wheels/ torch=={torch} torchvision=={tv} \
--index-url {index}
  2. 把整个仓库目录与 wheels/ 目录一并拷贝到离线机；
  3. 在离线机仓库根执行：
       python -m venv venv
       venv\\Scripts\\python.exe -m pip install --no-index --find-links wheels/ \
torch=={torch} torchvision=={tv}
       venv\\Scripts\\python.exe -m pip install --no-index --find-links wheels/ \
-r requirements.txt
  4. 写一行 venv 解释器路径到本目录 python_path.txt：
       <仓库根>\\venv\\Scripts\\python.exe
  5. 双击 run_main.bat 启动主界面（或运行 tools\\env_check.py 体检）。
  详细说明见 docs/FAQ.md 与 README「安装」一节。
"""


def _safe_console() -> None:
    """Windows 控制台编码兜底：无法编码的字符替换输出，不因中文而崩溃。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def _step(title: str) -> None:
    print("")
    print(f"===== {title} =====")


def _fail(title: str, lines: list) -> None:
    """失败分支统一出口：标题 + 逐条中文指引，退出码 1。"""
    print("")
    print(f"[失败] {title}")
    for line in lines:
        print(f"  - {line}")
    print("（已完成的步骤不回滚，处理后可直接重跑本脚本续装。）")
    raise SystemExit(1)


def check_host_python() -> None:
    """宿主解释器体检：必须是 64 位 Python 3（推荐 3.11）。

    解释器路径随体检打印——安装入口按"探测首个可用者"选定宿主，PATH 中
    的 python 可能被其他项目的环境占用（读错环境），路径可见即可即时发现。
    """
    print(f"[宿主解释器] {sys.executable}")
    if sys.version_info < (3,):
        _fail("Python 版本过旧", [
            f"当前 {sys.version.split()[0]}，XCcv 需要 Python 3.11 x64",
            "请到 https://www.python.org/downloads/ 安装 Python 3.11，"
            "安装时勾选 Add python.exe to PATH，然后重跑安装器",
        ])
    if sys.maxsize <= 2 ** 32:
        _fail("Python 是 32 位", [
            "32 位 Python 无法安装 PyTorch；请安装 64 位（x64）Python 3.11",
        ])
    if sys.version_info[:2] != (3, 11):
        # 依赖矩阵按 Python 3.11 固定：onnxruntime 等包在更新的 Python 上
        # 没有安装包（3.13/3.14 装到依赖段必失败），非 3.11 宿主直接拦截
        _fail("宿主解释器必须为 Python 3.11", [
            f"当前 {sys.version.split()[0]}；requirements.txt 的固定版本"
            "按 Python 3.11 打包验证，更新的 Python 缺少对应安装包"
            "（如 onnxruntime==1.19.2 在 3.13/3.14 上无 wheel），"
            "装到依赖段必失败",
            "请安装官方 Python 3.11（https://www.python.org/downloads/，"
            "勾选 Add python.exe to PATH）后重跑 install.bat",
            "已装 3.11 但未命中：命令行执行 `py -3.11 --version` 确认；"
            "或把 3.11 的 python.exe 完整路径写入本目录 python_path.txt，"
            "或设置环境变量 XCCV_PY 指向它",
        ])


def detect_gpu(runner=None) -> dict | None:
    """探测 NVIDIA GPU。

    runner=subprocess.run 形参可注入（测试用假 nvidia-smi 输出替代）。
    返回 {"name", "driver", "compute_cap"(float|None)}；探测失败返回 None。
    """
    if runner is None:
        runner = subprocess.run
        exe = shutil.which("nvidia-smi") or r"C:\Windows\System32\nvidia-smi.exe"
        if not Path(exe).is_file() and shutil.which("nvidia-smi") is None:
            return None
    else:
        # 注入替身时跳过本机存在性预检（CI/无 N 卡机器同样可测）
        exe = "nvidia-smi"
    cmd = [exe, "--query-gpu=name,driver_version,compute_cap",
           "--format=csv,noheader"]
    try:
        proc = runner(cmd, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    first = (proc.stdout or "").strip().splitlines()
    if not first:
        return None
    # CSV 单元格可能带引号（含空格的显卡名），逐格去引号与空白
    parts = [p.strip().strip('"') for p in first[0].split(",")]
    while len(parts) < 3:
        parts.append("")
    cap = None
    m = re.match(r"(\d+)\.(\d+)", parts[2])
    if m:
        cap = int(m.group(1)) * 10 + int(m.group(2))
    driver = None
    m = re.match(r"(\d+)", parts[1])
    if m:
        driver = int(m.group(1))
    return {"name": parts[0], "driver": driver, "compute_cap": cap}


def select_torch_index(gpu: dict | None, override: str = "",
                       torch_version: str = TORCH_VERSION) -> tuple:
    """根据 GPU/驱动选择 torch 安装源，返回 (源名, index URL, 选择原因)。

    口径：RTX 50 系（算力 ≥ sm_100）→ cu128；其余 N 卡（驱动 ≥ 525）→ cu126；
    无 N 卡或驱动过旧 → cpu。override 非空时直接采用（名字或完整 URL）。
    torch ≥ 2.10 违反 ultralytics 8.4.7 上界约束，此处直接拒绝。
    """
    parts = torch_version.split(".")
    if len(parts) < 2 or (int(parts[0]), int(parts[1])) >= ULTRALYTICS_TORCH_UPPER:
        _fail("torch 版本与 ultralytics 冲突", [
            f"请求的 torch=={torch_version} 不满足 ultralytics 8.4.7 的"
            "上界约束 torch<2.10",
            "请使用默认版本矩阵：torch==2.9.1 + torchvision==0.24.1",
        ])
    if override:
        name = override.strip().lower()
        if name in TORCH_INDEXES:
            return name, TORCH_INDEXES[name], f"命令行指定源 {name}"
        return "custom", override, f"命令行指定 URL {override}"
    if gpu is None:
        return ("cpu", TORCH_INDEXES["cpu"],
                "未检测到 NVIDIA 显卡/驱动，使用 CPU 源")
    cap, driver = gpu.get("compute_cap"), gpu.get("driver")
    if cap is not None and cap >= SM_REQUIRED_CU128:
        return ("cu128", TORCH_INDEXES["cu128"],
                f"{gpu['name']} 算力 sm_{cap}（Blackwell），必须 CUDA 12.8 源")
    if driver is not None and driver < MIN_DRIVER_CU:
        return ("cpu", TORCH_INDEXES["cpu"],
                f"驱动 {driver} 过旧（cu12x 轮子需 ≥ {MIN_DRIVER_CU}），"
                "先用 CPU 源；建议升级 NVIDIA 驱动后重跑换 CUDA 源")
    if driver is not None and driver >= 570:
        return ("cu128", TORCH_INDEXES["cu128"],
                f"{gpu['name']}（驱动 {driver}），选最新 CUDA 12.8 源")
    return ("cu126", TORCH_INDEXES["cu126"],
            f"{gpu['name']}（驱动 {driver}，算力 "
            f"sm_{cap if cap is not None else '?'}），选 CUDA 12.6 源")


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _py_version_tuple(py: Path) -> tuple[int, int] | None:
    """查询解释器的 (major, minor)；查询失败返回 None。"""
    try:
        proc = subprocess.run(
            [str(py), "-c",
             "import sys; print('%d.%d' % sys.version_info[:2])"],
            capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.match(r"(\d+)\.(\d+)", (proc.stdout or "").strip())
    return (int(m.group(1)), int(m.group(2))) if m else None


def ensure_venv(venv_dir: Path) -> Path:
    """创建（或复用）仓库根 venv，返回 venv 解释器路径。

    复用前校验 venv 的 Python 版本：与依赖矩阵锚定版本不符的旧环境
    （疑为此前其他 Python 所建）直接复用会在依赖段失败——改名留档后
    按宿主解释器重建（改名非删除，可恢复）。"""
    py = _venv_python(venv_dir)
    if py.is_file():
        ver = _py_version_tuple(py)
        if ver is not None and ver != HOST_PYTHON:
            backup = venv_dir.with_name(
                f"{venv_dir.name}_py{ver[0]}{ver[1]}_old")
            try:
                venv_dir.rename(backup)
            except OSError as exc:
                _fail("已有虚拟环境版本不符", [
                    f"{venv_dir} 是 Python {ver[0]}.{ver[1]}，依赖矩阵按 "
                    f"{HOST_PYTHON[0]}.{HOST_PYTHON[1]} 固定——直接复用"
                    "会在依赖段失败",
                    f"自动改名留档失败（{type(exc).__name__}: {exc}）："
                    f"请手动删除或改名 {venv_dir} 后重跑 install.bat",
                ])
            print(f"[重建] 已有虚拟环境是 Python {ver[0]}.{ver[1]}（与矩阵 "
                  f"{HOST_PYTHON[0]}.{HOST_PYTHON[1]} 不符，疑为此前安装"
                  f"所建），已改名留档：{backup.name}；按宿主解释器重建")
        else:
            print(f"[复用] 已存在虚拟环境：{venv_dir}"
                  + (f"（Python {ver[0]}.{ver[1]}）" if ver else ""))
            return py
    _step("创建虚拟环境")
    print(f"目标目录：{venv_dir}")
    try:
        builder = venv.EnvBuilder(with_pip=True, clear=False)
        builder.create(str(venv_dir))
    except OSError as exc:
        _fail("虚拟环境创建失败", [
            f"{type(exc).__name__}: {exc}",
            f"请确认 {venv_dir.parent} 目录可写（右键 属性 取消只读；"
            "或把仓库移到有写权限的目录，如 D:\\XCcv），"
            "杀毒软件拦截时添加目录白名单后重试",
        ])
    if not py.is_file():
        _fail("虚拟环境创建失败", [
            f"未找到 {py}（venv 模块未产出解释器）",
            "请确认系统 Python 安装完整（重装时勾选 Install launcher for all users）",
        ])
    print(f"[完成] venv 解释器：{py}")
    return py


def _run_pip(args: list, proxy_hint: str) -> None:
    """流式执行 pip（实时透传输出）；失败按无网/代理/版本不匹配分支给中文指引。"""
    print("$ " + " ".join(args))
    pip_log = Path(tempfile.gettempdir()) / "xccv-pip-last.log"
    try:
        proc = subprocess.run([*args, "--log", str(pip_log)])
    except OSError as exc:
        _fail("pip 启动失败", [f"{type(exc).__name__}: {exc}",
                               "请确认虚拟环境未被杀毒软件隔离"])
    if proc.returncode == 0:
        return
    lines = ["pip 退出码 " + str(proc.returncode)]
    tail = ""
    try:
        tail = pip_log.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        pass
    if ("No matching distribution found for" in tail
            or "Could not find a version that satisfies" in tail):
        lines.append("pip 报「无匹配版本」：当前 Python 缺少依赖矩阵对应的"
                     "安装包（矩阵按 Python 3.11 固定），并非网络问题")
        lines.append("请安装官方 Python 3.11 后重跑 install.bat"
                     "（见上方宿主解释器体检的指引）")
    proxy_env = [k for k in ("HTTP_PROXY", "HTTPS_PROXY")
                 if os.environ.get(k)]
    if proxy_env:
        lines.append("检测到代理环境变量 " + ",".join(proxy_env) +
                     "：确认代理可达；公司代理需加 pip 信任源参数或离线安装")
    else:
        lines.append("未检测到代理变量：网络不可达时设置代理后重试，如")
        lines.append("      set HTTPS_PROXY=http://公司代理:端口")
    lines.append(proxy_hint)
    _fail("依赖安装失败", lines)


def install_dependencies(venv_py: Path, root: Path, index_url: str,
                         torch_version: str, torchvision_version: str) -> None:
    """两段安装：torch（按源固定版本）→ requirements.txt（不含 torch）。"""
    _step(f"安装 PyTorch（torch=={torch_version} torchvision=={torchvision_version}）")
    _run_pip([str(venv_py), "-m", "pip", "install", "--upgrade", "pip"],
             "pip 自升级失败一般不影响后续安装，可忽略后重跑")
    _run_pip([str(venv_py), "-m", "pip", "install",
              f"torch=={torch_version}", f"torchvision=={torchvision_version}",
              "--index-url", index_url],
             "torch 安装包约 2~3 GB，低速网络建议改用离线安装（--offline 查看指引）")
    _step("安装其余依赖（requirements.txt）")
    _run_pip([str(venv_py), "-m", "pip", "install",
              "-r", str(root / "requirements.txt")],
             "无内网 PyPI 通道的机器请用离线安装（--offline 查看指引）")


def write_python_path(venv_py: Path, root: Path) -> None:
    """写 python_path.txt（一行 venv 解释器路径），启动脚本按它优先选中。"""
    cfg = root / "python_path.txt"
    try:
        cfg.write_text(str(venv_py) + "\n", encoding="utf-8")
    except OSError as exc:
        _fail("python_path.txt 写入失败", [f"{type(exc).__name__}: {exc}",
                                          f"请确认 {root} 可写"])
    print(f"[完成] 已写入 {cfg} -> {venv_py}")


def run_env_check(root: Path) -> None:
    """运行 tools/env_check.py 体检（中文明细），并给一行英文结论。"""
    _step("环境体检")
    check = root / "tools" / "env_check.py"
    proc = subprocess.run([sys.executable, str(check)])
    if proc.returncode == 0:
        print("Environment check PASSED - double-click run_main bat / "
              "run_main.bat to launch the workbench.")
    else:
        print("Environment check FAILED - see the Chinese diagnosis above; "
              "run install.bat again after fixing.")
        raise SystemExit(1)


def print_offline_guide(index_name: str) -> None:
    """离线机安装指引：在联网机 pip download，离线机 --no-index 安装。"""
    _step("离线安装指引")
    index = TORCH_INDEXES.get(index_name, TORCH_INDEXES["cpu"])
    print(OFFLINE_GUIDE.format(torch=TORCH_VERSION, tv=TORCHVISION_VERSION,
                               index=index))


def main(argv: list | None = None) -> int:
    _safe_console()
    ap = argparse.ArgumentParser(
        description="XCcv 一键环境安装器（GPU 探测 / torch 源选择 / venv / "
                    "依赖安装 / 环境体检）")
    ap.add_argument("--root", default=str(ROOT_DEFAULT),
                    help="仓库根目录（默认为本脚本上级目录）")
    ap.add_argument("--venv-name", default="venv",
                    help="虚拟环境目录名（默认 venv，置于仓库根）")
    ap.add_argument("--index", default="",
                    help="torch 安装源：cu128 / cu126 / cpu 或完整 URL"
                         "（缺省按 GPU 自动选择）")
    ap.add_argument("--torch", default=TORCH_VERSION,
                    help=f"torch 版本（默认 {TORCH_VERSION}）")
    ap.add_argument("--torchvision", default=TORCHVISION_VERSION,
                    help=f"torchvision 版本（默认 {TORCHVISION_VERSION}）")
    ap.add_argument("--offline", action="store_true",
                    help="只打印离线安装指引，不做任何安装动作")
    ap.add_argument("--skip-check", action="store_true",
                    help="最后不运行 env_check.py 体检")
    args = ap.parse_args(argv)

    # 防御性清洗：命令行展开可能让尾反斜杠转义闭引号（如 --root "E:\dir\"），
    # 路径里带入 " 或以 \ 结尾——去引号、去尾反斜杠后再解析
    root = Path(args.root.strip('"').rstrip('\\') or '.').resolve()
    if not (root / "requirements.txt").is_file():
        _fail("仓库根定位失败", [
            f"{root} 下没有 requirements.txt",
            "请在仓库根目录运行 install.bat，或用 --root 指定仓库根",
        ])
    print("=" * 62)
    print("XCcv 一键环境安装")
    print(f"仓库根：{root}")
    print(f"宿主解释器：{sys.executable}")
    print(f"版本矩阵：torch=={args.torch} torchvision=={args.torchvision}"
          "（ultralytics 8.4.7 约束 torch<2.10）")
    print("=" * 62)

    check_host_python()

    if args.offline:
        index_name, _, _ = select_torch_index(
            detect_gpu(), args.index, args.torch)
        print_offline_guide(index_name)
        return 0

    gpu = None
    if not args.index:
        _step("GPU 探测")
        gpu = detect_gpu()
        if gpu is None:
            print("[CPU] 未检测到 NVIDIA 显卡/驱动：将安装 CPU 版 torch"
                  "（平台全部功能可用，训练与推理速度受限）")
        else:
            print(f"[GPU] {gpu['name']}  驱动 {gpu.get('driver')}  "
                  f"算力 sm_{gpu.get('compute_cap')}")
    index_name, index_url, reason = select_torch_index(
        gpu, args.index, args.torch)
    print(f"[torch 源] {index_name} -> {index_url}")
    if not args.index:
        print(f"（选择原因：{reason}）")

    venv_py = ensure_venv(root / args.venv_name)
    install_dependencies(venv_py, root, index_url,
                         args.torch, args.torchvision)
    write_python_path(venv_py, root)
    if not args.skip_check:
        run_env_check(root)
    print("")
    print("全部完成：双击 run_main.bat 打开主界面；权重可按需运行")
    print("  python tools\\download_weights.py --list   查看并补齐。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
