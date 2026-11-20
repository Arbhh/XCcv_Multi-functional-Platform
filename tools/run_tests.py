"""逐模块子进程测试运行器：把 tests/ 下每个测试模块放入独立子进程执行。

Windows 上单进程连跑全套 Qt 测试会撞 Win32 计时器/资源表上限（表现为
0xC00000FD 级崩溃），逐模块子进程隔离是最稳定的跑法；同时天然隔离模块级
全局状态（语言偏好、单实例守卫等）。

用法：
    python tools/run_tests.py                 # 跑全部 tests/test_*.py
    python tools/run_tests.py tests.foo tests.bar
    python tools/run_tests.py --list          # 只列出模块

任一模块失败时进程退出码为 1，并在末尾汇总失败清单。
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT_S = 1800

# 子进程统一环境（本地与 CI 同一前提，随运行器自动带齐，无需手工设置）：
# - PYTHONIOENCODING：Windows runner 控制台为 cp1252，子进程 print 中文即
#   UnicodeEncodeError（0.8.0 CI 实测），统一 UTF-8 与运行器读取编码对齐；
# - QT_QPA_PLATFORM：无显示器环境默认 offscreen（各测试模块亦自行 setdefault）；
# - MPLBACKEND：Linux 无 DISPLAY 时避免散用 pyplot 的模块解析到 Qt 后端
#   （charts.py 强制 Qt5Agg 的画布不受此默认值影响）。
_CHILD_ENV = {
    "PYTHONIOENCODING": "utf-8",
    "QT_QPA_PLATFORM": "offscreen",
    "MPLBACKEND": "Agg",
}

_VCOMP_SITE = (
    "# vcomp140 预加载（tools/run_tests.py 生成）：先 import sklearn"
    "（其 .libs 载入新版\n# VC++ OpenMP 运行库），规避旧版系统运行库下"
    " PyQt5 -> torch 的 c10.dll 初始化\n# 失败（WinError 1114）。\n"
    "try:\n"
    "    import sklearn  # noqa: F401\n"
    "except Exception:\n"
    "    pass\n"
)


def _vcomp_preload_dir() -> str | None:
    """Windows：生成 sitecustomize 预载目录，规避 PyQt5 先于 torch 载入时
    旧版 VC++ OpenMP 运行库导致 c10.dll 初始化失败（WinError 1114）；
    与本机 vcomp_preload 方案同一机理，CI 与本地共用。"""
    if sys.platform != "win32":
        return None
    d = Path(tempfile.gettempdir()) / "xccv_vcomp_preload"
    try:
        d.mkdir(exist_ok=True)
        site = d / "sitecustomize.py"
        if not site.exists() or site.read_text(encoding="utf-8") != _VCOMP_SITE:
            site.write_text(_VCOMP_SITE, encoding="utf-8")
        return str(d)
    except OSError:
        return None


def _child_env() -> dict:
    env = os.environ.copy()
    for key, value in _CHILD_ENV.items():
        env.setdefault(key, value)
    preload = _vcomp_preload_dir()
    if preload:
        pp = env.get("PYTHONPATH")
        env["PYTHONPATH"] = f"{preload}{os.pathsep}{pp}" if pp else preload
    return env


def discover_modules() -> list[str]:
    mods = []
    for p in sorted((ROOT / "tests").glob("test_*.py")):
        mods.append(f"tests.{p.stem}")
    return mods


def run_one(module: str) -> tuple[bool, str]:
    """子进程跑单个测试模块，返回 (是否通过, 输出尾部)。"""
    cmd = [sys.executable, "-m", "unittest", module, "-v"]
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                              text=True, timeout=TIMEOUT_S, env=_child_env(),
                              encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {TIMEOUT_S}s"
    ok = proc.returncode == 0
    # 失败时多带一些上下文，便于不进 CI 控制台也能定位（stdout 尾 6 行 +
    # stderr 尾 20 行）
    tail = "\n".join((proc.stdout or "").splitlines()[-6:] +
                     (proc.stderr or "").splitlines()[-20:])
    return ok, tail


def main() -> int:
    # 运行器自身输出统一 UTF-8：Windows runner 控制台为 cp1252，直接 print
    # 子进程捕获到的中文/替换字符会 UnicodeEncodeError（0.8.0 CI 于 77/97
    # 处中断过整轮）
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("modules", nargs="*", help="指定模块（默认全部）")
    ap.add_argument("--list", action="store_true", help="只列出模块名")
    args = ap.parse_args()

    mods = args.modules or discover_modules()
    if args.list:
        print("\n".join(mods))
        return 0

    failed: list[str] = []
    for i, m in enumerate(mods, 1):
        ok, tail = run_one(m)
        mark = "PASS" if ok else "FAIL"
        print(f"[{i}/{len(mods)}] {mark}  {m}", flush=True)
        if not ok:
            failed.append(m)
            for line in tail.splitlines():
                print("      " + line, flush=True)

    print("=" * 60)
    print(f"total {len(mods)}  failed {len(failed)}")
    for m in failed:
        print("  FAIL", m)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
