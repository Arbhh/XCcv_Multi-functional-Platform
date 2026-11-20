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
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT_S = 1800


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
                              text=True, timeout=TIMEOUT_S,
                              encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {TIMEOUT_S}s"
    ok = proc.returncode == 0
    tail = "\n".join((proc.stdout or "").splitlines()[-3:] +
                     (proc.stderr or "").splitlines()[-8:])
    return ok, tail


def main() -> int:
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
