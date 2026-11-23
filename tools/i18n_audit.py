# -*- coding: utf-8 -*-
"""i18n AST 对账审计（报告制，不阻断）。

扫描 xccv/ 全部 Python 源码中的中文字符串字面量（AST 解析）：
- A 类：tr() 站点键不在 i18n_en.TABLE —— 英文模式将回退中文；
- B 类：未包 tr() 的中文裸字面量（按出现频次排序）——英文模式必然漏翻；
- C 类：TABLE 中从未被任何站点引用的键（增量清理线索）。

用法：python tools/i18n_audit.py [--out FILE]
输出摘要到 stdout；--out 时同时写明细报告。始终退出 0（报告制）。
"""
from __future__ import annotations

import argparse
import ast
import collections
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_CJK = re.compile(r"[\u4e00-\u9fff]")
SKIP_DIRS = {"__pycache__", "venv", ".venv", "platform_data", "weights",
             "runs", "testdata", ".git", ".idea", "static"}


def cjk(s: str) -> bool:
    return bool(_CJK.search(s or ""))


def iter_py_files():
    base = os.path.join(ROOT, "xccv")
    for dp, dn, fn in os.walk(base):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        for f in fn:
            if f.endswith(".py"):
                yield os.path.join(dp, f)


def audit() -> dict:
    from xccv.i18n_en import TABLE

    tr_keys: set = set()                    # tr() 站点键
    tr_missing: collections.Counter = collections.Counter()
    raw_sites: collections.Counter = collections.Counter()   # 未包 tr 的中文
    raw_detail: dict = {}
    table_used: set = set()

    for path in iter_py_files():
        rel = os.path.relpath(path, ROOT).replace("\\", "/")
        is_table_self = rel.replace("/", os.sep).endswith(
            os.path.join("xccv", "i18n_en.py"))
        try:
            with io.open(path, "r", encoding="utf-8", errors="replace") as fh:
                tree = ast.parse(fh.read())
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                name = f.id if isinstance(f, ast.Name) else (
                    f.attr if isinstance(f, ast.Attribute) else "")
                if name in ("tr", "_tr", "tr_"):
                    for a in node.args:
                        if isinstance(a, ast.Constant) \
                                and isinstance(a.value, str) \
                                and cjk(a.value):
                            tr_keys.add(a.value)
                            table_used.add(a.value)
                            if a.value not in TABLE:
                                tr_missing[a.value] += 1
            elif isinstance(node, ast.Constant) \
                    and isinstance(node.value, str) and cjk(node.value) \
                    and not is_table_self:
                # i18n_en.py 自身的 TABLE 键即中文串，不计入 B 类
                raw_sites[node.value] += 1
                raw_detail.setdefault(node.value, f"{rel}:{node.lineno}")

    table_keys = set(TABLE)
    unwrapped = {k: n for k, n in raw_sites.items() if k not in tr_keys}
    unused_keys = sorted(table_keys - table_used)
    return {
        "tr_sites": len(tr_keys),
        "tr_missing": tr_missing,
        "raw_total": len(raw_sites),
        "unwrapped": unwrapped,
        "raw_detail": raw_detail,
        "table_keys": len(table_keys),
        "unused_keys": unused_keys,
    }


def main() -> int:
    # 报告含中文；Windows 控制台默认代码页（cp1252 等）打印即
    # UnicodeEncodeError（CI 实测），统一 UTF-8 输出
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="明细报告输出文件")
    ap.add_argument("--top", type=int, default=30, help="展示未包 tr 高频串条数")
    args = ap.parse_args()

    r = audit()
    print("== i18n AST 对账（xccv/ 全量）==")
    print("tr() 站点键:", r["tr_sites"], "  TABLE 键:", r["table_keys"])
    print("A 类 tr() 缺 TABLE 键（英文回退中文）:", sum(r["tr_missing"].values()))
    for k, n in r["tr_missing"].most_common(args.top):
        print("   A", n, repr(k[:60]))
    print("中文裸字面量站点:", r["raw_total"],
          "  其中未包 tr 的唯一串:", len(r["unwrapped"]))
    for k, n in sorted(r["unwrapped"].items(), key=lambda kv: -kv[1])[:args.top]:
        print("   B", n, repr(k[:60]), "@", r["raw_detail"].get(k, ""))
    print("C 类 TABLE 未引用键:", len(r["unused_keys"]))
    for k in r["unused_keys"][:args.top]:
        print("   C", repr(k[:60]))

    if args.out:
        with io.open(args.out, "w", encoding="utf-8") as fh:
            fh.write("# i18n AST 对账报告\n\n")
            fh.write("- tr() 站点键: %d\n- TABLE 键: %d\n" %
                     (r["tr_sites"], r["table_keys"]))
            fh.write("- A 类 tr() 缺 TABLE 键: %d\n" % sum(r["tr_missing"].values()))
            for k, n in r["tr_missing"].most_common():
                fh.write("  - A %d %r\n" % (n, k))
            fh.write("- 未包 tr 的中文裸字面量（唯一串）: %d\n" % len(r["unwrapped"]))
            for k, n in sorted(r["unwrapped"].items(), key=lambda kv: -kv[1]):
                fh.write("  - B %d %r @ %s\n" %
                         (n, k, r["raw_detail"].get(k, "")))
            fh.write("- C 类 TABLE 未引用键: %d\n" % len(r["unused_keys"]))
            for k in r["unused_keys"]:
                fh.write("  - C %r\n" % k)
        print("report ->", args.out)
    return 0          # 报告制：始终不阻断


if __name__ == "__main__":
    raise SystemExit(main())
