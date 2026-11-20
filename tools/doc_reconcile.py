"""文档发布对账工具：README/SECURITY 对外口径 vs 代码事实。

- 算子数：流程引擎内置 ops 子模块实际注册数（不含自定义算子）；
- 节点数：内置 ops + 旧链兼容委托节点的总和（前端画布展示口径）；
- 模型数：engine models 注册表 list_models() 全量数；
- SECURITY.md 支持版本表须含当前支持线（0.8.x）。

用法：python tools/doc_reconcile.py [--root 仓库根]
缺省 root = 本脚本所在仓库根。口径不符退出码 1，
并打印应写入的数字（人工/脚本按输出修正文档后重跑应全绿）。
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xccv.engine.flow.registry import registered_names  # noqa: E402
from xccv.engine.models.registry import list_models  # noqa: E402

BUILTIN_OPS_MODULES = ("ai_bridge", "crop", "det_filters", "external_ops",
                       "listops", "locate", "logic", "measure", "output",
                       "preprocess", "source")
CURRENT_SUPPORT_LINE = "0.8.x"


def builtin_ops_count() -> int:
    import xccv.engine.flow.ops as ops_pkg

    for m in BUILTIN_OPS_MODULES:
        __import__(f"xccv.engine.flow.ops.{m}")
    return len(registered_names())


def all_nodes_count() -> int:
    """内置 ops + 旧链兼容委托节点（前端节点库总数）。"""
    builtin_ops_count()
    __import__("xccv.engine.flow.legacy_adapter")
    return len(registered_names())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    args = ap.parse_args()
    root = Path(args.root)

    ops_n = builtin_ops_count()
    nodes_n = all_nodes_count()
    models_n = len(list_models())
    print(f"代码事实：内置算子 {ops_n} 个；画布节点 {nodes_n} 个；内置模型 {models_n} 种")

    problems = []

    readme = (root / "README.md").read_text(encoding="utf-8")
    for m in re.finditer(r"(\d+)\s*个\s*AI 与通用算子", readme):
        if int(m.group(1)) != ops_n:
            problems.append(f"README.md 算子数 {m.group(1)} ≠ {ops_n}")
    for m in re.finditer(r"(\d+)\s*个节点", readme):
        if int(m.group(1)) != nodes_n:
            problems.append(f"README.md 节点数 {m.group(1)} ≠ {nodes_n}")
    for m in re.finditer(r"(\d+)\s*种内置模型", readme):
        if int(m.group(1)) != models_n:
            problems.append(f"README.md 模型数 {m.group(1)} ≠ {models_n}")

    readme_en = (root / "README_EN.md").read_text(encoding="utf-8")
    for m in re.finditer(r"(\d+)\s*AI & common operators", readme_en):
        if int(m.group(1)) != ops_n:
            problems.append(f"README_EN.md operators {m.group(1)} ≠ {ops_n}")
    for m in re.finditer(r"(\d+)\s*built-in models", readme_en):
        if int(m.group(1)) != models_n:
            problems.append(f"README_EN.md models {m.group(1)} ≠ {models_n}")
    for m in re.finditer(r"\((\d+) nodes", readme_en):
        if int(m.group(1)) != nodes_n:
            problems.append(f"README_EN.md nodes {m.group(1)} ≠ {nodes_n}")

    sec = (root / "SECURITY.md").read_text(encoding="utf-8")
    if CURRENT_SUPPORT_LINE not in sec:
        problems.append(f"SECURITY.md 缺当前支持线 {CURRENT_SUPPORT_LINE}")

    if problems:
        print("口径漂移（请修文档至下述事实后重跑）：")
        for p in problems:
            print("  -", p)
        return 1
    print("对账全绿：README/README_EN/SECURITY 与代码事实一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
