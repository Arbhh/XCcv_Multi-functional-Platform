# -*- coding: utf-8 -*-
"""XCcv 权重一键下载 CLI（纯命令行，不依赖 Qt）。

需求：用户装好环境后一条命令补齐全链路权重——YOLO 预训练 / SAM 系列（AI 智能标注）
/ DINO ViT-B/8 与 DINOv2（CIF / SubspaceAD）/ R-CNN 骨干初始权重；
--list 打印清单（落点/状态/来源/许可），--family 按族下载，
已存在且校验通过的权重跳过（断点续装提示）。

实现：
- 与运行时接管共用 engine.weights.ensure_family_weights 的注册表与下载
  核心（落点/多源 URL/大小下限/容器头校验/许可声明一致——--list 清单即
  运行时探测集合）；
- YOLO 预训练复用 engine.weights.ensure_pretrained_weight 的落位/多源/
  SHA-256 校验/notify(text, frac) 进度模式（与训练链路同一口径）；
- 其余权重复用同一 fetcher（连接/读取超时 + 单源耗时上限）与
  "大小下限 + 容器头（PK/\x80）+ 已知 SHA-256"校验口径；
- DINOv2 走 huggingface_hub 快照下载（共享层保留 HF_ENDPOINT 镜像口径）；
- 权重不随包分发：每族打印来源与许可声明，fetcher 可注入（测试不真实联网）。

使用：python tools/download_weights.py [--list] [--family yolo|sam|dino|rcnn|all]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from xccv.engine.offline import WEIGHTS_DIR  # noqa: E402
from xccv.engine.weights.ensure_family_weights import (  # noqa: E402
    _entry_ok, dino_entries, dinov2_discovered_dir, download_named,
    entries_for, ensure_dinov2, hf_local_dir)
from xccv.engine.weights.ensure_weight import (  # noqa: E402
    ensure_pretrained_weight)

YOLO_NAMES = ("yolo11n.pt", "yolo11s.pt", "yolo11n-seg.pt", "yolo11s-seg.pt")


def _cli_notify(state: dict, text: str, frac) -> None:
    """CLI 进度回调：notify(text, frac) 口径；按 10% 步进打印，不刷屏。"""
    if frac is None:
        print(text)
        return
    pct = int(min(1.0, max(0.0, frac)) * 100)
    last = state.get("last_pct", -1)
    if pct >= 100 or pct - last >= 10:
        state["last_pct"] = pct
        print(f"    {pct}%")


def download_generic(name: str, dest: Path, urls: tuple, min_bytes: int,
                     check_head: bool, notify_state: dict,
                     fetcher=None) -> str:
    """多源下载 + 校验 + 已存在跳过（共享核心的 CLI 节流包装）。"""
    return download_named(name, dest, urls, min_bytes, check_head,
                          notify=lambda t, f: _cli_notify(notify_state, t, f),
                          fetcher=fetcher)


def _notify_of(state: dict):
    return lambda t, f: _cli_notify(state, t, f)


def family_yolo(notify_state: dict, fetcher=None) -> int:
    """YOLO 预训练：复用 ensure_pretrained_weight（落位/多源/SHA-256 同口径）。"""
    print("YOLO 预训练（ultralytics 官方资产 + 镜像；"
          "ultralytics AGPL-3.0，权重许可以官方发布页为准）")
    done = 0
    for name in YOLO_NAMES:
        try:
            path = ensure_pretrained_weight(
                name, notify=_notify_of(notify_state), fetcher=fetcher)
            print(f"    就绪 {name} -> {path}")
            done += 1
        except RuntimeError as exc:
            print(f"    [失败] {exc}")
    return done


def family_sam(notify_state: dict, fetcher=None) -> int:
    """SAM 家族：AI 智能标注权重落到 weights/sam/（注册表驱动）。"""
    print("SAM 系列（AI 智能标注）：")
    print(f"  许可/来源：{entries_for('sam')[0].license}")
    done = 0
    for e in entries_for("sam"):
        try:
            download_generic(e.key, e.dest, e.urls, e.min_bytes, e.check_head,
                             notify_state, fetcher)
            done += 1
        except RuntimeError as exc:
            print(f"    [失败] {exc}")
    return done


def family_dino(notify_state: dict, fetcher=None) -> int:
    """DINO ViT-B/8（CIF 特征底座）+ DINOv2（SubspaceAD 骨干）。"""
    print("DINO / DINOv2（CIF 与 SubspaceAD 特征底座）：")
    print(f"  许可/来源：{dino_entries()[0].license}")
    done = 0
    e = dino_entries()[0]
    try:
        download_generic(e.key, e.dest, e.urls, e.min_bytes, e.check_head,
                         notify_state, fetcher)
        done += 1
    except RuntimeError as exc:
        print(f"    [失败] {exc}")
    # DINOv2：HF 快照下载（镜像口径在共享层：HF_ENDPOINT 未设时缺省 hf-mirror）
    hf_entry = dino_entries()[1]
    print(f"    下载 {hf_entry.key}（HuggingFace 快照；国内网络镜像口径见"
          " HF_ENDPOINT）…")
    if ensure_dinov2(_notify_of(notify_state)):
        done += 1
    else:
        print("    [失败] DINOv2 快照下载失败。请确认网络；被墙环境保持"
              " HF_ENDPOINT=https://hf-mirror.com（缺省已启用）后重试，"
              "或手动从官方页下载模型目录放入：")
        print(f"      {hf_local_dir()}")
    return done


def family_rcnn(notify_state: dict, fetcher=None) -> int:
    """R-CNN / 分类骨干初始权重（torchvision 官方枚举 URL，注册表驱动）。"""
    print("Mask R-CNN / 分类骨干初始权重（torchvision 官方发布）：")
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError:
        print("    [失败] 未安装 torch/torchvision：请先运行 install.bat "
              "或手动安装依赖后重试")
        return 0
    print(f"  许可/来源：{entries_for('rcnn')[0].license}")
    print(f"    torchvision 权重缓存目录：torch.hub（TORCH_HOME 重定向到随包"
          f"权重目录 {WEIGHTS_DIR / 'torch'}）")
    done = 0
    for e in entries_for("rcnn"):
        urls = e.resolved_urls
        if not urls:
            # 当前 torchvision 版本不含该权重枚举（旧版无 V2 等）——跳过不报错
            print(f"    [跳过] {e.label}：当前 torchvision 无此权重枚举")
            continue
        try:
            download_generic(e.key or e.label, e.dest, urls, e.min_bytes,
                             e.check_head, notify_state, fetcher)
            done += 1
        except RuntimeError as exc:
            print(f"    [失败] {exc}")
    slim = WEIGHTS_DIR / "maskrcnn18_slim_init.pt"
    if slim.is_file():
        print("    已就绪（跳过）：maskrcnn18_slim_init.pt")
    else:
        print("    maskrcnn18 slim 初始权重未就绪：运行 "
              "python tools/build_maskrcnn18_init.py 自建（不下载）")
    return done


FAMILY_FUNCS = {
    "yolo": family_yolo,
    "sam": family_sam,
    "dino": family_dino,
    "rcnn": family_rcnn,
}


def _list_all() -> None:
    """--list：打印全部权重清单（落点 / 状态 / 用途 / 许可），不下载。"""
    print("XCcv 权重清单（--family yolo|sam|dino|rcnn 按族下载；"
          "全部权重不随包分发，统一落引擎权重目录 "
          f"{WEIGHTS_DIR}，许可以各发布方官方页面为准）")
    print("")
    print("yolo 族（训练用 YOLO 预训练，落 ultralytics/）")
    for name in YOLO_NAMES:
        dest = WEIGHTS_DIR / "ultralytics" / name
        ok = _entry_ok(dest, 100 * 1024, True)
        print(f"  [{'已就绪' if ok else '缺失'}] {name}")
    for fam, title in (
            ("sam", "sam 族（AI 智能标注，落 sam/）"),
            ("dino", "dino 族（CIF / SubspaceAD 特征底座，落 dino/ 与 dinov2/）"),
            ("rcnn", "rcnn 族（R-CNN / 分类骨干初始权重，落 torch/hub/checkpoints/）")):
        print("")
        print(title)
        for e in entries_for(fam):
            if e.kind == "hf":
                # DINOv2：按 subspacead 发现口径优先显示已有本地目录
                target = str(dinov2_discovered_dir() or hf_local_dir())
            else:
                target = "（torchvision 枚举解析后确定）"
                try:
                    target = str(e.dest)
                except RuntimeError:
                    pass
            print(f"  [{'已就绪' if e.is_ready() else '缺失'}] {e.key or e.label}"
                  f"  {e.label}  -> {target}")
    print("")
    print("  maskrcnn18_slim_init.pt 由 tools/build_maskrcnn18_init.py 自建"
          f"（落引擎权重目录，当前{'已就绪' if (WEIGHTS_DIR / 'maskrcnn18_slim_init.pt').is_file() else '缺失'}）")


def main(argv: list | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(
        description="XCcv 权重一键下载（YOLO / SAM / DINO / DINOv2 / "
                    "R-CNN 骨干初始权重；--list 查看清单）")
    ap.add_argument("--list", action="store_true",
                    help="只打印权重清单与就绪状态，不下载")
    ap.add_argument("--family", default="all",
                    choices=["all", "yolo", "sam", "dino", "rcnn"],
                    help="按族下载（默认 all）")
    args = ap.parse_args(argv)

    if args.list:
        _list_all()
        return 0

    families = list(FAMILY_FUNCS) if args.family == "all" else [args.family]
    state: dict = {"last_pct": -1}
    for fam in families:
        print("=" * 62)
        done = FAMILY_FUNCS[fam](state)
        print(f"族 {fam}：{done} 项就绪")
    print("=" * 62)
    print("处理结束：缺失项按上方指引重试或手动下载放入对应目录。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
