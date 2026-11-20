# -*- coding: utf-8 -*-
"""剪枝子进程入口（P7/T7-1 强制防死机项）。

用法（父窗口 QProcess 启动）::

    python -m xccv.engine.prune_worker --mode post --src ... --dst ... --ratio 0.3
    python -m xccv.engine.prune_worker --mode pre  --dataset-dir ... --task detection \
        --model-name yolo11n --width 0 --height 0 --ratio 0.3 --dst ...
    python -m xccv.engine.prune_worker --selftest

协议（stdout，每行一个 JSON 对象，写后立即 flush）::

    {"event": "phase", "name": "scan|load|trace|apply|cleanup|save", "detail": "..."}
    {"event": "done",  "result": {prune_checkpoint / pretrain_prune 的返回 dict}}
    {"event": "error", "message": "..."}

退出码：0=done；1=error（未捕获异常）；2=用法错误。
- **CPU 强制**：入口先把 CUDA_VISIBLE_DEVICES 置空（先于一切 torch 导入）——
  剪枝全程 CPU，不占显存、不与训练/推理抢卡；
- 主窗口进程零剪枝内存：torch/torch_pruning/模型权重全部活在本子进程，
  内存失控时父窗口按水位自动终止（psutil 轮询），本进程无从拖垮界面；
- --sleep-after PHASE:SECONDS（隐藏参数，仅供测试）：发出该相位后挂起指定
  秒数——父窗口的紧急终止/水位自动终止测试需要一个"可被杀中的"子进程。
"""

import argparse
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 必须先于任何 torch 导入

# 允许任意工作目录启动（launcher/platform_entry 不保证 cwd=项目根）
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _emit(**kw) -> None:
    try:
        line = json.dumps(kw, ensure_ascii=False)
    except TypeError:
        # 防御：结果混入不可序列化对象时降级为字符串，保证协议行可产出
        line = json.dumps(
            {k: (v if k == "event" else str(v)) for k, v in kw.items()}
            | {"stripped": True},
            ensure_ascii=False)
    print(line, flush=True)


def _progress_factory(sleep_map: dict):
    """把 pruning 的 progress 回调接到 JSON 行协议；支持测试挂起钩子。"""

    def _progress(phase: str, detail: str = "") -> None:
        _emit(event="phase", name=phase, detail=detail)
        secs = sleep_map.get(phase)
        if secs:
            time.sleep(secs)

    return _progress


def main() -> int:
    ap = argparse.ArgumentParser(prog="xccv.engine.prune_worker")
    ap.add_argument("--mode", choices=["pre", "post"])
    ap.add_argument("--src")
    ap.add_argument("--dst")
    ap.add_argument("--ratio", type=float, default=0.0)
    ap.add_argument("--dataset-dir")
    ap.add_argument("--task")
    ap.add_argument("--model-name")
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--sleep-after", action="append", default=[],
                    help="测试钩子：PHASE:SECONDS，相位上报后挂起")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    sleep_map = {}
    for spec in args.sleep_after:
        name, _, secs = spec.partition(":")
        sleep_map[name] = float(secs or 1)

    if args.selftest:
        _emit(event="phase", name="start", detail="selftest")
        _emit(event="done", result={"status": "applied", "detail": "selftest"})
        return 0
    if args.mode not in ("pre", "post"):
        _emit(event="error", message="--mode 必须为 pre/post（或用 --selftest）")
        return 2
    if not args.dst or args.mode == "post" and not args.src:
        _emit(event="error", message="缺少 --dst（post 模式还需 --src）")
        return 2

    progress = _progress_factory(sleep_map)
    try:
        if args.mode == "post":
            from xccv.engine.models.pruning import prune_checkpoint

            _emit(event="phase", name="start",
                  detail=f"post src={args.src} ratio={args.ratio:g}")
            result = prune_checkpoint(args.src, args.dst, args.ratio,
                                      progress=progress)
        else:
            from xccv.engine.models.pruning import pretrain_prune
            from xccv.training.dataset_stats import collect_stats

            _emit(event="phase", name="start",
                  detail=f"pre model={args.model_name} ratio={args.ratio:g}")
            progress("scan", detail="扫描数据集类别…")
            stats = collect_stats(args.task, args.dataset_dir)
            if not stats.classes:
                _emit(event="done", result={
                    "status": f"数据集中未发现类别（{args.dataset_dir}）"})
                return 0
            result = pretrain_prune(
                args.model_name, args.task, len(stats.classes), args.ratio,
                classes=stats.classes, image_width=args.width,
                image_height=args.height, device="cpu", progress=progress)
            if result.get("status") == "applied":
                import os

                import torch

                tmp = str(args.dst) + ".part"
                with open(tmp, "wb") as f:
                    torch.save(result["ckpt"], f)
                os.replace(tmp, args.dst)
                result["dst"] = str(args.dst)
        # ckpt 含整个网络对象（不可 JSON 序列化也不该走 stdout）——
        # 产物已落盘，emit 前剥离，避免 done 行崩溃
        result.pop("ckpt", None)
        _emit(event="done", result=result)
        return 0
    except Exception as e:  # noqa: BLE001  任何未捕获异常=error 事件+退出码 1
        _emit(event="error", message=f"{type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
