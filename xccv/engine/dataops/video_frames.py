# -*- coding: utf-8 -*-
"""视频抽帧导出（OpenCV）。

- 抽帧方式三选一：interval（间隔帧数）/ target_fps（目标帧率）/ 时长上限；
  同一次执行 interval 与 target_fps 二选一，时长上限可叠加；
- 输出命名 <视频名>_f<序号>.png（序号为源帧号，零对齐便于排序）；
- 可选感知哈希去重（dHash 8×8，汉明距离 ≤ 阈值视为重复帧，跳过导出）。
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FramePlan:
    """一次抽帧计划（预览=估算帧数；执行=实际抽取）。"""

    video: Path
    dst: Path
    interval: int = 30          # 间隔帧数（与 target_fps 二选一，0 表示不用）
    target_fps: float = 0.0     # 目标帧率（>0 时按源帧率换算步长）
    max_duration_s: float = 0.0  # 时长上限（秒；0 = 不限）
    max_frames: int = 0         # 导出帧数上限（0 = 不限）
    dedup: bool = False
    dedup_threshold: int = 4    # 汉明距离阈值
    progress: object | None = None  # done:int, total:int, path:str 回调（执行期用）


@dataclass
class FrameResult:
    """抽帧执行结果。"""

    frames: int = 0
    skipped_dup: int = 0
    total_scanned: int = 0
    outputs: list = field(default_factory=list)


def _dhash(img) -> int:
    """dHash 8×8：缩放到 9×8 灰度，相邻像素亮度比较生成 64 位哈希。"""
    import cv2

    small = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(small, (9, 8), interpolation=cv2.INTER_AREA)
    h = 0
    for row in small:
        for i in range(8):
            h = (h << 1) | (1 if row[i] > row[i + 1] else 0)
    return h


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def plan_estimate(plan: FramePlan) -> dict:
    """打开视频估算导出帧数（预览用，不产出文件）。"""
    import cv2

    cap = cv2.VideoCapture(str(plan.video))
    if not cap.isOpened():
        return {"ok": False, "error": "无法打开视频"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if plan.target_fps > 0:
        step = max(1, int(round(fps / plan.target_fps)))
    else:
        step = max(1, int(plan.interval))
    limit = total
    if plan.max_duration_s > 0:
        limit = min(limit, int(plan.max_duration_s * fps))
    est = max(0, (min(limit, total) + step - 1) // step) if total else 0
    if plan.max_frames > 0:
        est = min(est, plan.max_frames)
    return {"ok": True, "fps": round(fps, 3), "total_frames": total,
            "step": step, "estimate": est}


def export_frames(plan: FramePlan) -> FrameResult:
    """执行抽帧导出（dst 不存在时创建）。"""
    import cv2

    result = FrameResult()
    plan.dst = Path(plan.dst)
    plan.dst.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(plan.video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {plan.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if plan.target_fps > 0:
        step = max(1, int(round(fps / plan.target_fps)))
    else:
        step = max(1, int(plan.interval))
    frame_limit = total
    if plan.max_duration_s > 0:
        frame_limit = min(frame_limit, int(plan.max_duration_s * fps))

    stem = plan.video.stem
    width = max(4, len(str(max(total, 1))))
    last_hash = None
    idx = -1
    emitted = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx >= frame_limit > 0:
            break
        if idx % step != 0:
            continue
        result.total_scanned += 1
        if plan.dedup:
            h = _dhash(frame)
            if last_hash is not None and _hamming(h, last_hash) <= plan.dedup_threshold:
                result.skipped_dup += 1
                continue
            last_hash = h
        name = f"{stem}_f{idx:0{width}d}.png"
        okw, buf = cv2.imencode(".png", frame)
        if okw:
            buf.tofile(str(plan.dst / name))
            result.frames += 1
            result.outputs.append(plan.dst / name)
            emitted += 1
        if plan.progress:
            plan.progress(idx + 1, max(total, 1), name)
        if plan.max_frames > 0 and emitted >= plan.max_frames:
            break
    cap.release()
    return result
