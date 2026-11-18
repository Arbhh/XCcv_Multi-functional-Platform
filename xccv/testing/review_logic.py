"""项目复盘分组逻辑（纯逻辑，不依赖 Qt；用户勘误 2026-11-13）。

复盘页在批量推理完成后，把每张图按当前维度阈值划入三组（VIMO「结果
分析」的过检/漏检口径）：

- 过检：该图存在不命中 GT 的检出（FP），或 OK 图上有任何检出；
- 漏检：无过检且有 GT 实例未被命中；
- 正确：其余（有标图全部命中且无 FP；OK 图零检出）；
- 无同名 GT json 的图不参与分组（未评估，只在统计行计数）。

阈值复用：批量只跑一次；调阈值时从 match_stats.per_image 的逐检出记录
（[score, matched, area, short, long, fg_ratio, gray_diff]）现算分组，
零重推理。检出需通过**全部启用维度**的阈值才算有效检出。
"""

from __future__ import annotations

from pathlib import Path

# per_image.dets 记录列序（与 batch_match.collect_match_stats 写入一致）
DET_SCORE, DET_MATCHED, DET_AREA, DET_SHORT, DET_LONG, DET_FG, DET_GRAY = range(7)

_DIM_INDEX = {"score": DET_SCORE, "area": DET_AREA, "short": DET_SHORT,
              "long": DET_LONG, "fg_ratio": DET_FG, "gray_diff": DET_GRAY}


def det_passes(rec: list, thresholds: dict) -> bool:
    """单检出是否通过全部启用维度的阈值（score 维固定参与）。"""
    for dim, thr in (thresholds or {}).items():
        col = _DIM_INDEX.get(dim)
        if col is None:
            continue
        v = rec[col]
        if v is None or float(v) < float(thr):
            return False
    return True


def _entry_from_per_image(img: dict, thresholds: dict,
                          mode: str = "detection") -> dict:
    """按阈值现算单图统计 → 分组条目。

    mode="classification"（§二）：徽章「过」改「错」（错预测 = 目录真值
    不符；分组键不变，复盘页把过检组改题"错误"）。
    """
    dets = img.get("dets") or []
    passing = [d for d in dets if det_passes(d, thresholds)]
    n_hit = sum(1 for d in passing if d[DET_MATCHED])
    n_det = len(passing)
    gt = int(img.get("gt") or 0)
    is_ok_image = img.get("gt_available") and gt == 0
    missed = max(0, gt - n_hit)
    fp = n_det - n_hit
    fp_ok = n_det if is_ok_image else 0
    entry = {
        "file": img.get("file") or Path(str(img.get("path") or "")).name,
        "path": img.get("path") or "",
        "gt": gt, "missed": missed, "fp": fp, "fp_ok": fp_ok,
        "is_ok": is_ok_image,
    }
    parts = []
    from ..i18n import tr

    if missed:
        parts.append(f"{tr('漏')}{missed}")
    if fp:
        parts.append(f"{tr('错')}{fp}" if mode == "classification"
                     else f"{tr('过')}{fp}")
    if is_ok_image and n_det:
        parts.append(f"{tr('OK误检')}{n_det}")
    entry["badges"] = " ".join(parts)
    return entry


def classify_images(match_stats: dict, thresholds: dict) -> dict:
    """按维度阈值把全部图分成 正确/漏检/过检/未评估 四组（过检优先）。

    OK 图口径与训练平台同源（再勘误③）：NEGATIVE_DIR_NAMES 目录内无 json
    的图（match_stats 已计为 OK）与"已审无缺陷"空 json 图都按 GT 空评估，
    有检出即过检。无 json 的生产图 = 未评估组（再勘误④：必须可见，
    否则加载新生产图完全无法确认）。
    §二：match_stats["task"]="anomaly"/"classification" 为 batch_match
    图像级口径（弱 GT），分组逻辑零改动直接复用（徽章随任务改字）。
    thresholds 形如 {"score": 0.25, "area": 500, ...}（enabled_dims；
    空字典=不过滤，等价于仅 score=0）。返回：
    {"correct": [entry], "missed": [entry], "over": [entry],
     "unjudged_list": [entry], "unjudged": n, "total": n}
    """
    mode = str((match_stats or {}).get("task") or "detection")
    groups = {"correct": [], "missed": [], "over": [], "unjudged_list": []}
    unjudged = 0
    for img in (match_stats or {}).get("per_image") or []:
        if img.get("gt") is None and not img.get("gt_available"):
            # 无 json 生产图：仍生成条目进"未评估"组（检出数照算，供肉眼确认）
            entry = _entry_from_per_image(
                {**img, "gt": 0, "gt_available": False}, thresholds, mode)
            entry["unjudged"] = True
            groups["unjudged_list"].append(entry)
            unjudged += 1
            continue
        entry = _entry_from_per_image(img, thresholds, mode)
        if entry["fp"] > 0 or entry["fp_ok"] > 0:
            groups["over"].append(entry)
        elif entry["missed"] > 0:
            groups["missed"].append(entry)
        else:
            groups["correct"].append(entry)
    groups["unjudged"] = unjudged
    # V0.7.2 任务③：未评估按检出过阈与否分 OK/NG（与切换后曲线同口径）
    groups["unjudged_ok"] = sum(
        1 for e in groups["unjudged_list"] if not e.get("fp"))
    groups["unjudged_ng"] = sum(
        1 for e in groups["unjudged_list"] if e.get("fp"))
    groups["total"] = (len(groups["correct"]) + len(groups["missed"])
                       + len(groups["over"]) + unjudged)
    return groups


def sweep_group_counts(match_stats: dict, dim: str, xs,
                       fixed: dict | None = None) -> dict:
    """阈值扫描（**图口径**，V0.7.2 真机勘误）：曲线读数 = 该阈值下三组的
    **图数**，与右侧分组/统计行完全同口径（过检 > 漏检 > 正确 优先级）。

    已评估范围（有 GT 图 + OK 图）；未评估生产图不进三类曲线（OK/NG
    图数曲线走 sweep_unjudged，本就是图口径）。§二：分类任务"过"组 =
    错预测（组名由复盘页改题）。
    返回 {"xs", "correct", "missed", "over"}（与 xs 等长）。"""
    mode = str((match_stats or {}).get("task") or "detection")
    imgs = []
    for img in (match_stats or {}).get("per_image") or []:
        if img.get("gt") is None and not img.get("gt_available"):
            continue
        imgs.append(img)
    correct, missed, over = [], [], []
    for x in xs:
        thr = {**(fixed or {}), dim: float(x)}
        c = m = o = 0
        for img in imgs:
            e = _entry_from_per_image(img, thr, mode)
            if e["fp"] > 0 or e["fp_ok"] > 0:
                o += 1
            elif e["missed"] > 0:
                m += 1
            else:
                c += 1
        correct.append(c)
        missed.append(m)
        over.append(o)
    return {"xs": list(xs), "correct": correct, "missed": missed,
            "over": over}


def stats_line(groups: dict, scope: str = "evaluated",
               mode: str = "detection") -> str:
    """统计行人话（复盘页底部）。scope=unjudged 给 OK/NG 行。

    §二：mode="classification" 把「过检」改「错误」（对/错分组语义）。"""
    from ..i18n import tr

    if scope == "unjudged":
        return (f"OK {groups.get('unjudged_ok', 0)} · "
                f"NG {groups.get('unjudged_ng', 0)}"
                f"（{tr('无 GT')}）")
    n_c, n_m, n_o = (len(groups["correct"]), len(groups["missed"]),
                     len(groups["over"]))
    un = groups.get("unjudged", 0)
    over_label = tr("错误") if mode == "classification" else tr("过检")
    text = (f"{tr('正确')} {n_c} · {tr('漏检')} {n_m} · "
            f"{over_label} {n_o}")
    if un:
        text += f" · {tr('未评估')} {un}（{tr('无 GT')}）"
    return text
