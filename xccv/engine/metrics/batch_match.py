"""批量推理的 GT 匹配统计（纯逻辑，不依赖 Qt）。

批量目录里与图片同名的 labelme .json 视为真值标注：
- 有 GT 的图：检出按"同类 + IoU≥阈值"贪心匹配——命中=绿（TP）、未命中=红（FP）；
- GT 为空（OK/良品图）：全部检出记为 OK 误检（红·OK）；
- 无同名 json 的图：不参与匹配统计（无法判定）。

产出 match_stats（随 摘要.json 落盘，additive 键）：
    gt_total        GT 实例总数（参与匹配的图）
    n_gt_images     有 GT 的图片数
    n_ok_images     GT 为空的 OK 图数
    n_unmatched     未被任何检出命中的 GT 实例数（漏检候选）
    green           TP 检出分数列表（升序）
    red_defect      有 GT 图上的 FP 分数列表
    red_ok          OK 图上的 FP 分数列表（OK 误检）
    per_image       [{path, gt, missed, fp, fp_ok, matched_all}]（复判建议用）

阈值 t 下的指标（threshold_metrics）：precision=TP/(TP+FP)、
recall=TP/gt_total、ok_fp_per_img=OK 误检框/OK 图数。

另收集 dims与 dims_missed（漏检 GT 框的
面积/短边/长边）——阈值面板"漏检"曲线的数据源。

V0.7.3 §二：task="anomaly"/"classification" 走图像级口径
（来源目录=弱 GT；与训练平台统一）——见 collect_match_stats 与两个
_collect_* 分支；per_image/dims 结构三口径同构（只增不改）。
"""

from __future__ import annotations
import logging
log = logging.getLogger(__name__)

import json
from pathlib import Path

import numpy as np

from ..dataset.ignore import OVERLAP_DROP_RATIO, _inter_ratio
from ..dataset.reader import is_negative_dir_name
from .detection import iou_matrix

# 阈值面板多维度（score=conf 单独走既有通道）
_DIM_KEYS = ("area", "short", "long", "fg_ratio", "gray_diff")


def _pts_bbox(points) -> tuple:
    """点集外接框 xyxy。"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _json_regions(shapes_raw) -> list:
    """labelme 原始 shapes → `__ignore__` 区域外接框列表（多边形按外接框近似）。"""
    out = []
    for sh in shapes_raw or []:
        if str(sh.get("label", "")).strip() != "__ignore__":
            continue
        pts = sh.get("points") or []
        if pts:
            out.append(_pts_bbox(pts))
    return out


def read_gt_full(json_path: Path) -> tuple[list, list, list]:
    """labelme json → (GT 框, GT 标签, `__ignore__` 区域外接框)。

    GT 已按 O8 近似语义过滤：ignore 标记不是 GT、与区域重叠率>50% 的 GT 剔除。
    """
    try:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except Exception:
        log.warning("GT 标注 JSON 读取失败（按空统计）", exc_info=True)
        return [], [], []
    raw = data.get("shapes") or []
    regions = _json_regions(raw)
    boxes: list = []
    labels: list = []
    for sh in raw:
        pts = sh.get("points") or []
        st = sh.get("shape_type", "polygon")
        if not pts:
            continue
        if str(sh.get("label", "")).strip() == "__ignore__":
            continue  # 忽略区域标记不是真值实例
        if st not in ("rectangle", "polygon"):
            continue  # point/line/circle 等：本期不参与框匹配
        bb = _pts_bbox(pts)
        x1, y1, x2, y2 = bb
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        boxes.append([x1, y1, x2, y2])
        labels.append(str(sh.get("label", "")))
    if regions:
        keep = [i for i, b in enumerate(boxes)
                if not any(_inter_ratio(tuple(b), r) > OVERLAP_DROP_RATIO
                           for r in regions)]
        boxes = [boxes[i] for i in keep]
        labels = [labels[i] for i in keep]
    return boxes, labels, regions


def read_gt_shapes(json_path: Path) -> tuple[list, list]:
    """labelme json → ([xyxy 框], [标签])（read_gt_full 的 GT 部分）。"""
    boxes, labels, _ = read_gt_full(json_path)
    return boxes, labels


def _match_full(det_boxes, det_labels, det_scores,
                gt_boxes, gt_labels, iou_thr: float) -> tuple:
    """贪心匹配内部实现 → (检出侧命中列表, GT 侧被命中列表)。

    同类 + IoU≥iou_thr，按分数降序贪心，GT 一对一；与
    与 detection_confusion_matrix 相同的匹配语义。
    """
    n_det, n_gt = len(det_boxes), len(gt_boxes)
    matched_gt = [False] * n_gt
    out = [False] * n_det
    if n_det == 0 or n_gt == 0:
        return out, matched_gt
    boxes_det = np.asarray(det_boxes, dtype=float).reshape(-1, 4)
    boxes_gt = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    ious = iou_matrix(boxes_det, boxes_gt)
    order = np.argsort(-np.asarray(det_scores, dtype=float))
    for pi in order:
        best, best_iou = -1, iou_thr
        for gi in range(n_gt):
            if matched_gt[gi] or gt_labels[gi] != det_labels[pi]:
                continue
            if ious[pi, gi] >= best_iou:
                best, best_iou = gi, float(ious[pi, gi])
        if best >= 0:
            matched_gt[best] = True
            out[int(pi)] = True
    return out, matched_gt


def match_detections(det_boxes, det_labels, det_scores,
                     gt_boxes, gt_labels, iou_thr: float = 0.5) -> list[bool]:
    """逐检出是否命中 GT（同类 IoU≥iou_thr，按分数降序贪心，GT 一对一）。

    返回与 det_boxes 等长的 matched 列表（True=TP，False=FP）。
    """
    return _match_full(det_boxes, det_labels, det_scores,
                       gt_boxes, gt_labels, iou_thr)[0]


def _det_dim_values(det: dict) -> dict:
    """单检出的多维度值：面积/短长边由 bbox 现算，
    fg_ratio/gray_diff 取批量线程预计算的 det["_dims"]（缺则 None）。"""
    box = det.get("bbox")
    if not box:
        return {}
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
    values = {"area": w * h, "short": min(w, h), "long": max(w, h)}
    pre = det.get("_dims") or {}
    values["fg_ratio"] = pre.get("fg_ratio")
    values["gray_diff"] = pre.get("gray_diff")
    return values


def _det_record(det: dict, matched: bool) -> list:
    """逐检出紧凑记录（复盘页阈值复用的数据源，列序见 review_logic）：
    [score, matched, area, short, long, fg_ratio, gray_diff]。"""
    v = _det_dim_values(det)
    return [float(det.get("score", 0.0)), bool(matched),
            v.get("area"), v.get("short"), v.get("long"),
            v.get("fg_ratio"), v.get("gray_diff")]


def _new_match_stats(score_semantics: dict | None = None, task: str = "detection") -> dict:
    """match_stats 骨架（三口径共用；score_semantics/task 为 §二/§三 additive 键）。"""
    return {"gt_total": 0, "n_gt_images": 0, "n_ok_images": 0,
            "n_unmatched": 0, "green": [], "red_defect": [], "red_ok": [],
            "per_image": [],
            "dims": {k: {"green": [], "red_defect": [], "red_ok": []}
                     for k in _DIM_KEYS},
            # 漏检 GT 框自身的几何维度（面积/短边/长边）——
            # 阈值面板"漏检"曲线的数据源；分数/灰度类维度 GT 侧无值不收集
            "dims_missed": {k: [] for k in ("area", "short", "long")},
            "task": task,
            "score_semantics": dict(score_semantics or {})}


def _weak_dir_names(path: Path, weak_classes) -> tuple[bool, bool]:
    """图片路径父目录 → (是 OK 图, 是 NG 图)：来源目录当弱 GT（§二口径）。

    OK = 任一父目录命中负样本名（NEGATIVE_DIR_NAMES，与训练平台同源）；
    NG = 任一父目录命中类别目录名（weak_classes 优先；缺省退"直接父目录
    名非负样本名即 NG"——OK/NG 子目录布局下直接父目录即类别目录）。
    """
    names = [p.name for p in path.parents]
    if any(is_negative_dir_name(n) for n in names):
        return True, False
    if weak_classes:
        return False, any(n in weak_classes for n in names)
    return False, not is_negative_dir_name(path.parent.name)


def _collect_anomaly_stats(results: list, score_semantics, weak_classes) -> dict:
    """§二：ANOMALY 图像级复盘口径（与训练平台 OK 留出 + NG 验证统一）。

    OK 图有图像级检出=过检、NG 图零检出=漏检、NG 图有检出=正确（阈值
    复用/分组逻辑零改动：一图一条检出记录 [图像分, matched(NG 恒 True),
    无几何维度]，review_logic 按阈值现算分组）。图像分 = 模型标定 z
    （Prediction.probs[0]，缺省退检出最大分）。
    """
    stats = _new_match_stats(score_semantics, task="anomaly")
    seen_gt = False
    for r in results:
        path = Path(str(r.get("path") or r.get("file") or ""))
        score = r.get("image_score")
        if score is None:
            dets = r.get("detections") or []
            score = max((float(d.get("score", 0.0)) for d in dets),
                        default=None)
        ok_dir, ng_dir = _weak_dir_names(path, weak_classes)
        # 检出记录 matched 语义：NG 图的图像级检出=命中(True)；OK 图/来源
        # 不明图的检出=误报(False)——否则精确率/召回率把 OK 误报计成 TP
        # （实测复盘抓到：precision 1.0 / recall 1.5 假象）
        rec = ([float(score), bool(ng_dir), None, None, None, None, None]
               if score is not None else [])
        if ok_dir:
            seen_gt = True
            stats["n_ok_images"] += 1
            if rec:
                stats["red_ok"].append(rec[0])
            stats["per_image"].append({
                "path": str(path), "gt": 0, "gt_available": True,
                "ok_dir": True, "missed": 0,
                "fp": 1 if rec else 0, "fp_ok": 1 if rec else 0,
                "matched_all": not rec,
                "dets": [rec] if rec else []})
        elif ng_dir:
            seen_gt = True
            stats["n_gt_images"] += 1
            stats["gt_total"] += 1
            if rec:
                stats["green"].append(rec[0])
            else:
                stats["n_unmatched"] += 1
            stats["per_image"].append({
                "path": str(path), "gt": 1, "gt_available": True,
                "ok_dir": False, "missed": 0 if rec else 1,
                "fp": 0, "fp_ok": 0, "matched_all": bool(rec),
                "dets": [rec] if rec else []})
        else:
            # 来源目录不明（非 OK/NG 布局）：未评估，仅展示图像分供肉眼确认
            stats["per_image"].append({
                "path": str(path), "gt": None, "gt_available": False,
                "ok_dir": False, "missed": None, "fp": None,
                "fp_ok": None, "matched_all": None,
                "dets": [rec] if rec else []})
    stats["gt_available"] = seen_gt
    return stats


def _collect_classification_stats(results: list, score_semantics,
                                  weak_classes) -> dict:
    """§二：分类图像级口径——预测类别 vs 数据集目录真值（子文件夹=GT）。

    对/错分组（错预测进"错误"组，复盘页按任务改组名）；分数 = softmax
    概率；一图一条检出记录 [prob, matched(预测=目录名), 无几何维度]。
    """
    stats = _new_match_stats(score_semantics, task="classification")
    seen_gt = False
    for r in results:
        path = Path(str(r.get("path") or r.get("file") or ""))
        cls = list(r.get("classifications") or [])
        if not cls:
            dets = r.get("detections") or []
            if dets:
                cls = [{"label": str(dets[0].get("label", "")),
                        "score": float(dets[0].get("score", 0.0))}]
        gt_label = path.parent.name
        judged = (gt_label in weak_classes) if weak_classes else (
            bool(gt_label) and not is_negative_dir_name(gt_label))
        if cls:
            pred_label = str(cls[0].get("label", ""))
            rec = [float(cls[0].get("score", 0.0)), pred_label == gt_label,
                   None, None, None, None, None]
        else:
            pred_label, rec = "", []
        if not judged:
            stats["per_image"].append({
                "path": str(path), "gt": None, "gt_available": False,
                "ok_dir": False, "missed": None, "fp": None,
                "fp_ok": None, "matched_all": None,
                "dets": [rec] if rec else []})
            continue
        seen_gt = True
        stats["n_gt_images"] += 1
        stats["gt_total"] += 1
        if rec and rec[1]:
            stats["green"].append(rec[0])
        elif rec:
            stats["red_defect"].append(rec[0])
        else:
            stats["n_unmatched"] += 1
        wrong = bool(rec) and not rec[1]
        stats["per_image"].append({
            "path": str(path), "gt": 1, "gt_available": True,
            "ok_dir": False, "missed": 1 if wrong else 0,
            "fp": 1 if wrong else 0, "fp_ok": 0,
            "matched_all": bool(rec) and rec[1],
            "dets": [rec] if rec else []})
    stats["gt_available"] = seen_gt
    return stats


def collect_match_stats(results: list, iou_thr: float = 0.5, task=None,
                        score_semantics: dict | None = None,
                        weak_classes=None) -> dict:
    """批量结果（含 path/detections）→ match_stats dict（按任务路由口径）。

    results 每项需含 "path" 与 "detections"（label/score/bbox，批量线程
    既有产出）。无任何同名 GT json 时返回 {"gt_available": False}。

    V0.7.3 §二（复盘与训练平台统一口径）：
    - task="anomaly"：图像级判定（来源目录=弱 GT）→ _collect_anomaly_stats；
    - task="classification"：预测类别 vs 目录真值 → _collect_classification_stats；
    - 其余（检测/分割）：labelme json GT 框匹配（既有口径）。

    O2 扩展（V0.7 追加）：同时收集多维度值表 stats["dims"]
    （{dim: {green, red_defect, red_ok}}）——阈值面板各维度直方图数据源；
    fg_ratio/gray_diff 取批量线程预计算的 det["_dims"]（无图维度为空表）。
    """
    task_v = str(getattr(task, "value", task) or "")
    if task_v == "anomaly":
        return _collect_anomaly_stats(results, score_semantics, weak_classes)
    if task_v == "classification":
        return _collect_classification_stats(results, score_semantics,
                                             weak_classes)
    stats = _new_match_stats(score_semantics)

    def _push(values: dict, bucket: str):
        for k in _DIM_KEYS:
            v = values.get(k)
            if v is not None:
                stats["dims"][k][bucket].append(float(v))

    seen_gt = False
    for r in results:
        path = Path(str(r.get("path") or r.get("file") or ""))
        dets = r.get("detections") or []
        json_path = path.with_suffix(".json")
        gt_boxes, gt_labels, regions = read_gt_full(json_path)
        det_boxes = [d.get("bbox") for d in dets]
        det_labels = [str(d.get("label", "")) for d in dets]
        det_scores = [float(d.get("score", 0.0)) for d in dets]
        # 落在 ignore 区域内的检出不计 FP（模糊区不算误检，
        # 与训练侧同口径）；GT 侧的过滤已在 read_gt_full 内完成
        kept_dets = list(dets)
        if regions and det_boxes:
            keep = [i for i, b in enumerate(det_boxes)
                    if not any(_inter_ratio(tuple(b), rg) > OVERLAP_DROP_RATIO
                               for rg in regions)]
            det_boxes = [det_boxes[i] for i in keep]
            det_labels = [det_labels[i] for i in keep]
            det_scores = [det_scores[i] for i in keep]
            kept_dets = [dets[i] for i in keep]
        if not gt_boxes and not gt_labels:
            # OK 图判定与训练平台同源——
            # NEGATIVE_DIR_NAMES 目录（OK/良品/正常/负样本/背景…）内**无同名
            # json** 的图 = OK 负样本，全部检出记 OK 误检；其余无 json 图 =
            # 未评估（生产图），空 json 图 = 已审无缺陷样本（按 GT 空评估）
            ok_dir = any(is_negative_dir_name(p.name) for p in path.parents)
            if json_path.exists():
                seen_gt = True
                stats["n_ok_images"] += 1
                stats["red_ok"].extend(det_scores)
                for d in kept_dets:
                    _push(_det_dim_values(d), "red_ok")
                stats["per_image"].append({
                    "path": str(path), "gt": 0, "gt_available": True,
                    "ok_dir": ok_dir, "missed": 0,
                    "fp": len(det_scores), "fp_ok": len(det_scores),
                    "matched_all": not det_scores,
                    "dets": [_det_record(d, False) for d in kept_dets]})
            elif ok_dir:
                seen_gt = True
                stats["n_ok_images"] += 1
                stats["red_ok"].extend(det_scores)
                for d in kept_dets:
                    _push(_det_dim_values(d), "red_ok")
                stats["per_image"].append({
                    "path": str(path), "gt": 0, "gt_available": True,
                    "ok_dir": True, "missed": 0,
                    "fp": len(det_scores), "fp_ok": len(det_scores),
                    "matched_all": not det_scores,
                    "dets": [_det_record(d, False) for d in kept_dets]})
            else:
                stats["per_image"].append({
                    "path": str(path), "gt": None, "gt_available": False,
                    "ok_dir": False, "missed": None, "fp": None,
                    "fp_ok": None, "matched_all": None,
                    # 未评估图的检出照记——OK/NG 扫描曲线与分组需要
                    "dets": [_det_record(d, False) for d in kept_dets]})
            continue
        seen_gt = True
        matched, gt_hit = _match_full(det_boxes, det_labels, det_scores,
                                      gt_boxes, gt_labels, iou_thr)
        n_fp = sum(1 for m in matched if not m)
        n_missed = len(gt_boxes) - sum(1 for m in matched if m)
        stats["n_gt_images"] += 1
        stats["gt_total"] += len(gt_boxes)
        stats["n_unmatched"] += n_missed
        stats["green"].extend(s for s, m in zip(det_scores, matched) if m)
        stats["red_defect"].extend(s for s, m in zip(det_scores, matched) if not m)
        for d, m in zip(kept_dets, matched):
            _push(_det_dim_values(d), "green" if m else "red_defect")
        # 漏检 GT 框自身的几何维度（bbox 现算，与检出同口径）
        for gb, hit in zip(gt_boxes, gt_hit):
            if hit:
                continue
            x1, y1, x2, y2 = (float(v) for v in gb)
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            stats["dims_missed"]["area"].append(w * h)
            stats["dims_missed"]["short"].append(min(w, h))
            stats["dims_missed"]["long"].append(max(w, h))
        stats["per_image"].append({
            "path": str(path), "gt": len(gt_boxes), "gt_available": True,
            "ok_dir": any(is_negative_dir_name(p.name) for p in path.parents),
            "missed": n_missed,
            "fp": n_fp, "fp_ok": 0, "matched_all": n_fp == 0 and n_missed == 0,
            "dets": [_det_record(d, m) for d, m in zip(kept_dets, matched)]})
    # 即使批量目录完全没有 GT/OK json（纯生产图），
    # 也返回完整 per_image/dims——复盘页的「未评估」组要展示每张生产图的
    # 检出供肉眼确认；gt_available 仅表示"是否出现过可评估的 GT"
    stats["gt_available"] = seen_gt
    return stats


def threshold_metrics_for(stats: dict, dim: str, thr: float) -> dict:
    """多维度指标现算：与 threshold_metrics 同口径，但取
    stats["dims"][dim] 的红绿表按维度阈值过滤（score 维走 threshold_metrics）。"""
    if dim == "score":
        return threshold_metrics(stats, thr)
    d = (stats.get("dims") or {}).get(dim) or {}
    green = [v for v in d.get("green", []) if v >= thr]
    fp_def = [v for v in d.get("red_defect", []) if v >= thr]
    fp_ok = [v for v in d.get("red_ok", []) if v >= thr]
    tp = len(green)
    fp_all = len(fp_def) + len(fp_ok)
    gt_total = int(stats.get("gt_total") or 0)
    n_ok = int(stats.get("n_ok_images") or 0)
    return {"tp": tp, "fp": fp_all,
            "precision": (round(tp / (tp + fp_all), 4) if tp + fp_all else None),
            "recall": (round(tp / gt_total, 4) if gt_total else None),
            "ok_fp_per_img": (round(len(fp_ok) / n_ok, 3) if n_ok else None),
            "ok_fp_images": None}


def threshold_metrics(stats: dict, conf: float) -> dict:
    """在阈值 conf 下从 match_stats 现算指标（拖线实时刷新用，零重推理）。"""
    green = [s for s in stats.get("green", []) if s >= conf]
    fp_def = [s for s in stats.get("red_defect", []) if s >= conf]
    fp_ok = [s for s in stats.get("red_ok", []) if s >= conf]
    gt_total = int(stats.get("gt_total") or 0)
    n_ok = int(stats.get("n_ok_images") or 0)
    tp = len(green)
    fp_all = len(fp_def) + len(fp_ok)
    out = {"tp": tp, "fp": fp_all,
           "precision": (round(tp / (tp + fp_all), 4) if tp + fp_all else None),
           "recall": (round(tp / gt_total, 4) if gt_total else None),
           "ok_fp_per_img": (round(len(fp_ok) / n_ok, 3) if n_ok else None),
           "ok_fp_images": None}
    if n_ok:
        # OK 误检率：按 fp_ok 归属图无法从分数列表还原——汇总口径用
        # per_image 在该阈值的现算（调用方需要时另取），此处给框/张口径
        out["ok_fp_images"] = None
    return out


# ---------------------------------------------------------------------------
# V0.7.2（重设计）：阈值扫描曲线数据层。
# 曲线含义改为"阈值=x 处 正常/过检/漏检 的总数"（联合过滤口径，与复盘
# 分组同源）：正常=阈值下命中 GT 的检出数；过检=未命中检出数（含 OK 图
# 误检）；漏检=gt_total−正常（贪心一对一匹配下被阈值筛掉的命中 + 未命中
# GT）。未评估范围（无 GT 生产图）另给 OK/NG 图数曲线：NG=有检出通过
# 阈值的图数。全部从 per_image 逐检出记录现算，零重推理。
DET_COL = {"score": 0, "matched": 1, "area": 2, "short": 3, "long": 4,
           "fg_ratio": 5, "gray_diff": 6}


def _det_record_value(rec: list, dim: str):
    col = DET_COL.get(dim)
    if col is None:
        return None
    return rec[col]


def _passes_fixed(rec: list, fixed: dict, dim: str) -> bool:
    """除当前维外，检出是否通过全部固定维度阈值（联合过滤口径）。"""
    for d, thr in (fixed or {}).items():
        if d == dim:
            continue
        v = _det_record_value(rec, d)
        if v is None or float(v) < float(thr):
            return False
    return True


def _is_unjudged(img: dict) -> bool:
    return img.get("gt") is None and not img.get("gt_available")


def _iter_eval_dets(stats: dict, dim: str, fixed: dict | None,
                    unjudged: bool = False):
    """遍历（可选范围）检出 → (当前维值, 是否命中)。

    unjudged=False 取已评估范围（有 GT 图 + OK 图）；True 仅取未评估生产图。
    当前维值缺失（None）或未通过 fixed 维度阈值的检出不返回。
    """
    for img in (stats or {}).get("per_image") or []:
        if _is_unjudged(img) != unjudged:
            continue
        for rec in img.get("dets") or []:
            v = _det_record_value(rec, dim)
            if v is None or not _passes_fixed(rec, fixed, dim):
                continue
            yield float(v), bool(rec[DET_COL["matched"]]), img, rec


def sweep_counts(stats: dict, dim: str, xs, fixed: dict | None = None) -> dict:
    """已评估范围阈值扫描 → {"xs", "normal", "over", "missed"}（与 xs 等长）。

    normal(x)=命中检出且当前维≥x 的数目；over(x)=未命中检出≥x（含 OK 图
    误检）；missed(x)=gt_total−normal(x)（一对一匹配下未命中的 GT 实例）。
    """
    col = DET_COL.get(dim)
    pairs = list(_iter_eval_dets(stats, dim, fixed)) if col is not None else []
    tp_vals = sorted(v for v, m, _i, _r in pairs if m)
    fp_vals = sorted(v for v, m, _i, _r in pairs if not m)
    gt_total = int((stats or {}).get("gt_total") or 0)
    normal, over, missed = [], [], []
    for x in xs:
        x = float(x)
        tp = len(tp_vals) - int(np.searchsorted(tp_vals, x, side="left"))
        fp = len(fp_vals) - int(np.searchsorted(fp_vals, x, side="left"))
        normal.append(tp)
        over.append(fp)
        missed.append(max(0, gt_total - tp))
    return {"xs": list(xs), "normal": normal, "over": over, "missed": missed}


def sweep_unjudged(stats: dict, dim: str, xs,
                   fixed: dict | None = None) -> dict:
    """未评估范围阈值扫描 → {"xs", "ok", "ng"}（图数口径，与 xs 等长）。

    ng(x)=有至少一个检出（当前维≥x 且通过 fixed）的生产图数；
    ok(x)=未评估图总数−ng(x)。
    """
    col = DET_COL.get(dim)
    maxima: list = []
    total = 0
    for img in (stats or {}).get("per_image") or []:
        if not _is_unjudged(img):
            continue
        total += 1
        best = None
        for rec in img.get("dets") or []:
            v = _det_record_value(rec, dim) if col is not None else None
            if v is None or not _passes_fixed(rec, fixed, dim):
                continue
            v = float(v)
            if best is None or v > best:
                best = v
        if best is not None:
            maxima.append(best)
    maxima = sorted(maxima)
    ng, ok = [], []
    for x in xs:
        g = len(maxima) - int(np.searchsorted(maxima, float(x), side="left"))
        ng.append(g)
        ok.append(max(0, total - g))
    return {"xs": list(xs), "ok": ok, "ng": ng,
            "unjudged_total": total}


def sweep_metrics(stats: dict, dim: str, thr: float,
                  fixed: dict | None = None) -> dict:
    """当前维阈值 thr 处指标（联合过滤口径；面板数字行用）。

    返回 tp/fp/fp_ok/precision/recall/ok_fp_per_img（已评估口径）与
    ng_img/ok_img/unjudged_total（未评估口径）。
    """
    tp = fp = fp_ok = 0
    for v, m, img, _rec in _iter_eval_dets(stats, dim, fixed):
        if float(v) < float(thr):
            continue
        if m:
            tp += 1
        else:
            fp += 1
            if img.get("gt") == 0 and img.get("gt_available"):
                fp_ok += 1
    gt_total = int((stats or {}).get("gt_total") or 0)
    n_ok = int((stats or {}).get("n_ok_images") or 0)
    un = sweep_unjudged(stats, dim, [float(thr)], fixed)
    return {"tp": tp, "fp": fp, "fp_ok": fp_ok,
            "precision": (round(tp / (tp + fp), 4) if tp + fp else None),
            "recall": (round(tp / gt_total, 4) if gt_total else None),
            "ok_fp_per_img": (round(fp_ok / n_ok, 3) if n_ok else None),
            "ng_img": un["ng"][0], "ok_img": un["ok"][0],
            "unjudged_total": un["unjudged_total"]}


