"""平台-部署推理一致性自检（一次性工具，不驻留）。

选一张图，本地 predict 与经 HTTP 服务的 predict 各跑一次，逐框 IoU/
逐类 diff（容差内视为一致），输出"一致/不一致 + 差异明细"。架构上
平台/服务/流程全走同一 model.predict（同源），本工具把该优势显性化——
现场"两边不一样"从排查题变成一键题。

对比口径：检出框贪心匹配 IoU≥0.98、同 label、分数差 <0.02；分类
top1 一致；无检出 vs 无检出视为一致。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import requests
from PyQt5.QtCore import QThread, pyqtSignal

from .engine.io_utils import imread_unicode
from .engine.metrics.detection import iou_matrix
from .engine.models.base import Prediction

log = logging.getLogger(__name__)

IOU_TOL = 0.98
SCORE_TOL = 0.02


def _local_predict(model_path: Path, image_path: Path,
                   conf: float, iou: float) -> dict:
    """本地同权重推理（registry 唯一加载入口；.engine 需 GPU 环境）。"""
    from .engine.models import load_model
    from .engine.models.base import TaskType

    model = load_model(model_path, device="auto")
    img = imread_unicode(str(image_path))
    if img is None:
        raise ValueError(f"图片无法读取: {image_path}")
    pred: Prediction = model.predict(img, conf=conf, iou=iou)
    task = getattr(model, "task", TaskType.DETECTION)
    from .engine.result_json import build_result_json

    return build_result_json(
        task=task, source={"file": Path(image_path).name,
                           "path": str(image_path)},
        model={"file": Path(model_path).name},
        inference={"conf": conf, "iou": iou},
        prediction=pred, classes=list(getattr(model, "classes") or []))


def _http_predict(base_url: str, api_key: str, model_path: Path,
                  image_path: Path, conf: float, iou: float,
                  timeout: float = 120) -> dict:
    """经部署服务推理（与 C# 上位机同一条链路）。"""
    url = f"{base_url.rstrip('/')}/v1/infer/image"
    with open(image_path, "rb") as f:
        files = {"image": (Path(image_path).name, f.read())}
        r = requests.post(
            url, files=files,
            data={"model_id": str(model_path), "conf": str(conf),
                  "iou": str(iou)},
            headers={"X-XCcv-Key": api_key}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def compare_results(local: dict, http: dict,
                    iou_tol: float = IOU_TOL,
                    score_tol: float = SCORE_TOL) -> dict:
    """两份统一 JSON → {"consistent": bool, "details": [...]}。"""
    ld = (local or {}).get("detections") or []
    hd = (http or {}).get("detections") or []
    details: list = []
    if len(ld) != len(hd):
        details.append(f"检出数不一致：本地 {len(ld)} vs 服务 {len(hd)}")
    lb = [d.get("bbox") for d in ld]
    hb = [d.get("bbox") for d in hd]
    ls = [float(d.get("score") or 0) for d in ld]
    hs = [float(d.get("score") or 0) for d in hd]
    ious = iou_matrix(np.asarray(hb, float).reshape(-1, 4),
                      np.asarray(lb, float).reshape(-1, 4)) if hb and lb else None
    used: set = set()
    for hi, h in enumerate(hd):
        best, bi = -1.0, -1
        if ious is not None:
            for li in range(len(ld)):
                if li in used:
                    continue
                if ious[hi, li] > best:
                    best, bi = ious[hi, li], li
        if bi < 0 or best < iou_tol:
            details.append(f"服务检出#{hi + 1} {h.get('label')}@"
                           f"{h.get('score')} 无本地匹配框（IoU<{iou_tol}）")
            continue
        used.add(bi)
        l = ld[bi]
        if str(l.get("label")) != str(h.get("label")):
            details.append(f"类别不一致：本地 {l.get('label')} vs 服务 "
                           f"{h.get('label')}（框#{bi + 1}）")
        if abs(ls[bi] - hs[hi]) > score_tol:
            details.append(f"分数差偏大：本地 {ls[bi]:.3f} vs 服务 {hs[hi]:.3f}"
                           f"（框#{bi + 1}）")
    # 分类口径
    lc = (local or {}).get("classifications") or []
    hc = (http or {}).get("classifications") or []
    if (lc or hc) and (not lc or not hc or
                       str(lc[0].get("label")) != str(hc[0].get("label"))):
        details.append("分类 top1 不一致：本地 "
                       f"{lc[0].get('label') if lc else '—'} vs 服务 "
                       f"{hc[0].get('label') if hc else '—'}")
    return {"consistent": not details, "details": details,
            "local_detections": len(ld), "http_detections": len(hd)}


def run_consistency_check(model_path, image_path, base_url: str, api_key: str,
                          conf: float = 0.25, iou: float = 0.20) -> dict:
    """同步执行全流程（QThread.run 内调用；UI 层经线程包装）。"""
    model_path = Path(model_path)
    image_path = Path(image_path)
    local = _local_predict(model_path, image_path, conf, iou)
    http = _http_predict(base_url, api_key, model_path, image_path, conf, iou)
    out = compare_results(local, http)
    out["local"] = local
    out["http"] = http
    return out


class ConsistencyThread(QThread):
    """后台线程：本地+服务各推一次并对比（结果经 done 信号回 UI）。

    信号必须类级声明（pyqtSignal 注入实例无效）；本地推理加载模型可能
    数秒，绝不能在 UI 线程同步跑。
    """

    done = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, model_path, image_path, base_url: str, api_key: str,
                 conf: float = 0.25, iou: float = 0.20, parent=None):
        super().__init__(parent)
        self._args = (model_path, image_path, base_url, api_key, conf, iou)

    def run(self):
        try:
            self.done.emit(run_consistency_check(*self._args))
        except Exception as e:  # noqa: BLE001
            log.error("一致性自检失败", exc_info=True)
            self.failed.emit(f"{type(e).__name__}: {e}")

