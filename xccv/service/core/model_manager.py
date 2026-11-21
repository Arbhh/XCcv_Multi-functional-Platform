"""已加载模型池：LRU 驻留 + 逐模型串行（GPU）+ 卸载清理。

线程安全约定：FastAPI sync 端点在线程池中并发执行——池级 RLock 保护
注册表；每个模型自带 lock 串行其推理（同 model_id 同时只有一次 GPU 推理，
与测试平台「批量期间禁 UI 线程并发推理」的 ToE3 防护同语义）。
加载/推理一律经 engine.models.registry.load_model / model.predict，
不复制任何引擎逻辑（服务是引擎的消费者，非改造者）。
"""

import gc
import itertools
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


class ModelNotFound(Exception):
    """model_id 不存在（→ 404）。"""


@dataclass
class ModelEntry:
    """一个已加载模型的服务侧句柄（model_id 对 C# 暴露，权重永不外泄）。"""

    model_id: str
    alias: str
    path: str
    model: object
    task: str
    classes: list
    device: str
    loaded_at: float
    params_mb: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    last_used: float = 0.0
    train_input: int = 0   # 训练输入边长（meta；>0 时推理自动降采样并还原）


def _estimate_params_mb(model) -> float:
    """参数量估算（fp32 口径 MB）：包装类沿 model/net 逐层解包取真实参数量，
    全部失败时回退 state_dict 计数（有些包装仅暴露 state_dict）。"""
    cur = model
    seen = {id(cur)}
    for _ in range(4):                       # 包装层级上限（防循环引用）
        try:
            n = sum(p.numel() for p in cur.parameters())
            if n:
                return round(n * 4 / 1e6, 1)
        except Exception:
            pass
        nxt = getattr(cur, "model", None) or getattr(cur, "net", None)
        if nxt is None or id(nxt) in seen:
            break
        seen.add(id(nxt))
        cur = nxt
    try:
        sd = cur.state_dict()
        return round(sum(v.numel() for v in sd.values()) * 4 / 1e6, 1)
    except Exception:
        log.debug("参数量查询失败（按 0.0 上报）")
        return 0.0


def _vram_line() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return "（GPU 显存 %.0f/%.0f MB）" % ((total - free) / 2**20,
                                                 total / 2**20)
    except Exception:
        log.debug("显存行探测失败")
    return ""


def _restore_prediction(pred, orig, scaled):
    """降采样推理结果 → 原图坐标（框线性还原，掩码最近邻放大）。"""
    oh, ow = orig.shape[:2]
    sh, sw = scaled.shape[:2]
    if (oh, ow) == (sh, sw):
        return pred
    fx, fy = ow / float(sw), oh / float(sh)
    boxes = getattr(pred, "boxes", None)
    if boxes:
        import numpy as np

        b = np.asarray(boxes, dtype=float).reshape(-1, 4)
        b[:, [0, 2]] *= fx
        b[:, [1, 3]] *= fy
        boxes = b.tolist()
    masks = getattr(pred, "masks", None)
    if masks:
        import cv2
        import numpy as np

        masks = [cv2.resize(np.asarray(m, np.uint8), (ow, oh),
                            interpolation=cv2.INTER_NEAREST) > 0
                 for m in masks]
    from ...engine.models.base import Prediction

    return Prediction(labels=list(pred.labels), scores=list(pred.scores),
                      boxes=boxes or [], masks=masks,
                      probs=getattr(pred, "probs", None),
                      texts=getattr(pred, "texts", None))


class ModelManager:
    def __init__(self, max_loaded: int = 3):
        self._max = max(1, int(max_loaded))
        self._lock = threading.RLock()
        self._entries: dict[str, ModelEntry] = {}
        self._seq = itertools.count(1)
        self._loading: dict[str, threading.Lock] = {}  # per-path 单飞锁（防同路径并发双载）

    # ---- 查询 ----
    def list(self) -> list[dict]:
        with self._lock:
            items = sorted(self._entries.values(), key=lambda e: e.loaded_at)
            return [{
                "model_id": e.model_id, "alias": e.alias, "task": e.task,
                "device": e.device, "path": e.path, "params_mb": e.params_mb,
                "loaded_at": e.loaded_at, "last_used": e.last_used,
                # §三（V0.7.3）：对外分数语义随模型状态透出（additive）
                "score_semantics": dict(getattr(e.model, "score_semantics",
                                                None) or {}),
            } for e in items]

    def get(self, model_id: str) -> ModelEntry:
        with self._lock:
            entry = self._entries.get(model_id)
        if entry is None:
            raise ModelNotFound(f"模型未加载或已卸载: {model_id}")
        return entry

    # ---- 生命周期 ----
    def load(self, path: Path, device: str = "auto", alias: str = "",
             warmup: bool = False) -> ModelEntry:
        """加载（阻塞）；同一路径重复加载返回既有句柄（幂等）。

        并发语义：同路径并发加载经 per-path 单飞锁只执行一次真实加载
        （其余调用等待后复用既有句柄，防止显存翻倍）。
        """
        from ...engine.models import registry  # 惰性：torch 重导入

        path = Path(path)
        key = str(path.resolve())
        with self._lock:
            for e in self._entries.values():
                if e.path == key:
                    e.last_used = time.time()
                    return e
            path_gate = self._loading.setdefault(key, threading.Lock())
        with path_gate:
            with self._lock:
                # 等锁期间他人可能已完成加载——单飞复用出口
                for e in self._entries.values():
                    if e.path == key:
                        e.last_used = time.time()
                        return e
            t0 = time.perf_counter()
            model = registry.load_model(path, device=device)
            task = getattr(getattr(model, "task", None), "value", "") or "unknown"
            classes = list(getattr(model, "classes", None) or [])
            try:
                train_input = int(getattr(model, "train_input", 0) or 0)
            except (TypeError, ValueError):
                train_input = 0
            if not train_input:
                meta = getattr(model, "meta", None) or {}
                try:
                    train_input = int(meta.get("train_input", 0) or 0)
                except (TypeError, ValueError, AttributeError):
                    train_input = 0
            safe = re.sub(r"[\\/:*?\"<>|\s]+", "_",
                          (alias or path.stem)).strip("_")[:32] or "model"
            with self._lock:
                entry = ModelEntry(
                    model_id=f"{safe}-{next(self._seq)}", alias=alias or path.stem,
                    path=key, model=model, task=task, classes=classes,
                    device=str(getattr(model, "device", device)),
                    loaded_at=time.time(),
                    params_mb=_estimate_params_mb(model),
                    train_input=train_input)
                # LRU：超上限先卸最久未用（淘汰留痕）
                while len(self._entries) >= self._max:
                    oldest = min(self._entries.values(), key=lambda e: e.last_used)
                    log.info("LRU 卸载模型 %s（%s）：超过驻留上限 %d",
                             oldest.model_id, oldest.alias, self._max)
                    self._drop(oldest.model_id)
                self._entries[entry.model_id] = entry
            log.info("模型已加载: %s（%s，task=%s，%.1fs，约 %.1f MB 参数）%s",
                     entry.model_id, path.name, task, time.perf_counter() - t0,
                     entry.params_mb, _vram_line())
            if warmup:
                self._warmup(entry)
            return entry

    def _warmup(self, entry: ModelEntry) -> None:
        import numpy as np

        try:
            size = max(getattr(entry.model, "default_image_size", 320) or 320, 64)
            dummy = np.zeros((size, size, 3), np.uint8)
            with entry.lock:
                entry.model.predict(dummy)
            log.info("模型预热完成: %s", entry.model_id)
        except Exception:
            log.warning("模型预热失败（不影响服务）: %s", entry.model_id,
                        exc_info=True)

    def infer(self, model_id: str, image_bgr, conf: float, iou: float,
              max_side: int = 0):
        """单帧推理（逐模型串行；返回 (Prediction, entry)）。

        max_side>0（或模型 meta.train_input 自适应）且图更大时先降采样
        推理，坐标/掩码按原图尺寸还原（大图速度/显存护栏）。
        """
        entry = self.get(model_id)
        side = int(max_side or 0) or int(entry.train_input or 0)
        with entry.lock:
            scaled = None
            h, w = image_bgr.shape[:2]
            if side and max(h, w) > side:
                scale = side / float(max(h, w))
                nw = max(1, int(round(w * scale)))
                nh = max(1, int(round(h * scale)))
                import cv2

                scaled = cv2.resize(image_bgr, (nw, nh),
                                    interpolation=cv2.INTER_AREA)
            pred = entry.model.predict(
                image_bgr if scaled is None else scaled, conf=conf, iou=iou)
            if scaled is not None:
                pred = _restore_prediction(pred, image_bgr, scaled)
            entry.last_used = time.time()   # 锁内更新（与 LRU 淘汰互斥）
        return pred, entry

    def unload(self, model_id: str) -> bool:
        with self._lock:
            if model_id not in self._entries:
                return False
            self._drop(model_id)
        return True

    def preload(self, paths, device: str = "auto", warmup: bool = False) -> None:
        """启动预载（config.min_loaded_models）；单个失败不阻断服务。"""
        for p in paths or []:
            try:
                self.load(Path(p), device=device, warmup=warmup)
            except Exception:
                log.warning("预载模型失败（跳过）: %s", p, exc_info=True)

    def _drop(self, model_id: str) -> None:
        entry = self._entries.pop(model_id, None)
        if entry is None:
            return
        # 与推理互斥：entry.lock 被 in-flight 推理持有时延迟到其结束再释放，
        # 卸载/LRU 淘汰不再能删掉"推理中"的模型
        with entry.lock:
            try:
                del entry.model
            except Exception:
                log.warning("模型对象释放失败", exc_info=True)
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                log.warning("CUDA 缓存清理失败", exc_info=True)
        log.info("模型已卸载: %s %s", model_id, _vram_line())

