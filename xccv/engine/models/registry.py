"""模型注册表：平台可选模型的名称、任务与创建/加载入口。"""

import atexit
import logging
import os
import tempfile
import time
from pathlib import Path

import torch

from ..dataset.base import TaskType
from .anomaly import AnomalyModel
from .base import BaseModel
from .cif import CifModel
from .classifier import ClassifierModel
from .ocr import OcrModel
from .rcnn import RCNNModel
from .subspacead import SubspaceAdModel
from .yolo import YoloModel

log = logging.getLogger(__name__)

# 旧版加密容器魔数（历史版本曾把权重封为 .xva/.xcs）。本版本只支持明文
# 权重，加载到该魔数时给出明确指引而不是抛出晦涩的解析错误。
_LEGACY_MAGIC = b"XCVA"

ENGINE_SUFFIX = ".engine"  # TensorRT 导出引擎（明文，含元数据头）


def _unlink_quiet(path: str) -> None:
    """atexit 清理：进程结束时删除 .engine 桥接临时文件（缺删不报错）。"""
    try:
        os.unlink(path)
    except OSError:
        pass


def _is_legacy_container(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(len(_LEGACY_MAGIC)) == _LEGACY_MAGIC
    except OSError:
        return False


# name -> (支持的任务, 中文说明, 创建函数(task, num_classes, device, pretrained))
MODEL_SPECS: dict[str, tuple] = {
    "resnet18": (TaskType.CLASSIFICATION, "ResNet18 分类",
                 lambda task, n, dev, pre: ClassifierModel("resnet18", n, dev, pre)),
    "resnet50": (TaskType.CLASSIFICATION, "ResNet50 分类",
                 lambda task, n, dev, pre: ClassifierModel("resnet50", n, dev, pre)),
    "mobilenet_v3_small": (TaskType.CLASSIFICATION, "MobileNetV3-Small 分类（轻量）",
                           lambda task, n, dev, pre: ClassifierModel("mobilenet_v3_small", n, dev, pre)),
    "yolo11n": (TaskType.DETECTION, "YOLO11n 检测（轻量）",
                lambda task, n, dev, pre: YoloModel("yolo11n", task, n, dev, pre)),
    "yolo11s": (TaskType.DETECTION, "YOLO11s 检测",
                lambda task, n, dev, pre: YoloModel("yolo11s", task, n, dev, pre)),
    "yolo11n-seg": (TaskType.SEGMENTATION, "YOLO11n 分割（轻量）",
                    lambda task, n, dev, pre: YoloModel("yolo11n", TaskType.SEGMENTATION, n, dev, pre)),
    "yolo11s-seg": (TaskType.SEGMENTATION, "YOLO11s 分割",
                    lambda task, n, dev, pre: YoloModel("yolo11s", TaskType.SEGMENTATION, n, dev, pre)),
    "fasterrcnn": (TaskType.DETECTION, "Faster R-CNN 检测",
                   lambda task, n, dev, pre: RCNNModel("fasterrcnn", TaskType.DETECTION, n, dev, pre)),
    "maskrcnn": (TaskType.SEGMENTATION, "Mask R-CNN 实例分割",
                 lambda task, n, dev, pre: RCNNModel("maskrcnn", TaskType.SEGMENTATION, n, dev, pre)),
    "maskrcnn18": (TaskType.SEGMENTATION, "Mask R-CNN 实例分割（ResNet18 轻量）",
                   lambda task, n, dev, pre: RCNNModel("maskrcnn18", TaskType.SEGMENTATION, n, dev, pre)),
    # 无监督异常检测（OK 图训练 + 热力图/连通域检出）
    "patchcore": (TaskType.ANOMALY, "PatchCore 异常检测（无监督，OK 图训练）",
                  lambda task, n, dev, pre: AnomalyModel(n, dev, pre)),
    # CIF 少样本异常检测（DINO 特征 + 模糊超图，RGB 分支）
    "cif": (TaskType.ANOMALY, "CIF 异常检测（少样本超图，DINO 特征）",
            lambda task, n, dev, pre: CifModel(n, dev, pre)),
    # SubspaceAD 少样本异常检测（DINOv2 PCA 子空间）
    "subspacead": (TaskType.ANOMALY, "SubspaceAD 异常检测（少样本 PCA 子空间）",
                   lambda task, n, dev, pre: SubspaceAdModel(n, dev, pre)),
    # OCR 字符识别（CRNN+CTC；rectangle 标注、label=文本内容）
    "crnn": (TaskType.OCR, "CRNN 字符识别（OCR，行文本）",
             lambda task, n, dev, pre: OcrModel(n, dev, pre)),
}

# 模型元数据：train_mode 声明训练形态（UI 适配：free=免训练范式，隐藏轮次类
# 参数、阶段化进度、Loss 区口径切换）；support_k_shot=支持集张数参数
# （k-shot 可调 1~12）。未列出的模型按 "epoch" 处理。
MODEL_META: dict[str, dict] = {
    "patchcore": {"train_mode": "free"},
    "cif": {"train_mode": "free"},
    "subspacead": {"train_mode": "free", "support_k_shot": True},
}


def list_models(task: TaskType | str | None = None) -> list[dict]:
    """列出可选模型；task 给定时只列出支持该任务的。

    附带 MODEL_META 元数据（train_mode/support_k_shot），训练平台按此做
    免训练范式 UI 适配。"""
    if task is not None:
        task = TaskType(task)
    return [
        {"name": name, "task": info[0].value, "desc": info[1],
         "train_mode": str(MODEL_META.get(name, {}).get("train_mode")
                            or "epoch"),
         "support_k_shot": bool(MODEL_META.get(name, {}).get(
             "support_k_shot", False))}
        for name, info in MODEL_SPECS.items()
        if task is None or info[0] is task
    ]


def create_model(name: str, task: TaskType | str, num_classes: int,
                 device: str = "auto", use_pretrained: bool = True) -> BaseModel:
    task = TaskType(task)
    if name not in MODEL_SPECS:
        raise ValueError(f"未知模型: {name}，可选: {', '.join(MODEL_SPECS)}")
    supported, _, factory = MODEL_SPECS[name]
    if task not in (supported,):
        raise ValueError(f"模型 {name} 不支持任务 {task.label}（仅支持 {supported.label}）")
    return factory(task, num_classes, device, use_pretrained)


def load_model(weights_path: Path, device: str = "auto") -> BaseModel:
    """按权重文件自动识别模型类型并加载（测试平台入口）。

    平台自有格式（xccv-classifier / xccv-rcnn / xccv-anomaly / xccv-cif /
    xccv-subspacead / xccv-ocr）按 ckpt format 直接重建；
    否则按 ultralytics YOLO 权重处理。
    .engine 为 TensorRT 导出引擎（加载需 GPU + TensorRT），按内嵌元数据路由。
    """
    path = Path(weights_path)
    if _is_legacy_container(path):
        raise ValueError(
            f"{path.name}：检测到旧版加密权重容器（.xva/.xcs）。"
            "当前版本只支持明文 .pt / .engine 权重，请重新训练或重新导出。")
    if path.suffix == ENGINE_SUFFIX and not torch.cuda.is_available():
        # TensorRT 引擎必须有 CUDA 设备；装了 tensorrt 但无 GPU 时
        # trt.Runtime 构造会以 pybind11 TypeError 失败，先验设备再加载。
        from ..trt_runtime import _TRT_UNAVAILABLE_MSG, TrtUnavailableError

        log.info("跳过 .engine 加载：当前环境无可用 CUDA 设备（%s）", path.name)
        raise TrtUnavailableError(_TRT_UNAVAILABLE_MSG)
    t0 = time.perf_counter()
    log.info("加载权重: %s (%.1f MB)", path, path.stat().st_size / 1e6)

    # TensorRT 导出引擎：元数据/检测头权重内嵌容器（XCVE 元数据头），按 kind
    # 路由；YOLO 引擎经临时文件由 ultralytics 原生加载；无元数据的引擎按
    # YOLO 处理
    if path.suffix == ENGINE_SUFFIX:
        from ..model_export import unpack_engine_blob
        from ..trt_runtime import (
            TrtComputeMismatchError,
            engine_compute_mismatch,
        )

        meta, engine_bytes, extra = unpack_engine_blob(path.read_bytes())
        if meta:  # 元数据含生成时算力：跨显卡不匹配加载前即拦
            pre = engine_compute_mismatch(meta)
            if pre:
                log.info("跳过 .engine 加载：生成显卡与本机算力不同（%s）", path.name)
                raise TrtComputeMismatchError(pre)
        if meta and meta.get("kind") == "classifier":
            from ..trt_runtime import TrtClassifierModel

            m = TrtClassifierModel(meta["classes"], meta.get("image_size", 160), device)
            m.load_engine(engine_bytes)
            log.info("格式: TensorRT 分类引擎 (arch=%s, %d 类, 加载耗时 %.1fs)",
                     meta.get("arch"), len(meta["classes"]), time.perf_counter() - t0)
            return m
        if meta and meta.get("kind") == "rcnn":
            from ..trt_runtime import TrtRcnnModel

            m = TrtRcnnModel(meta, device)
            m.load_engine(engine_bytes, heads_ckpt=extra or None)
            log.info("格式: TensorRT R-CNN 引擎 (arch=%s, 任务=%s, %d 类, 加载耗时 %.1fs)",
                     meta.get("arch"), meta.get("task"), len(meta["classes"]),
                     time.perf_counter() - t0)
            return m
        log.info("格式: TensorRT YOLO 引擎（ultralytics 原生加载）")
        # YOLO 需要真实文件路径：用剥离元数据后的引擎字节桥接。ultralytics
        # 在 predict 时才惰性读取引擎文件（AutoBackend 重新 open），临时文件
        # 必须保留到进程退出（加载后立即 unlink 会让推理找不到文件）
        fd, tmp = tempfile.mkstemp(suffix=".engine")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(engine_bytes)
            model = YoloModel("yolo11n", TaskType.DETECTION, 1, device,
                              use_pretrained=False)
            model.load_weights(Path(tmp))
            model.apply_engine_meta(meta)  # 容器头 task/classes/imgsz 回填
        except Exception as e:
            # 旧版引擎无算力记录：从 ultralytics 的 TRT 报错文本里识别
            # 跨显卡不匹配并转译
            from ..trt_runtime import compute_mismatch_from_text

            msg = compute_mismatch_from_text(str(e))
            os.unlink(tmp)  # 失败路径无后续推理，立即可删
            if msg:
                raise TrtComputeMismatchError(msg) from e
            raise
        atexit.register(lambda p=tmp: _unlink_quiet(p))
        return model

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        log.warning("权重文件 torch.load 失败，按 ultralytics YOLO 权重重试", exc_info=True)
        ckpt = None
    if isinstance(ckpt, dict) and ckpt.get("format") == "xccv-classifier":
        log.info("格式: xccv-classifier (arch=%s, %s 类, 加载耗时 %.1fs)",
                 ckpt.get("arch"), len(ckpt.get("classes") or []), time.perf_counter() - t0)
        model = ClassifierModel("resnet18", 2, device, use_pretrained=False)  # 占位，load_weights 会重建
        model.load_weights(path)
        return model
    if isinstance(ckpt, dict) and ckpt.get("format") == "xccv-rcnn":
        log.info("格式: xccv-rcnn (arch=%s, task=%s, 加载耗时 %.1fs)",
                 ckpt.get("arch"), ckpt.get("task"), time.perf_counter() - t0)
        model = RCNNModel("fasterrcnn", TaskType.DETECTION, 1, device, use_pretrained=False)
        model.load_weights(path)
        return model
    if isinstance(ckpt, dict) and ckpt.get("format") == "xccv-anomaly":
        log.info("格式: xccv-anomaly (记忆库 %s patch, 加载耗时 %.1fs)",
                 getattr(ckpt.get("memory"), "shape", "?"), time.perf_counter() - t0)
        model = AnomalyModel(1, device, use_pretrained=False)
        model.load_weights(path)
        return model
    if isinstance(ckpt, dict) and ckpt.get("format") == "xccv-cif":
        log.info("格式: xccv-cif (记忆库 %s 节点, 加载耗时 %.1fs)",
                 getattr(ckpt.get("memory"), "shape", "?"), time.perf_counter() - t0)
        model = CifModel(1, device, use_pretrained=False)
        model.load_weights(path)
        return model
    if isinstance(ckpt, dict) and ckpt.get("format") == "xccv-subspacead":
        log.info("格式: xccv-subspacead (PCA 子空间 k=%s, 加载耗时 %.1fs)",
                 ckpt.get("params", {}).get("k_components", "?"),
                 time.perf_counter() - t0)
        model = SubspaceAdModel(1, device, use_pretrained=False)
        model.load_weights(path)
        return model
    if isinstance(ckpt, dict) and ckpt.get("format") == "xccv-ocr":
        log.info("格式: xccv-ocr (字符集 %d, 加载耗时 %.1fs)",
                 len(ckpt.get("charset") or []), time.perf_counter() - t0)
        model = OcrModel(1, device, use_pretrained=False)
        model.load_weights(path)
        return model
    log.info("格式: ultralytics YOLO (加载耗时 %.1fs)", time.perf_counter() - t0)
    model = YoloModel("yolo11n", TaskType.DETECTION, 1, device, use_pretrained=False)
    model.load_weights(path)
    return model

