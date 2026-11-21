"""模型工具：训练后模型的结构化通道剪枝（Pre-Finetuning Pruning 范式）。

- Torch-Pruning 的 DepGraph 自动识别并处理层间通道耦合（ResNet 捷径、FPN 等）；
- 按 L1 范数重要性剪枝，通道数对齐到 8 的倍数便于部署；
- 只剪骨干/颈部冗余通道：分类头、检测头、RPN 头与工业模块（CBAM/ECA/DCN）
  一律忽略，精度敏感部件不受影响；
- R-CNN 通过 _RcnnAdapter 适配器参与剪枝；YOLO 结构化剪枝**已受控停用**
  （torch-pruning 1.4~1.6 对 ultralytics 图的分组 index-mapping 会内存指数
  爆炸，默认快速失败并报原因；XCCV_YOLO_PRUNE=1 专家模式强制启用，配合
  prune_worker 进程隔离执行）；
- 剪后的权重文件保持原格式（xccv-classifier / xccv-rcnn / ultralytics YOLO），
  剪后网络以 net_module / ckpt.model 形式随文件保存；输出为明文 .pt，
  测试平台可直接加载推理；
- 源文件为明文 .pt；
- 剪枝失败（结构暂不支持、内存不足等）时返回原因，源文件不受影响。
"""

import logging
import os
import time
from pathlib import Path

import torch

from ..dataset.base import TaskType

log = logging.getLogger(__name__)

# 固定的示例输入：通道耦合与空间尺寸无关，统一用小尺寸张量降低追踪开销


def _classifier_ex():
    return torch.randn(1, 3, 64, 64)


def _rcnn_ex():
    # 注意：检测模型要求 3D 单图 [C, H, W]（与训练数据管线一致），不能给 batch 维；
    # masks 必须是 (N, H, W)——N 为该图目标数（与 boxes 行数一致），
    # 给成 (H, W) 会让 torchvision 的 roi_align 以 3D 输入崩溃
    img = torch.randn(3, 160, 160)
    target = {
        "boxes": torch.tensor([[10.0, 10.0, 120.0, 120.0]]),
        "labels": torch.tensor([1]),
        "image_id": torch.tensor([0]),
        "area": torch.tensor([12100.0]),
        "iscrowd": torch.tensor([0]),
        "masks": torch.zeros((1, 160, 160), dtype=torch.uint8),
    }
    return ([img], [target])


def _yolo_ex():
    return torch.randn(1, 3, 160, 160)


def _yolo_base_name(model_name: str, task: TaskType) -> str:
    """任务对应的 YOLO 预训练权重名（分割自动补 -seg 且不重复叠加）。"""
    if task is TaskType.SEGMENTATION and not model_name.endswith("-seg"):
        return f"{model_name}-seg"
    return model_name


class _RcnnAdapter(torch.nn.Module):
    """剪枝适配器：把检测模型的"多输入 + 后处理"包装成 TP 友好的单标量前向。

    TP 会用规范化后的参数直接调用 model(*args) 建图，检测模型的原生 forward
    （要求 list 图像、eval 下输出不可反传的检测结果）无法满足；适配器统一
    以训练模式计算损失和，始终返回可反传标量。
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, *args):
        if len(args) == 1:
            images, targets = args[0]
        else:
            images, targets = args
        if isinstance(images, torch.Tensor):
            images = [images]
        was = self.net.training
        self.net.train()
        try:
            losses = self.net(images, targets)
        finally:
            self.net.train(was)
        return sum(l for l in losses.values())


class _YoloForwardWrapper(torch.nn.Module):
    """YOLO 剪枝适配器：绕过 @smart_inference_mode 并临时开启参数梯度。

    ultralytics 加载模型时把所有参数 requires_grad=False（推理用途），
    predict 又挂有 @smart_inference_mode()——TP 的梯度追踪需要可反传前向，
    这里直接调用未被装饰的 _predict_once 走层序列并返回可反传标量。
    """

    def __init__(self, detection_model):
        super().__init__()
        self.detection_model = detection_model

    def forward(self, img):
        was = [p.requires_grad for p in self.detection_model.parameters()]
        for p in self.detection_model.parameters():
            p.requires_grad_(True)
        try:
            with torch.enable_grad():
                out = self.detection_model._predict_once(img)
        finally:
            for p, f in zip(self.detection_model.parameters(), was):
                p.requires_grad_(f)
        feats = []

        def _collect(o):
            if torch.is_tensor(o):
                feats.append(o)
            elif isinstance(o, (list, tuple)):
                for t in o:
                    _collect(t)

        _collect(out)
        loss = None
        for t in feats:
            if t.requires_grad:
                s = t.float().abs().sum()
                loss = s if loss is None else loss + s
        if loss is None:
            raise RuntimeError("YOLO 前向输出不可反传")
        return loss


def classifier_ignored_layers(net) -> list:
    """分类网忽略项：分类头（输出=类别数）与 stem 首层。"""
    if hasattr(net, "fc"):  # ResNet 系
        return [net.fc, net.conv1]
    return [net.classifier, net.features[0]]  # MobileNetV3


def rcnn_ignored_layers(net) -> list:
    """R-CNN 忽略项：RPN 头与 RoI 分类/回归/掩码头（保持检测输出结构不变）。"""
    ignored = [net.rpn.head, net.roi_heads.box_head, net.roi_heads.box_predictor]
    if hasattr(net.roi_heads, "mask_predictor"):
        ignored += [net.roi_heads.mask_head, net.roi_heads.mask_predictor]
    return ignored


def prune_model(model, example_inputs, ratio: float, ignored_layers=None,
                progress=None) -> str:
    """对 model 原地结构化剪枝；返回结果描述。

    progress: 可选回调 progress(phase, detail=...)，分相上报
    （trace=依赖追踪建图【重】、apply=执行剪枝、cleanup=内存回收），
    剪枝子进程据此向父窗口发 JSON 相位事件。
    """

    def _emit(phase: str, detail: str = "") -> None:
        if progress is not None:
            try:
                progress(phase, detail=detail)
            except Exception:  # noqa: BLE001  上报失败不影响剪枝
                pass

    if ratio <= 0:
        return "off"
    try:
        import gc

        import torch_pruning as tp
    except Exception as e:  # 依赖缺失
        log.error("torch-pruning 导入失败: %s", e, exc_info=True)
        return f"skipped(no torch-pruning: {type(e).__name__})"
    try:
        model.eval()
        before = sum(p.numel() for p in model.parameters())
        log.info("开始剪枝: ratio=%s, 参数量=%s, 追踪中…", ratio, f"{before:,}")
        _emit("trace", f"参数量 {before:,}，依赖追踪中…")
        t0 = time.perf_counter()
        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs,
            importance=tp.importance.MagnitudeImportance(p=1),
            pruning_ratio=float(ratio),
            ignored_layers=list(ignored_layers or []),
            round_to=8,
        )
        _emit("apply", "依赖图构建完成，执行剪枝…")
        pruner.step()
        del pruner  # 依赖图引用大量中间激活引用，分相尽早释放
        gc.collect()
        _emit("cleanup")
        after = sum(p.numel() for p in model.parameters())
        info = f"applied(ratio={ratio:g}, params {before}→{after}, " \
               f"{time.perf_counter() - t0:.1f}s)"
        log.info("剪枝完成: %s", info)
        return info
    except Exception as e:
        # 失败原因连同完整堆栈写入日志，便于目标部署机排障回传
        log.error("剪枝失败: %s: %s", type(e).__name__, e, exc_info=True)
        return f"failed({type(e).__name__}: {e})"


def _load_ckpt(src: Path):
    """读取权重 ckpt（明文 torch.load）。

    返回 (ckpt, raw)：raw 恒为 None（保留占位，供 YOLO 等需要真实文件
    路径的加载方扩展）。
    """
    return torch.load(src, map_location="cpu", weights_only=False), None


def _save_ckpt(ckpt, dst: Path) -> None:
    """保存 ckpt（明文落盘，.part 原子替换：子进程被紧急终止不会留下
    半截目标文件）。
    """
    tmp = dst.with_name(dst.name + ".part")
    try:
        torch.save(ckpt, tmp)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def prune_checkpoint(src: Path, dst: Path, ratio: float, progress=None) -> dict:
    """对已训练权重文件做剪枝，输出同格式新文件（源文件不动）。

    返回 {"status": applied/unsupported/..., "detail": ...,
    "params_before/after": int}。
    """
    src, dst = Path(src), Path(dst)
    log.info("剪枝任务开始: src=%s (%.1f MB), dst=%s, ratio=%s",
             src, src.stat().st_size / 1e6, dst, ratio)
    ckpt, raw = _load_ckpt(src)
    if progress is not None:
        try:
            progress("load", detail=f"已加载 {src.name}")
        except Exception:  # noqa: BLE001
            pass
    if isinstance(ckpt, dict) and ckpt.get("format") == "xccv-classifier":
        log.info("识别为分类权重 (arch=%s)", ckpt.get("arch"))
        return _prune_classifier_ckpt(ckpt, dst, ratio, progress)
    if isinstance(ckpt, dict) and ckpt.get("format") == "xccv-rcnn":
        log.info("识别为 R-CNN 权重 (arch=%s, task=%s)", ckpt.get("arch"), ckpt.get("task"))
        return _prune_rcnn_ckpt(ckpt, dst, ratio, progress)
    log.info("识别为 ultralytics YOLO 权重（实验性剪枝路径）")
    return _prune_yolo_ckpt(ckpt, raw, src, dst, ratio, progress)


def pretrain_prune(model_name: str, task, num_classes: int, ratio: float,
                   classes: list[str] | None = None, image_width: int = 0,
                   image_height: int = 0, device: str = "cpu",
                   industrial: bool = True, progress=None) -> dict:
    """训练前剪枝：按数据集构建预训练模型并剪枝，返回可直接训练的 ckpt 字典。

    与"训练后剪枝"的区别：源是官方预训练权重（而非训练产物），剪后模型保存到
    项目文件夹，训练平台将其作为起始模型进行训练 / 迁移学习。
    通道耦合追踪与 ignored 策略与训练后剪枝完全一致。
    """
    task = TaskType(task)
    w = int(image_width) if int(image_width) > 0 else 0
    h = int(image_height) if int(image_height) > 0 else 0
    classes = list(classes or [])
    log.info("训练前剪枝: 模型=%s 任务=%s 类别数=%d 比例=%s 输入=%s 设备=%s",
             model_name, task.value, num_classes, ratio, (h, w) if w and h else "默认",
             device)

    if task is TaskType.CLASSIFICATION:
        from .classifier import ClassifierModel

        model = ClassifierModel(model_name, num_classes, device=device,
                                use_pretrained=True, industrial=industrial)
        ex = torch.randn(1, 3, h or 224, w or 224)
        info = prune_model(model.net, ex, ratio, classifier_ignored_layers(model.net),
                           progress=progress)
        if not info.startswith("applied"):
            return {"status": info}
        ckpt = {
            "format": "xccv-classifier", "arch": model_name, "classes": classes,
            "image_size": (h, w) if w and h else ClassifierModel.default_image_size,
            "state_dict": model.net.state_dict(), "industrial_opt": industrial,
            "loss_type": "cross_entropy", "net_module": model.net,
            "pruned_pretrain": {"ratio": ratio},
        }
    elif task in (TaskType.DETECTION, TaskType.SEGMENTATION) and model_name in ("fasterrcnn", "maskrcnn"):
        from .rcnn import RCNNModel

        min_size = min(w, h) if w and h else RCNNModel.default_image_size
        max_size = max(w, h) if w and h else None
        model = RCNNModel(model_name, task, num_classes, device=device,
                          use_pretrained=True, industrial=industrial,
                          mask_head_light=False)
        if max_size is not None:  # 输入尺寸与项目统一时重建网络（保持先冻结后插入）
            model.net = model._build(model_name, num_classes, min_size, True,
                                     industrial, mask_head_light=False,
                                     max_size=max_size).to(model.device)
        adapter = _RcnnAdapter(model.net)
        info = prune_model(adapter, _rcnn_ex(), ratio, rcnn_ignored_layers(model.net),
                           progress=progress)
        if not info.startswith("applied"):
            return {"status": info}
        ckpt = {
            "format": "xccv-rcnn", "arch": model_name, "task": task.value,
            "classes": classes, "min_size": min_size,
            "state_dict": model.net.state_dict(), "industrial_opt": industrial,
            "mask_head_light": False, "net_module": model.net,
            "pruned_pretrain": {"ratio": ratio},
        }
    else:  # YOLO 家族
        if os.environ.get("XCCV_YOLO_PRUNE") != "1":
            return {"status": "failed(yolo-unsupported: torch-pruning 与 "
                              "ultralytics 图不兼容，剪枝分组会内存指数爆炸；"
                              "分类/R-CNN 剪枝不受影响。专家模式可设 "
                              "XCCV_YOLO_PRUNE=1 强制启用（配合独立进程执行）)"}
        base = _yolo_base_name(model_name, task)
        net = _build_yolo_net(base, num_classes, task, classes)
        wrapper = _YoloForwardWrapper(net)
        info = prune_model(wrapper, _yolo_ex(), ratio,
                           yolo_ignored_layers(net), progress=progress)
        if not info.startswith("applied"):
            return {"status": info}
        ckpt = {
            "model": net, "train_args": {"task": "segmentation" if task is TaskType.SEGMENTATION else "detect"},
            "xccv_classes": classes, "xccv_pruned_pretrain": {"ratio": ratio},
        }

    before = sum(p.numel() for p in _net_of(ckpt).parameters())
    log.info("训练前剪枝完成: %s", info)
    ckpt["pruned_pretrain"]["detail"] = info
    ckpt["pruned_pretrain"]["params"] = before
    return {"status": "applied", "ckpt": ckpt, "detail": info, "params": before}


def _net_of(ckpt: dict) -> torch.nn.Module:
    """取 ckpt 中的网络模块（classifier/rcnn 为 net_module，yolo 为 model）。"""
    return ckpt["net_module"] if "net_module" in ckpt else ckpt["model"]


def _build_yolo_net(base: str, num_classes: int, task,
                    classes: list[str] | None = None) -> torch.nn.Module:
    """按数据集类别数构建 YOLO 网络：yaml 定结构（头按 nc 重建）+ 预训练权重交集加载。

    classes 给定时同步设置 net.names——否则类别名为数字（"0"/"1"），
    训练平台加载后与数据集类别校验会误报不一致。
    """
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    from ..offline import WEIGHTS_DIR

    bundled = WEIGHTS_DIR / "ultralytics" / f"{base}.pt"
    weights_path = str(bundled) if bundled.is_file() else f"{base}.pt"
    pre = YOLO(weights_path)
    yaml_cfg = f"{base.replace('-seg', '')}{'-seg' if task is TaskType.SEGMENTATION else ''}.yaml"
    net = DetectionModel(yaml_cfg, nc=num_classes, ch=3, verbose=False)
    net.load(pre.model)  # 交集加载：骨干/颈部就位，检测头按 nc 重建
    if classes:
        net.names = {i: n for i, n in enumerate(classes)}
    return net


def yolo_ignored_layers(net) -> list:
    """YOLO 忽略项：Detect/Segment 头（保持输出结构不变）。

    net 可传 ultralytics YOLO 对象（model.model 为层列表）或裸 DetectionModel。
    """
    return [net.model[-1]]


def _finish(ckpt, dst: Path, net, info: str, before: int, after: int,
            progress=None) -> dict:
    if progress is not None:
        try:
            progress("save", detail=str(dst.name))
        except Exception:  # noqa: BLE001
            pass
    ckpt["net_module"] = net  # 剪后结构随文件保存，测试平台直接加载推理
    ckpt["pruned"] = {"info": info}
    # 瘦身：ultralytics 源 ckpt 携带的 optimizer 状态等训练期数据，对剪后推理无用
    for key in ("optimizer", "ema", "updates"):
        ckpt.pop(key, None)
    _save_ckpt(ckpt, dst)
    log.info("剪枝结果已保存: %s (%.1f MB)", dst, dst.stat().st_size / 1e6)
    return {"status": "applied", "detail": info,
            "params_before": before, "params_after": after}


def _prune_classifier_ckpt(ckpt: dict, dst: Path, ratio: float,
                           progress=None) -> dict:
    from .classifier import ClassifierModel

    model = ClassifierModel(ckpt["arch"], len(ckpt["classes"]), device="cpu",
                            use_pretrained=False,
                            industrial=ckpt.get("industrial_opt", False))
    from .modules import load_state_dict_compat

    load_state_dict_compat(model.net, ckpt["state_dict"])
    before = sum(p.numel() for p in model.net.parameters())
    info = prune_model(model.net, _classifier_ex(), ratio,
                       classifier_ignored_layers(model.net), progress=progress)
    if not info.startswith("applied"):
        return {"status": info}
    after = sum(p.numel() for p in model.net.parameters())
    ckpt["state_dict"] = model.net.state_dict()
    return _finish(ckpt, dst, model.net, info, before, after, progress)


def _prune_rcnn_ckpt(ckpt: dict, dst: Path, ratio: float,
                     progress=None) -> dict:
    from .rcnn import RCNNModel

    model = RCNNModel(ckpt["arch"], TaskType(ckpt["task"]), len(ckpt["classes"]),
                      device="cpu", use_pretrained=False,
                      industrial=ckpt.get("industrial_opt", False),
                      mask_head_light=ckpt.get("mask_head_light", False))
    from .modules import load_state_dict_compat

    load_state_dict_compat(model.net, ckpt["state_dict"])
    before = sum(p.numel() for p in model.net.parameters())
    adapter = _RcnnAdapter(model.net)
    info = prune_model(adapter, _rcnn_ex(), ratio,
                       rcnn_ignored_layers(model.net), progress=progress)
    if not info.startswith("applied"):
        return {"status": info}
    after = sum(p.numel() for p in model.net.parameters())
    ckpt["state_dict"] = model.net.state_dict()
    return _finish(ckpt, dst, model.net, info, before, after, progress)


def _prune_yolo_ckpt(ckpt: dict, raw: bytes | None, src: Path, dst: Path,
                     ratio: float, progress=None) -> dict:
    """YOLO 家族剪枝。

    **受控停用**：torch-pruning 1.4~1.6 各版本对 ultralytics 图（save 多消费
    + Concat/Reshape 链）病态不兼容——分组阶段的 index-mapping 按通道倍增
    连锁展开，内存指数爆炸（yolo11n 在 64px 输入下即可在数分钟内耗尽数十
    GB 内存）。默认快速失败并给出原因；`XCCV_YOLO_PRUNE=1` 可强制启用专家
    模式（务必配合 prune_worker 的独立进程隔离执行，水位看门狗会兜底终止）。
    """
    if os.environ.get("XCCV_YOLO_PRUNE") != "1":
        return {"status": "failed(yolo-unsupported: torch-pruning 与 "
                          "ultralytics 图不兼容，剪枝分组会内存指数爆炸；"
                          "分类/R-CNN 剪枝不受影响。专家模式可设 "
                          "XCCV_YOLO_PRUNE=1 强制启用（配合独立进程执行）)"}
    from ultralytics import YOLO

    model = YOLO(str(src))
    before = sum(p.numel() for p in model.model.parameters())
    wrapper = _YoloForwardWrapper(model.model)
    info = prune_model(wrapper, _yolo_ex(), ratio,
                       yolo_ignored_layers(model.model), progress=progress)
    if not info.startswith("applied"):
        return {"status": info}
    after = sum(p.numel() for p in model.model.parameters())
    ckpt["model"] = model.model  # ultralytics ckpt 本就以模块形式保存
    return _finish(ckpt, dst, model.model, info, before, after, progress)


