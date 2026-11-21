"""TensorRT 引擎推理运行时：加载 .engine（TensorRT 导出引擎）做产线推理。

- 显存缓冲统一用 torch 的 CUDA 分配器（无需 pycuda / cuda-python）；
- TrtEngine：通用引擎封装（反序列化 + 同步推理，静态 shape）；
- TrtClassifierModel：分类引擎的 BaseModel 适配（预处理与 ClassifierModel
  完全一致），测试平台无需感知差异。
- YOLO 引擎由 ultralytics 原生加载（自带元数据），不走本模块。
"""

import logging
from pathlib import Path

import numpy as np
import torch

from .dataset.base import TaskType
from .model_export import _ensure_trt_dll_search_path
from .models.base import BaseModel, Prediction, resolve_device

log = logging.getLogger(__name__)


class TrtUnavailableError(RuntimeError):
    """当前环境缺 GPU/TensorRT，无法加载 .engine 加速模型。

    平台侧按"已知预期情况"处理：只给中文指引文案+日志 INFO，不弹裸堆栈。
    """


_TRT_UNAVAILABLE_MSG = (
    "该模型为 TensorRT 加速模型（.engine）：加载与推理需要 GPU 环境和 TensorRT "
    "运行时，当前环境不可用。请在配备 NVIDIA 显卡且安装 TensorRT 的机器上使用，"
    "或改用未加速的模型文件（.pt）。")


class TrtComputeMismatchError(TrtUnavailableError):
    """加速模型由另一型号显卡（不同算力）生成。

    TensorRT 引擎与显卡架构（compute capability）绑定，跨架构反序列化必然
    失败——不是文件损坏。指引：在本机「模型工具 → 模型加速」重新生成。
    平台侧同样按预期情况处理（TrtUnavailableError 子类，纯文案不弹堆栈）。
    """


# 常见算力 → 架构名（文案人话化；未列出的只报算力数值）
_CC_NAMES = {(7, 0): "Volta", (7, 5): "Turing（RTX 20 系）",
             (8, 0): "Ampere（A 系）", (8, 6): "Ampere（RTX 30 系）",
             (8, 9): "Ada（RTX 40 系）", (9, 0): "Hopper（H 系）",
             (12, 0): "Blackwell"}


def _cc_label(cc) -> str:
    if not cc:
        return "未知显卡"
    major, minor = int(cc[0]), int(cc[1])
    arch = _CC_NAMES.get((major, minor))
    return f"算力 {major}.{minor}" + (f"（{arch}）" if arch else "")


def local_compute_capability():
    """本机显卡算力 (major, minor)；无 CUDA 设备/查询失败返回 None。"""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        cap = torch.cuda.get_device_capability(0)
        return int(cap[0]), int(cap[1])
    except Exception:
        log.debug("CUDA 算力查询失败")
        return None

def engine_mismatch_message(built_cc, local_cc) -> str:
    """跨显卡算力不匹配的中文指引（built=引擎生成时的算力，local=本机）。"""
    return (f"该加速模型（.engine）是在{_cc_label(built_cc)}的显卡上生成的，与当前"
            f"显卡{_cc_label(local_cc)}架构不同；TensorRT 引擎与显卡架构绑定，"
            "不能跨架构使用，文件本身没有损坏。请在「模型工具 → 模型加速」用源"
            "权重（.pt）在本机重新生成加速模型。")


def engine_compute_mismatch(meta) -> str:
    """引擎元数据算力预检（加载前，不必先反序列化）。

    meta 记录了生成时算力（cc_build）且与本机不同 → 返回中文指引；其余情况
    （旧版无记录 / 无 CUDA 设备 / 算力一致）返回 None（交给既有链路）。
    """
    if not isinstance(meta, dict):
        return None
    built = meta.get("cc_build")
    if not (isinstance(built, (list, tuple)) and len(built) == 2):
        return None
    local = local_compute_capability()
    if local is None:
        return None
    built_cc = (int(built[0]), int(built[1]))
    if built_cc == local:
        return None
    return engine_mismatch_message(built_cc, local)


def _parse_compute_mismatch(texts):
    """TRT 日志/异常文本 → ((本机算力), (生成算力))；无匹配返回 None。

    TRT 10.x 文案形如 "… expecting compute 7.5 got compute 8.9, please rebuild."
    （expecting=当前设备，got=引擎构建时算力）。
    """
    import re

    for t in texts:
        m = re.search(r"expecting compute (\d+)\.(\d+) got compute (\d+)\.(\d+)",
                      str(t))
        if m:
            return ((int(m.group(1)), int(m.group(2))),
                    (int(m.group(3)), int(m.group(4))))
    return None


def compute_mismatch_from_text(text: str):
    """异常文本含跨显卡特征 → 中文指引；否则 None（YOLO/ultralytics 链路用）。"""
    hit = _parse_compute_mismatch([text])
    if hit is None and "incompatible" in str(text).lower() \
            and "compute" in str(text).lower():
        return engine_mismatch_message(None, None)
    if hit is None:
        return None
    expect, built = hit
    return engine_mismatch_message(built, expect)


def _make_capture_logger(trt):
    """捕获 TRT WARNING 及以上日志的 Logger（反序列化失败原因从日志取）。

    TRT 10.x 的 ILogger 构造不收参数（severity 过滤由 log() 自行完成）；
    Severity 枚举越小越严重：INTERNAL_ERROR=0 / ERROR=1 / WARNING=2。
    """

    class _CaptureLogger(trt.ILogger):
        def __init__(self):
            trt.ILogger.__init__(self)
            self.messages: list = []

        def log(self, severity, msg):
            try:
                keep = int(severity) <= int(trt.ILogger.WARNING)
            except Exception:
                log.debug("TRT 日志级别比较失败（全保留）")
                keep = True
            if keep:
                self.messages.append(str(msg))

    return _CaptureLogger()


def _trt_dtype_to_torch(dt):
    import torch
    import tensorrt as trt

    return {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT64: torch.int64,
        trt.DataType.INT8: torch.int8,
        trt.DataType.BOOL: torch.bool,
    }[dt]


class TrtEngine:
    """TensorRT 静态引擎封装：反序列化 + 一次性显存缓冲 + 同步推理。"""

    def __init__(self, path):
        # DLL 搜索路径必须先布置（weights\tensorrt / venv\bin 候选）：TRT
        # 绑定只查 PATH 与 ..\tensorrt.libs，裸 import 在"DLL 不在系统
        # PATH"的机器上直接 FileNotFoundError
        _ensure_trt_dll_search_path()
        try:
            import tensorrt as trt
        except ModuleNotFoundError as e:  # 无 GPU / 未装 TensorRT：友好降级
            raise TrtUnavailableError(_TRT_UNAVAILABLE_MSG) from e
        if not torch.cuda.is_available():  # 装了 TensorRT 但无 CUDA 设备
            raise TrtUnavailableError(_TRT_UNAVAILABLE_MSG)

        if isinstance(path, (bytes, bytearray)):  # 明文引擎字节直传
            raw = bytes(path)
        else:
            path = Path(path)
            raw = path.read_bytes()
        # 捕获日志的 Logger：跨显卡算力不匹配从日志文案转译。
        # 构造放在 nullptr 转译块之外：自身构造失败只降级为普通 Logger，
        # 绝不能被下面的 TypeError 转译伪装成"无 GPU"
        try:
            logger = _make_capture_logger(trt)
        except Exception:
            log.debug("捕获型 Logger 构造失败（降级普通 Logger）")
            logger = trt.Logger(trt.Logger.WARNING)
        try:
            runtime = trt.Runtime(logger)
        except TypeError as e:  # pybind11 "factory function returned nullptr"：驱动/设备不可用
            raise TrtUnavailableError(_TRT_UNAVAILABLE_MSG) from e
        try:
            engine = runtime.deserialize_cuda_engine(raw)
        except Exception as e:  # 部分 TRT 版本直接抛错而非返回 None
            msg = compute_mismatch_from_text(
                str(e) + " " + " ".join(getattr(logger, "messages", [])))
            if msg:
                log.info(".engine 跨显卡算力不匹配：%s", path if not isinstance(
                    path, (bytes, bytearray)) else "(内存引擎)")
                raise TrtComputeMismatchError(msg) from e
            raise
        if engine is None:
            hit = _parse_compute_mismatch(getattr(logger, "messages", []))
            if hit is not None:
                expect, built = hit
                log.info(".engine 跨显卡算力不匹配：构建 %s / 本机 %s", built, expect)
                raise TrtComputeMismatchError(engine_mismatch_message(built, expect))
            raise RuntimeError(
                "TensorRT 引擎反序列化失败：.engine 文件可能损坏，或与本机 TensorRT "
                "版本不匹配（请在本机重新执行模型加速）")
        self._ctx = engine.create_execution_context()
        if self._ctx is None:
            raise RuntimeError("TensorRT 执行上下文创建失败（显存不足或引擎不兼容）")
        self._torch = torch
        self._io: dict[str, tuple] = {}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = tuple(int(s) for s in engine.get_tensor_shape(name))
            dtype = _trt_dtype_to_torch(engine.get_tensor_dtype(name))
            is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            self._io[name] = (torch.empty(shape, dtype=dtype, device="cuda"), is_input)
        self.input_names = [n for n, (_, is_in) in self._io.items() if is_in]
        self.output_names = [n for n, (_, is_in) in self._io.items() if not is_in]

    @property
    def input_shape(self) -> tuple:
        name = self.input_names[0]
        return tuple(int(s) for s in self._io[name][0].shape)

    def infer(self, feed: dict, gpu: bool = False) -> dict:
        """执行一次同步推理。gpu=True 时返回 torch CUDA 张量（零拷贝，
        供后续 torch 模块直接消费）；默认返回 float32 numpy。"""
        torch = self._torch
        for name, arr in feed.items():
            t, _ = self._io[name]
            if not torch.is_tensor(arr):
                arr = torch.from_numpy(np.ascontiguousarray(arr))
            t.copy_(arr.to(t.device, t.dtype))
        for name, (t, _) in self._io.items():
            self._ctx.set_tensor_address(name, int(t.data_ptr()))
        if not self._ctx.execute_async_v3(int(torch.cuda.current_stream().cuda_stream)):
            raise RuntimeError("TensorRT 推理执行失败（详见日志）")
        torch.cuda.synchronize()
        if gpu:
            return {n: self._io[n][0] for n in self.output_names}
        return {n: self._io[n][0].float().cpu().numpy() for n in self.output_names}


def paste_frame_masks(mask_probs, boxes_frame, frame_hw, orig_hw,
                      thresh: float = 0.5) -> list:
    """把 RoI 头输出的低分辨率掩码概率图贴回原图尺寸。

    torchvision 完整前向里由 GeneralizedRCNNTransform.postprocess 调
    paste_masks_in_image 完成（28×28 → 各检出框区域 → 原图坐标系）；
    特征段引擎路径跳过了该步骤，必须自行补上：掩码只在框内插值，
    框外恒为背景。若把 28×28 概率图直接拉伸到整张原图，小目标掩码
    会膨胀几个数量级（面积虚高的根因）。

    mask_probs: (N, h, w) 概率图（通常 28×28，sigmoid 后）；
    boxes_frame: (N, 4) 引擎帧坐标系 xyxy；
    frame_hw / orig_hw: (H, W) 引擎输入帧与原图尺寸。
    返回 N 张原图尺寸 bool 掩码（与 Prediction.masks 同构）。
    """
    import cv2

    fh, fw = int(frame_hw[0]), int(frame_hw[1])
    h0, w0 = int(orig_hw[0]), int(orig_hw[1])
    out = []
    for prob, box in zip(mask_probs, boxes_frame):
        canvas = np.zeros((h0, w0), dtype=bool)
        x1f, y1f, x2f, y2f = (float(v) for v in box)
        # 引擎帧坐标 → 原图坐标（predict 用同一组缩放系数）
        x1 = int(round(x1f * w0 / fw))
        y1 = int(round(y1f * h0 / fh))
        x2 = int(round(x2f * w0 / fw))
        y2 = int(round(y2f * h0 / fh))
        x1, y1 = max(0, min(x1, w0 - 1)), max(0, min(y1, h0 - 1))
        x2, y2 = max(x1 + 1, min(x2, w0)), max(y1 + 1, min(y2, h0))
        rw, rh = x2 - x1, y2 - y1
        if rw > 0 and rh > 0 and prob.size:
            resized = cv2.resize(np.asarray(prob, dtype=np.float32), (rw, rh),
                                 interpolation=cv2.INTER_LINEAR)
            canvas[y1:y2, x1:x2] = resized > thresh
        out.append(canvas)
    return out


class TrtRcnnModel(BaseModel):
    """R-CNN（检测/分割）引擎模型（特征段加速）。

    引擎只承担 归一化 + Backbone + FPN（计算量占比最高、纯卷积、静态 shape，
    TRT 原生支持）；RPN/RoI 头含动态 NMS 与 RoiAlign（TRT 无原生层/插件），
    留在 torch 侧直接消费引擎输出的显存特征（零拷贝）。predict 把整图拉伸
    到引擎输入帧，推理后把检出缩放回原图坐标。
    """

    family = "trt-rcnn"
    supported_tasks = (TaskType.DETECTION, TaskType.SEGMENTATION)

    def __init__(self, meta: dict, device: str = "cuda"):
        super().__init__(len(meta["classes"]), device)
        self.task = TaskType(meta["task"])
        self.arch = meta.get("arch", "maskrcnn")
        self.classes = list(meta["classes"])
        self.num_classes = len(self.classes)
        self.with_masks = self.task is TaskType.SEGMENTATION
        self._min_size = int(meta.get("min_size", 640))
        self._max_size = int(meta.get("max_size", 1066))
        self._feat_keys = list(meta.get("feat_keys") or ["0", "1", "2", "3", "pool"])
        self._heads = None  # 惰性重建的 RPN/RoI 头（引擎构建时一并打包进容器）
        self._engine: TrtEngine | None = None
        self.device = resolve_device(device)

    def load_engine(self, engine_path, heads_ckpt: bytes | None = None) -> None:
        # 支持明文引擎字节直传（registry 已剥离元数据头）或文件路径
        if isinstance(engine_path, (bytes, bytearray)):
            self._engine = TrtEngine(bytes(engine_path))
        else:
            self._engine = TrtEngine(engine_path)
        log.info("R-CNN 特征引擎就绪: 输入 %s, 特征 %s, 任务=%s, %d 类",
                 self._engine.input_shape, self._engine.output_names,
                 self.task.value, self.num_classes)
        if heads_ckpt:
            self._load_heads(heads_ckpt)

    def _load_heads(self, heads_ckpt: bytes) -> None:
        """从容器内嵌的头权重重建 RPN/RoI 检测头（torch 侧执行）。

        容器格式两代：
        - xcve-heads-v2（当前）：state_dict 只含 rpn./roi_heads. 键（骨干与 FPN
          在引擎里，不重复封装），需携带 mask_head_light 等结构开关——漏传
          该开关会导致轻量掩码头模型（2 层 128 通道）被按默认头（4 层 256 通道）
          重建，加载 .engine 必然报 shape mismatch；
        - 旧容器：整网 state_dict，按原逻辑严格加载。
        """
        import io

        from .models.rcnn import RCNNModel

        ckpt = torch.load(io.BytesIO(heads_ckpt), map_location="cpu", weights_only=False)
        m = RCNNModel(self.arch, self.task, self.num_classes, device="cpu",
                      use_pretrained=False, industrial=ckpt.get("industrial_opt", False),
                      mask_head_light=ckpt.get("mask_head_light", False))
        try:
            if ckpt.get("format") in ("xcve-heads-v2", "xcs-heads-v2"):
                result = m.net.load_state_dict(ckpt["state_dict"], strict=False)
                if result.unexpected_keys:
                    raise RuntimeError(f"权重包含未知参数: {result.unexpected_keys[:8]}")
                # 骨干/FPN 由引擎承担：置空本地随机初始化骨干，省一份无用显存
                m.net.backbone = torch.nn.Identity()
            else:
                m.net.load_state_dict(ckpt["state_dict"])
        except RuntimeError as e:
            raise RuntimeError(
                f"引擎容器中的检测头权重与模型结构不符（{e}）。"
                "该引擎容器可能由旧版本导出：请在模型工具平台对源权重重新执行「模型加速」。") from e
        # transform 的 min/max 不在 state_dict 中：必须显式对齐到引擎输入帧，
        # 否则头在错误的分辨率上消费特征（分数严重偏差的根因）
        m.net.transform.min_size = (self._min_size,)
        m.net.transform.max_size = self._max_size
        self._heads = m.net.to(self.device).eval()
        log.info("R-CNN 检测头（torch 侧）就绪: RPN + RoI 头, %d 参数",
                 sum(p.numel() for p in self._heads.parameters()))

    def load_weights(self, path) -> None:
        self.load_engine(path)

    def fit(self, data, cfg, reporter) -> dict:  # pragma: no cover - 引擎不可训练
        raise RuntimeError("TensorRT 引擎模型为只读推理产物，不支持训练")

    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray, conf: float = 0.25, iou: float = 0.20) -> Prediction:
        import cv2

        if self._engine is None:
            raise RuntimeError("引擎尚未加载")
        if self._heads is None:
            raise RuntimeError("检测头未加载（引擎容器缺少头权重）")
        H, W = self._engine.input_shape[1:]
        h0, w0 = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        im = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(im).permute(2, 0, 1).float().div_(255.0)

        feats = self._engine.infer({self._engine.input_names[0]: x.numpy()}, gpu=True)
        # FP16 引擎下输出缓冲仍为 FP32（I/O 保持 ONNX dtype），.float() 兜底无开销
        features = {k: feats[f"feat_{k}"].float()
                    for k in self._feat_keys if f"feat_{k}" in feats}

        heads = self._heads
        images, _ = heads.transform([x.to(self.device)], None)
        proposals, _ = heads.rpn(images, features, None)
        detections, _ = heads.roi_heads(features, proposals, images.image_sizes, None)
        det = detections[0]

        scores = det["scores"].cpu().numpy()
        keep = scores >= conf
        boxes = det["boxes"].cpu().numpy()[keep]
        scores = scores[keep]
        labels = det["labels"].cpu().numpy()[keep]
        out_masks = None
        if self.with_masks and "masks" in det and keep.any():
            # det["masks"] 是 RoI 头输出的 (N,1,28,28) 概率图（贴回原图是
            # GeneralizedRCNNTransform.postprocess 的职责，头本身不做）：
            # 必须按框贴回，严禁整图拉伸
            mk = det["masks"][:, 0].cpu().numpy()[keep]
            out_masks = paste_frame_masks(mk, boxes, (H, W), (h0, w0))
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, [0, 2]] *= w0 / W
            boxes[:, [1, 3]] *= h0 / H
        return Prediction(
            labels=(labels - 1).astype(int).tolist(), scores=scores.tolist(),
            boxes=boxes.tolist(), masks=out_masks)




class TrtClassifierModel(BaseModel):
    """分类引擎模型：对外接口与 ClassifierModel 一致（predict / classes / task）。"""

    family = "trt-classifier"
    supported_tasks = (TaskType.CLASSIFICATION,)

    def __init__(self, classes: list[str], input_size, device: str = "cuda"):
        super().__init__(len(classes), device)
        self.task = TaskType.CLASSIFICATION
        self.classes = list(classes)
        self.num_classes = len(classes)
        self._input_size = input_size
        self._engine: TrtEngine | None = None
        self.device = resolve_device(device)

    def load_engine(self, engine_path) -> None:
        # 支持明文引擎字节直传（registry 已剥离元数据头）或文件路径
        if isinstance(engine_path, (bytes, bytearray)):
            self._engine = TrtEngine(bytes(engine_path))
        else:
            self._engine = TrtEngine(engine_path)
        log.info("分类引擎就绪: 输入 %s, %d 类, 输出 %s",
                 self._engine.input_shape, self.num_classes, self._engine.output_names)

    def load_weights(self, path) -> None:
        """统一加载入口：引擎文件即权重（与 BaseModel.load_weights 对齐）。"""
        self.load_engine(path)

    def fit(self, data, cfg, reporter) -> dict:  # pragma: no cover - 引擎不可训练
        raise RuntimeError("TensorRT 引擎模型为只读推理产物，不支持训练")

    def predict(self, image_bgr: np.ndarray, conf: float = 0.0, iou: float = 0.0) -> Prediction:
        import cv2
        import torch
        from PIL import Image
        from torchvision import transforms as T

        from .models.classifier import IMAGENET_MEAN, IMAGENET_STD

        if self._engine is None:
            raise RuntimeError("引擎尚未加载")
        hw = (self._input_size, self._input_size) if isinstance(self._input_size, int) \
            else tuple(self._input_size)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        x = T.Compose([
            T.Resize((int(hw[0]), int(hw[1]))),  # 与 ClassifierModel 同口径：方形拉伸
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])(Image.fromarray(rgb)).unsqueeze(0).numpy()
        outs = self._engine.infer({self._engine.input_names[0]: x})
        logits = outs[self._engine.output_names[0]][0]
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        top = int(np.argmax(probs))
        return Prediction(labels=[top], scores=[float(probs[top])],
                          probs=[float(p) for p in probs])

