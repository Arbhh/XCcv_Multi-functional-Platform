"""TensorRT 引擎导出：把训练/剪枝产物编译为 TensorRT engine，产线推理提速。

- YOLO 家族：走 ultralytics 原生 export(format="engine")（官方优化路径）；
- 分类 / R-CNN：torch.onnx.export 到内存缓冲 → TensorRT 解析构建 → engine，
  ONNX 中间产物全程在内存中，不落盘；
- 引擎产物为明文 .engine（含 XCVE 元数据头）：单文件自包含，测试平台可直接
  加载预测（需 GPU + TensorRT 环境）；
- 构建需要 NVIDIA GPU + TensorRT 运行库：缺失时抛出带指引的 RuntimeError，
  日志记录完整过程（依赖检查、输入尺寸、耗时、构建错误），便于目标部署机排障。
"""

import io
import json
import logging
import os
import struct
import sys
import time
from pathlib import Path

import torch

from .models.base import resolve_device
from .models.registry import load_model

log = logging.getLogger(__name__)

# 引擎容器 payload 格式：b"XCVE" + <I 元数据长> + <I 附加数据长> + JSON + 附加 + 引擎。
# 元数据：模型类型/类别/输入尺寸/生成环境等；附加数据：R-CNN 方案下为 torch 侧
# RPN/RoI 头权重。单文件自包含；旧版无头 payload 自动回退按 YOLO 引擎处理。
_META_MAGIC = b"XCVE"

ENGINE_SUFFIX = ".engine"


def pack_engine_blob(meta: dict, engine: bytes, extra: bytes = b"") -> bytes:
    """把元数据 JSON、附加数据与引擎字节打包为单一 blob。"""
    meta_b = json.dumps(meta, ensure_ascii=False).encode("utf-8")
    return (_META_MAGIC + struct.pack("<II", len(meta_b), len(extra))
            + meta_b + extra + engine)


def unpack_engine_blob(payload: bytes) -> tuple[dict | None, bytes, bytes]:
    """解析引擎 blob：返回 (元数据 or None, 引擎字节, 附加数据)。

    旧版（无 XCVE 魔数）payload 视为纯引擎字节。
    """
    meta = None
    if not payload.startswith(_META_MAGIC):
        return None, payload, b""
    n_meta, n_extra = struct.unpack_from("<II", payload, len(_META_MAGIC))
    off = len(_META_MAGIC) + 8
    try:
        meta = json.loads(payload[off:off + n_meta].decode("utf-8"))
    except Exception:
        log.warning("引擎容器元数据解析失败（按无头引擎处理）", exc_info=True)
    if not isinstance(meta, dict) or "kind" not in meta:
        return None, payload, b""
    off += n_meta
    extra = payload[off:off + n_extra]
    return meta, payload[off + n_extra:], extra

_TRT_DLL_READY = False


def _ensure_trt_dll_search_path() -> None:
    """把 TensorRT 运行库目录加入 DLL 搜索路径（Windows 绑定加载必需）。

    依次尝试：环境变量 TENSORRT_DIR → 运行根 weights\\tensorrt → python
    环境 bin\\（sys.prefix）。只扩充本进程的搜索路径，不修改外部目录。
    """
    global _TRT_DLL_READY
    if _TRT_DLL_READY or not sys.platform.startswith("win"):
        _TRT_DLL_READY = True
        return
    candidates = []
    env_dir = os.environ.get("TENSORRT_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    try:
        from .offline import WEIGHTS_DIR

        candidates.append(Path(WEIGHTS_DIR) / "tensorrt")
    except Exception:
        log.debug("TensorRT 候选目录探测失败")
    candidates.append(Path(sys.prefix) / "bin")
    for d in candidates:
        if (d / "nvinfer_10.dll").is_file():
            os.environ["PATH"] = f"{d}{os.pathsep}{os.environ.get('PATH', '')}"
            try:
                os.add_dll_directory(str(d))
            except Exception:
                log.debug("TensorRT 运行库目录注册失败")
            log.info("TensorRT 运行库目录: %s", d)
            break
    _TRT_DLL_READY = True


def tensorrt_available() -> bool:
    """TensorRT Python 运行库是否可用。"""
    _ensure_trt_dll_search_path()
    try:
        import tensorrt  # noqa: F401

        return True
    except Exception as e:
        log.info("TensorRT 不可用: %s", e)
        return False


def _require_tensorrt():
    if not tensorrt_available():
        raise RuntimeError(
            "未检测到 TensorRT 运行库。请安装与 CUDA 版本匹配的 TensorRT"
            "（pip install tensorrt，或设置 TENSORRT_DIR 指向含 nvinfer_10.dll 的目录）")


def _resolve_gpu_device(device: str) -> str:
    """TensorRT 构建必须在 GPU 上进行：auto 解析为 cuda，否则报错。"""
    resolved = resolve_device(device)
    if resolved != "cuda":
        raise RuntimeError(
            "TensorRT 引擎构建需要 NVIDIA GPU（当前解析设备为 CPU）。"
            "请在带 GPU 的目标部署机上执行模型加速")
    return resolved


def export_engine(src, dst, imgsz: int | tuple[int, int] | None = None,
                  device: str = "auto",
                  half: bool = True, progress=None) -> dict:
    """把权重文件导出为 TensorRT 引擎（.engine，按格式自动选择导出路径）。

    imgsz 支持 640 或 (高, 宽) 矩形（YOLO）；None 时 YOLO 回源权重训练尺寸
    （矩形训练的模型自动按矩形导出）。分类/R-CNN 按训练尺寸自动确定，
    不受此参数影响。实际导出形状会记录进引擎容器头 meta["imgsz"]，加载侧
    据此复原推理几何。
    half=True 构建 FP16 引擎（体积约减半、推理更快，检出分数可能有小数级
    漂移）；False 为 FP32 全精度（精度优先）。
    返回 {"dst": 引擎路径, "seconds": 耗时, "kind": yolo/torch}。

    progress: 可选回调 (frac: float, text: str)。frac∈[0,1] 为阶段进度；
    frac<0 表示该阶段拿不到真实百分比（TRT 构建期），界面切忙碌指示。
    """
    _require_tensorrt()
    gpu = _resolve_gpu_device(device)
    src, dst = Path(src), Path(dst)
    if dst.suffix != ENGINE_SUFFIX:
        dst = dst.with_suffix(ENGINE_SUFFIX)
    t0 = time.perf_counter()
    log.info("TensorRT 导出开始: src=%s (%.1f MB) dst=%s device=%s",
             src, src.stat().st_size / 1e6, dst, gpu)

    _safe_progress(progress, 0.02, "读取并加载源权重…")
    model = load_model(src)  # 统一入口：自动识别格式（含日志）
    _safe_progress(progress, 0.10, "权重已加载，准备导出…")
    from .models.classifier import ClassifierModel
    from .models.rcnn import RCNNModel
    from .models.yolo import YoloModel

    if isinstance(model, YoloModel):
        kind = "yolo"
        _export_yolo_engine(src, dst, imgsz, gpu, half=half,
                            meta={"kind": "yolo", "task": model.task.value,
                                  "classes": model.classes or []},
                            progress=progress)
    elif isinstance(model, ClassifierModel):
        kind = "torch"
        size = model._input_size
        h, w = (size, size) if isinstance(size, int) else (int(size[0]), int(size[1]))
        log.info("分类模型 ONNX 导出: 输入 %dx%d, arch=%s, %s",
                 w, h, model.arch, "FP16" if half else "FP32")
        _torch_to_engine(model.net.cpu().eval(), (1, 3, h, w), dst, half=half,
                         meta={"kind": "classifier", "arch": model.arch,
                               "classes": model.classes, "image_size": size},
                         progress=progress)
    elif isinstance(model, RCNNModel):
        kind = "torch"
        min_size = int(model._min_size)
        max_size = int(min_size * 1333 / 800)
        log.info("R-CNN 特征提取段导出: 输入 %dx%d, arch=%s, %s",
                 max_size, min_size, model.arch, "FP16 半精度" if half else "FP32")
        net = model.net.cpu().eval()

        class _FpnFeatures(torch.nn.Module):
            """引擎段 = 预处理归一化 + Backbone + FPN（纯卷积、静态 shape）。

            RPN/RoI 头含动态 NMS 与 RoiAlign（TRT 无对应原生层/插件），留在
            torch 侧直接消费引擎输出的 FPN 特征（显存零拷贝）。
            """

            def __init__(self, g):
                super().__init__()
                self.g = g

            def forward(self, x):
                images, _ = self.g.transform([x], None)
                return self.g.backbone(images.tensors)

        keys = list(net.backbone(torch.zeros(1, 3, min_size, max_size)).keys())
        # torch 侧检测头权重只打包 RPN + RoI 头键（结构开关一并携带）。
        # 骨干与 FPN 在引擎里已有一份，整网 state_dict 重复封入会让引擎
        # 体积接近"引擎+整模型"之和，比源权重还大
        heads = {k: v for k, v in net.state_dict().items()
                 if k.startswith(("rpn.", "roi_heads."))}
        heads_buf = io.BytesIO()
        torch.save({"format": "xcve-heads-v2", "state_dict": heads,
                    "industrial_opt": model.industrial,
                    "mask_head_light": getattr(model, "mask_head_light", False)},
                   heads_buf)
        _torch_to_engine(_FpnFeatures(net), (3, min_size, max_size), dst,
                         batched=True, half=half,
                         output_names=[f"feat_{k}" for k in keys],
                         meta={"kind": "rcnn", "arch": model.arch,
                               "task": model.task.value, "classes": model.classes,
                               "min_size": min_size, "max_size": max_size,
                               "feat_keys": keys},
                         extra=heads_buf.getvalue(),
                         progress=progress)
    else:
        raise RuntimeError(f"暂不支持该模型家族的 TensorRT 导出: {type(model).__name__}")

    seconds = time.perf_counter() - t0
    log.info("TensorRT 导出完成: %s (%.1f MB, %.1fs)", dst,
             dst.stat().st_size / 1e6, seconds)
    return {"dst": str(dst), "seconds": round(seconds, 1), "kind": kind}


def _write_engine(engine_bytes: bytes, dst: Path,
                  meta: dict | None = None, extra: bytes = b"") -> None:
    """引擎字节（可携带元数据头）→ 明文 .engine 落盘。

    meta 记录生成时显卡算力（cc_build）与 TensorRT 版本（trt_build）：加载侧
    据此在反序列化前预检跨显卡不匹配并给出指引。记录失败不阻塞导出
    （best-effort）。
    """
    if meta is not None:
        meta = dict(meta)
        try:
            from .trt_runtime import local_compute_capability

            cc = local_compute_capability()
            if cc:
                meta.setdefault("cc_build", [cc[0], cc[1]])
            # DLL 路径先布置再 import：裸 import 在 DLL 不在系统 PATH 的
            # 机器上会 FileNotFoundError，导致 trt_build 静默缺失
            _ensure_trt_dll_search_path()
            import tensorrt as _trt

            meta.setdefault("trt_build", str(_trt.__version__))
        except Exception:
            log.warning("TensorRT 元数据记录失败（trt_build 缺失）", exc_info=True)
        engine_bytes = pack_engine_blob(meta, engine_bytes, extra)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(engine_bytes)


def _safe_progress(progress, frac: float, text: str) -> None:
    """进度回调安全包装：回调失败绝不阻塞导出主流程。"""
    if progress is not None:
        try:
            progress(frac, text)
        except Exception:  # noqa: BLE001
            log.debug("导出进度回调失败")


def _engine_input_hw(engine_bytes: bytes) -> list[int] | None:
    """读 TensorRT 引擎的真实输入 H/W（导出请求的 imgsz 可能被 stride 取整）。

    ultralytics 导出的 .engine 文件头部是 [4 字节元数据长度][JSON 元数据]，
    跳过该头后才是可反序列化的 plan。反序列化只读 shape，秒级完成；
    失败返回 None（best-effort，退回请求尺寸）。
    """
    try:
        import json as _json

        import tensorrt as trt

        blob = engine_bytes
        try:
            meta_len = int.from_bytes(blob[:4], byteorder="little")
            _json.loads(blob[4:4 + meta_len].decode("utf-8"))
            blob = blob[4 + meta_len:]  # 跳过 ultralytics 元数据头
        except Exception:  # noqa: BLE001  无头引擎：整段交给 runtime
            log.debug("无元数据头引擎（整段交运行时）")
        rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        eng = rt.deserialize_cuda_engine(blob)
        try:
            for i in range(eng.num_io_tensors):
                name = eng.get_tensor_name(i)
                if eng.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    shape = eng.get_tensor_shape(name)
                    if len(shape) == 4:
                        return [int(shape[2]), int(shape[3])]
        finally:
            del eng
    except Exception:  # noqa: BLE001
        return None
    return None


def _export_yolo_engine(src: Path, dst: Path, imgsz, device: str, half: bool = False,
                        meta: dict | None = None, progress=None):
    """YOLO：ultralytics 原生 engine 导出。

    imgsz 支持 640 或 (高, 宽) 矩形；None 时回源权重训练尺寸（矩形训练模型
    自动按矩形导出）。引擎真实输入形状写进 meta["imgsz"]。
    """
    from ultralytics import YOLO

    _safe_progress(progress, -1, "YOLO 引擎构建中（ultralytics，1~10 分钟）…")
    y = YOLO(str(src))
    imgsz = _resolve_yolo_imgsz(imgsz, y)
    out = y.export(format="engine", imgsz=imgsz, device=device,
                   simplify=True, half=half, verbose=False, workspace=4)
    engine = Path(out)
    _safe_progress(progress, 0.90, "构建完成，保存引擎…")
    try:
        if meta is not None:
            hw = _engine_input_hw(engine.read_bytes())
            if hw is None:  # 反序列化读不到形状：退回请求尺寸（已对齐 stride）
                hw = ([imgsz, imgsz] if isinstance(imgsz, int)
                      else [int(imgsz[0]), int(imgsz[1])])
            meta["imgsz"] = hw
        _write_engine(engine.read_bytes(), dst, meta=meta)
    finally:
        engine.unlink(missing_ok=True)  # 清理 ultralytics 构建中间产物


def _resolve_yolo_imgsz(imgsz, y) -> "int | list[int]":
    """导出尺寸归一：显式参数优先；缺省回源权重训练尺寸（矩形保留）。"""
    if imgsz is not None:
        return imgsz
    v = (getattr(y, "overrides", None) or {}).get("imgsz")
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return [int(v[0]), int(v[1])]
    if isinstance(v, (int, float)) and int(v) >= 32:
        return int(v)
    return 640


def _torch_to_engine(net: torch.nn.Module, input_shape: tuple, dst: Path,
                     batched: bool = True, meta: dict | None = None,
                     output_names: list[str] | None = None, extra: bytes = b"",
                     half: bool = False, progress=None):
    """torch 模块 → 内存 ONNX → TensorRT 构建（ONNX 中间产物不落盘）。

    half=True 时引擎内部以 FP16 精度执行（I/O 保持 FP32），引擎体积约减半、
    推理在 RT 系显卡上更快；仅用于体积/耗时敏感的特征段加速。
    """
    dummy = torch.randn(*input_shape)
    args = (dummy,) if batched else ([dummy],)
    names = ["images", "boxes", "labels", "scores", "masks"] if not batched \
        else ["images", "outputs"]
    # 工业优化 DCN 无 ONNX 算子：导出期间切换为等价的 GridSample 实现
    from .models.modules import DCNv2, set_dcn_export_mode

    _safe_progress(progress, 0.18, "导出 ONNX…")
    set_dcn_export_mode(True)
    buf = io.BytesIO()
    try:
        torch.onnx.export(
            net, args, buf, opset_version=17,
            input_names=names[:1],
            output_names=output_names or names[1:],
            do_constant_folding=True,
        )
    finally:
        set_dcn_export_mode(False)
    log.info("ONNX 序列化完成（内存）: %.1f MB", buf.getbuffer().nbytes / 1e6)
    _safe_progress(progress, 0.25, "开始构建 TensorRT 引擎…")
    _build_engine_from_onnx(buf.getvalue(), dst, meta=meta, extra=extra, half=half,
                            progress=progress)


def _build_engine_from_onnx(onnx_bytes: bytes, dst: Path, meta: dict | None = None,
                            extra: bytes = b"", half: bool = False, progress=None):
    """TensorRT 从内存 ONNX 字节构建引擎 → 明文 .engine 落盘（含元数据头）。"""
    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)
    _safe_progress(progress, 0.30, "解析 ONNX…")
    if not parser.parse(onnx_bytes):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        log.error("ONNX 解析失败: %s", "; ".join(errs[:8]))
        raise RuntimeError("TensorRT 解析 ONNX 失败：\n" + "\n".join(errs[:8]))
    log.info("TensorRT ONNX 解析成功，开始构建引擎（视模型规模约需 1~10 分钟）…")

    config = builder.create_builder_config()
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    except AttributeError:  # 老 TensorRT API 回退
        config.max_workspace_size = 4 << 30
    try:  # 禁用 TF32：TF32 卷积的数值偏差经深层传播会被放大，影响与
        # torch 权重推理的一致性（检出分数在阈值边缘会翻转）
        config.clear_flag(trt.BuilderFlag.TF32)
    except Exception:
        log.debug("TF32 标志清除失败（保持默认）")
    if half:
        # FP16 半精度：引擎内部卷积/矩阵乘以 FP16 执行（I/O 仍为 ONNX 的
        # FP32 dtype），权重体积约减半；不支持 FP16 的层自动回退 FP32
        config.set_flag(trt.BuilderFlag.FP16)
    _safe_progress(progress, -1,
                   "TensorRT 引擎构建中（1~10 分钟，取决于模型规模）…")
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT 引擎构建失败（详见日志输出）")
    _safe_progress(progress, 0.92, "构建完成，保存引擎…")
    _write_engine(bytes(engine), dst, meta=meta, extra=extra)

