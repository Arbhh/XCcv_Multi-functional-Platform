# -*- coding: utf-8 -*-
"""SAM / DINO / DINOv2 / R-CNN 骨干权重的运行时接管下载（与 CLI 共享层）。

需求：训练/推理/标注启动时探测到预训练权重缺失即自动经统一下载器补齐——
notify(text, frac) 进度、多源回退、连接/读取超时与单源耗时上限（复用
ensure_weight 的 fetcher）、大小下限与容器头校验、原子替换、全失败抛
RuntimeError（中文原因），不再静默失败或依赖第三方库自行弹窗/无超时下载；
与 tools/download_weights.py CLI 共用同一注册表与落位/校验口径
（CLI --list 清单 = 运行时探测集合）。

实现：WeightEntry 注册表（family/key/落点/URL 源/大小下限/容器头校验/
许可声明，url 可经 torchvision 官方权重枚举惰性解析）；下载复用 ensure_weight
的 fetcher 与镜像候选；DINOv2 走 huggingface_hub 快照下载（保留 HF_ENDPOINT
镜像口径，环境未显式设置时缺省 hf-mirror）。

落位口径（与 CLI / 各加载点同源）：全部统一在引擎权重目录 offline.WEIGHTS_DIR
（开发态=xccv/engine/weights，安装态=<安装根>/weights）下做族细分——
  WEIGHTS_DIR/sam/                      ai_assist 的 SAM 档位权重；
  WEIGHTS_DIR/dino/                     dino_vit 的 DINO ViT-B/8；
  WEIGHTS_DIR/dinov2/<模型目录>/        subspacead 的 DINOv2（HF 快照）；
  WEIGHTS_DIR/ultralytics/              YOLO 预训练（ensure_weight 承载）；
  WEIGHTS_DIR/torch/hub/checkpoints/    torchvision 骨干（offline.py 的
                                        torch_dir/TORCH_HOME 重定向口径）。
禁止向仓库根 weights/ 或其他位置落任何下载物。

使用：模型族加载点最小挂钩——ai_assist._engine_by_name（SAM 档位权重）、
dino_vit.find_dino_checkpoint（DINO ViT-B/8）、subspacead._find_backbone
（DINOv2）、rcnn._build（R-CNN ImageNet 骨干）；挂钩均 try/except 包裹，
失败回退原缺失指引路径，不阻塞建模。CLI 经 entries_for(family) 取同一
注册表做 --list 与按族下载。
"""

import importlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from ..offline import WEIGHTS_DIR
from .ensure_weight import _default_fetcher, _url_candidates

log = logging.getLogger(__name__)

# 统一落位（引擎权重目录下的族子目录；唯一写入根，禁止仓库根 weights/）。
# 全部由 WEIGHTS_DIR 动态派生（单一来源；测试经替换模块级 WEIGHTS_DIR 隔离）。
def sam_dir() -> Path:
    """SAM 档位权重落点（ai_assist 的发现目录）。"""
    return WEIGHTS_DIR / "sam"


def dino_dir() -> Path:
    """DINO ViT-B/8 落点（dino_vit 的发现目录）。"""
    return WEIGHTS_DIR / "dino"


def dinov2_root() -> Path:
    """DINOv2 本地 HF 模型目录根（subspacead 的扫描目录）。"""
    return WEIGHTS_DIR / "dinov2"


def tv_hub_dir() -> Path:
    """torchvision 骨干权重落点（offline.py torch_dir 口径下的 hub 缓存）。"""
    return WEIGHTS_DIR / "torch" / "hub" / "checkpoints"

DINO_VITB8_FILE = "dino_vitbase8_pretrain.pth"
_DINO_VITB8_URL = ("https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/"
                   "dino_vitbase8_pretrain.pth")
DINOV2_REPO_ID = "facebook/dinov2-with-registers-small"
HF_MIRROR_DEFAULT = "https://hf-mirror.com"
_SAM_ONNX_BASE = ("https://github.com/labelmeai/efficient-sam/releases/"
                  "download/onnx-models-20231225/")

SAM_PT_MIN = 30 * 1024 * 1024          # 最小的 mobile_sam 约 40 MB
SAM_ONNX_MIN = 5 * 1024 * 1024         # decoder 约 16 MB，encoder 约 85 MB
DINO_PTH_MIN = 100 * 1024 * 1024       # ViT-B/8 约 343 MB
TV_PTH_MIN = 1 * 1024 * 1024
HF_WEIGHT_MIN = 10 * 1024 * 1024       # DINOv2 safetensors >80 MB

SAM_LICENSE = ("SAM/SAM2/SAM-B/MobileSAM 权重经 ultralytics 资产库重发布"
               "（ultralytics 训练链路为 AGPL-3.0，权重许可以 Meta 与原项目"
               "官方发布页为准）；EfficientSAM onnx 来自 labelmeai/"
               "efficient-sam 官方发布（Apache-2.0，源自 yformer/"
               "EfficientSAM）；本仓库不随包分发。")
DINO_LICENSE = ("DINO ViT-B/8 来自 facebookresearch/dino 官方发布"
                "（Apache-2.0）；DINOv2 骨干为 HuggingFace "
                "facebook/dinov2-with-registers-small（代码 Apache-2.0，"
                "权重许可以其官方发布页为准）；本仓库不随包分发。")
RCNN_LICENSE = ("torchvision ImageNet/COCO 初始权重随 torchvision 官方发布"
                "（BSD-3）；maskrcnn18 slim 初始权重不下载，用 "
                "tools/build_maskrcnn18_init.py 自建。")


def hub_checkpoints_dir() -> Path:
    """torchvision 骨干权重落点（offline.py torch_dir 口径下的 hub 缓存）。

    offline.setup_offline_weights 已把 TORCH_HOME 重定向到
    WEIGHTS_DIR/torch，torchvision 运行时解析与这里一致，预取文件直接命中。
    """
    return tv_hub_dir()


def _head_ok(path: Path) -> bool:
    """torch 容器头校验（与 ensure_weight 同口径：zip 容器 PK / 旧版 \x80）。"""
    try:
        with open(path, "rb") as fh:
            head = fh.read(4)
    except OSError:
        return False
    return head[:2] == b"PK" or head[:1] == b"\x80"


def _entry_ok(path: Path, min_bytes: int, check_head: bool) -> bool:
    """已存在文件是否可用：大小下限（+容器头）。"""
    try:
        if not path.is_file() or path.stat().st_size < min_bytes:
            return False
    except OSError:
        return False
    return (not check_head) or _head_ok(path)


def hf_local_dir() -> Path:
    """DINOv2 HF 快照的默认下载落点（WEIGHTS_DIR/dinov2/<repo id>/）。"""
    return dinov2_root() / DINOV2_REPO_ID


def dinov2_discovered_dir() -> Path | None:
    """按 subspacead._find_backbone 的发现口径扫描 WEIGHTS_DIR/dinov2/。

    支持两种放置形态：子目录即模型目录（任意命名，如缓存改名），或
    子目录为组织名、其下才是模型目录（接管下载落点 facebook/<model>）；
    只要 config.json + 大权重齐备即认为就绪，避免重复下载到另一目录。
    """
    root = dinov2_root()
    if not root.is_dir():
        return None
    cands = []
    if (root / "config.json").is_file():
        cands.append(root)
    try:
        subs = sorted(root.iterdir())
    except OSError:
        subs = []
    for sub in subs:
        if not sub.is_dir():
            continue
        cands.append(sub)
        try:
            cands.extend(p for p in sorted(sub.iterdir())
                         if p.is_dir() and (p / "config.json").is_file())
        except OSError:
            pass
    for c in cands:
        if _hf_ready(c):
            return c
    return None


def _hf_ready(d: Path) -> bool:
    """HF 快照是否可用：config.json + 大权重文件齐备。"""
    if not (d / "config.json").is_file():
        return False
    for f in ("model.safetensors", "pytorch_model.bin",
              "model.safetensors.index.json"):
        try:
            if (d / f).is_file() and (d / f).stat().st_size > HF_WEIGHT_MIN:
                return True
        except OSError:
            continue
    return False


@dataclass(frozen=True)
class WeightEntry:
    """单一权重条目：落点 + 下载源 + 校验口径（CLI 与运行时共享）。"""

    family: str
    key: str                     # 文件名（url_provider 条目可空，解析后推导）
    label: str = ""
    subpath: str = ""            # WEIGHTS_DIR 下的族子目录
    urls: tuple = ()             # 静态多源（官方在前，镜像在后）
    url_provider: str = ""       # "torchvision.models.Xxx_Weights.YYY" 惰性解析
    min_bytes: int = TV_PTH_MIN
    check_head: bool = True
    kind: str = "url"            # url / hf
    license: str = ""

    @property
    def resolved_urls(self) -> tuple:
        """下载源：静态 urls 或经 torchvision 枚举解析（失败返回空并记日志）。

        url_provider 形如 "torchvision.models.ResNet18_Weights.IMAGENET1K_V1"：
        按"最长可导入模块前缀 + 其余逐级 getattr"解析，兼容枚举多级属性。
        """
        if not self.url_provider:
            return self.urls
        parts = self.url_provider.split(".")
        for cut in range(len(parts) - 1, 0, -1):
            try:
                mod = importlib.import_module(".".join(parts[:cut]))
            except ImportError:
                continue
            try:
                obj = mod
                for a in parts[cut:]:
                    obj = getattr(obj, a)
                return (obj.url,)
            except AttributeError:
                break
        log.warning("权重 URL 解析失败（%s）", self.url_provider)
        return ()

    @property
    def dest(self) -> Path:
        """落点；url_provider 条目的文件名经解析推导（未解析时抛 RuntimeError）。"""
        if self.kind == "hf":
            return hf_local_dir()
        key = self.key
        if not key:
            urls = self.resolved_urls
            if not urls:
                raise RuntimeError(
                    f"权重文件名待解析（{self.label}：torchvision 枚举不可用）")
            key = urls[0].rsplit("/", 1)[-1]
        return WEIGHTS_DIR / key if not self.subpath \
            else WEIGHTS_DIR / self.subpath / key

    def is_ready(self) -> bool:
        """已就绪判定（与下载校验同一口径）。"""
        if self.kind == "hf":
            return dinov2_discovered_dir() is not None
        try:
            return _entry_ok(self.dest, self.min_bytes, self.check_head)
        except RuntimeError:
            return False


def sam_entries() -> list:
    """SAM 族条目（AI 智能标注，落 WEIGHTS_DIR/sam/，与 ai_assist 对齐）。"""
    pts = (("sam2.1_t.pt", "SAM2.1-T：均衡（默认档）"),
           ("sam2.1_s.pt", "SAM2.1-S：精度更高"),
           ("sam2.1_b.pt", "SAM2.1-B+：高精度（建议 GPU）"),
           ("sam2.1_l.pt", "SAM2.1-L：最高精度（需大显存）"),
           ("mobile_sam.pt", "MobileSAM：轻量快速"),
           ("sam_b.pt", "SAM ViT-B：经典版"))
    # .pt 权重复用 ensure_weight 的多源候选（官方资产 + 镜像 + 环境变量覆盖）
    out = [WeightEntry("sam", n, lbl, subpath="sam",
                       urls=tuple(_url_candidates(n)),
                       min_bytes=SAM_PT_MIN, license=SAM_LICENSE)
           for n, lbl in pts]
    for n in ("efficient_sam_vits_encoder.onnx",
              "efficient_sam_vits_decoder.onnx"):
        out.append(WeightEntry(
            "sam", n, "EfficientSAM onnx（CPU 快速）", subpath="sam",
            urls=(_SAM_ONNX_BASE + n,), min_bytes=SAM_ONNX_MIN,
            check_head=False, license=SAM_LICENSE))
    return out


def dino_entries() -> list:
    """DINO 族条目（ViT-B/8 落 WEIGHTS_DIR/dino；DINOv2 为 HF 快照条目）。"""
    return [
        WeightEntry("dino", DINO_VITB8_FILE, "DINO ViT-B/8（CIF 特征底座）",
                    subpath="dino", urls=(_DINO_VITB8_URL,),
                    min_bytes=DINO_PTH_MIN, license=DINO_LICENSE),
        WeightEntry("dino", DINOV2_REPO_ID,
                    "DINOv2（SubspaceAD 骨干，HF 快照）", kind="hf",
                    license=DINO_LICENSE),
    ]


def rcnn_entries() -> list:
    """R-CNN / 分类骨干族条目（torchvision 官方枚举解析，落 WEIGHTS_DIR/torch）。

    条目与各版本 torchvision 实际存在的权重枚举一一对应（ResNet18 官方
    只有 V1；ResNet50 有 V1/V2），避免死条目在 --list 里常驻告警。
    """
    enums = (
        ("torchvision.models.ResNet18_Weights.IMAGENET1K_V1",
         "resnet18 ImageNet V1"),
        ("torchvision.models.ResNet50_Weights.IMAGENET1K_V1",
         "resnet50 ImageNet V1"),
        ("torchvision.models.ResNet50_Weights.IMAGENET1K_V2",
         "resnet50 ImageNet V2"),
        ("torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1",
         "mobilenet_v3_small ImageNet V1"),
    )
    return [WeightEntry("rcnn", "", lbl, subpath="torch/hub/checkpoints",
                        url_provider=ep,
                        min_bytes=TV_PTH_MIN, license=RCNN_LICENSE)
            for ep, lbl in enums]


_FAMILY_FACTORIES = {"sam": sam_entries, "dino": dino_entries,
                     "rcnn": rcnn_entries}


def entries_for(family: str) -> list:
    """按族取注册表条目（yolo 族由 ensure_weight 单独承载，不在此列）。"""
    factory = _FAMILY_FACTORIES.get(family)
    return factory() if factory else []


def download_named(name: str, dest: Path, urls: tuple, min_bytes: int,
                   check_head: bool, notify=None, fetcher=None) -> str:
    """通用多源下载：已就绪跳过 / .part 清理 / 全失败 RuntimeError（中文）。

    fetcher 可注入（测试用本地写入替代网络下载），缺省复用 ensure_weight
    的默认下载器（连接/读取超时 + 单源耗时上限）；进度经 notify(text, frac)
    上报（首次 frac=None）。
    """
    if _entry_ok(dest, min_bytes, check_head):
        return "已就绪（校验通过，跳过）"
    log.info("权重缺失，经统一下载器补齐：%s -> %s", name, dest)
    if dest.exists():
        log.info("发现不完整/校验未通过的旧文件，将重新下载：%s", dest.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if fetcher is None:
        fetcher = _default_fetcher
    errors = []
    for idx, url in enumerate(urls, start=1):
        tmp = dest.with_name(dest.name + f".part{idx}")
        if notify:
            notify(f"正在下载 {name}（下载源 {idx}/{len(urls)}）…", None)

        def _prog(frac, _i=idx):
            if notify:
                notify(f"正在下载 {name}（下载源 {_i}/{len(urls)}）", frac)

        try:
            fetcher(url, tmp, _prog)
            if not _entry_ok(tmp, min_bytes, check_head):
                raise ValueError("下载内容不完整或容器头无效")
            tmp.replace(dest)
            if notify:
                notify(f"就绪 {name}", 1.0)
            return "新下载完成"
        except Exception as exc:  # noqa: BLE001（逐源记录，最后统一上报）
            errors.append(f"源{idx} {type(exc).__name__}: {str(exc)[:160]}")
            log.warning("权重下载失败（%s/%s，%s）：%s", idx, len(urls), url, exc)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            if notify and idx < len(urls):
                notify(f"下载源 {idx} 失败（{type(exc).__name__}），"
                       "正在切换下一个下载源…", None)
    raise RuntimeError(f"{name} 下载失败（已尝试 {len(urls)} 个下载源）。"
                       "请检查网络或代理后重试，也可手动下载放入："
                       f"{dest.parent}。详情：{'；'.join(errors[-2:]) or '未知错误'}")


def download_entry(entry: WeightEntry, notify=None, fetcher=None) -> Path:
    """确保单个条目就绪并返回落点（kind=hf 走快照下载路径）。"""
    if entry.kind == "hf":
        got = ensure_dinov2(notify)
        if got is None:
            raise RuntimeError(
                f"{DINOV2_REPO_ID} 下载失败（HuggingFace 快照；可设 "
                "HF_ENDPOINT 更换镜像或手动下载模型目录放入 "
                f"{hf_local_dir()}）")
        return got
    download_named(entry.key or entry.label, entry.dest, entry.resolved_urls,
                   entry.min_bytes, entry.check_head, notify, fetcher)
    return entry.dest


def ensure_sam_weight(filename: str, notify=None, fetcher=None) -> Path:
    """按文件名补齐 SAM 档位权重，返回落点（未知文件名抛 RuntimeError）。"""
    for e in sam_entries():
        if e.key == filename:
            download_named(e.key, e.dest, e.urls, e.min_bytes, e.check_head,
                           notify, fetcher)
            return e.dest
    raise RuntimeError(f"未知 SAM 权重文件：{filename}")


def ensure_sam_missing_files(filenames, notify=None, fetcher=None) -> int:
    """ai_assist 挂钩入口：逐个补齐缺失文件；失败仅记日志不抛（不阻塞其余档位）。"""
    done = 0
    for f in filenames:
        try:
            ensure_sam_weight(f, notify, fetcher)
            done += 1
        except Exception:  # noqa: BLE001（接管失败回退原缺失指引）
            log.warning("SAM 权重 %s 运行时补齐失败", f, exc_info=True)
    return done


def ensure_dino_vitb8(notify=None, fetcher=None) -> Path | None:
    """补齐 DINO ViT-B/8 到 WEIGHTS_DIR/dino；失败返回 None（回退原缺失指引）。"""
    e = dino_entries()[0]
    try:
        download_named(e.key, e.dest, e.urls, e.min_bytes, e.check_head,
                       notify, fetcher)
        return e.dest
    except Exception:  # noqa: BLE001（接管失败回退原缺失指引）
        log.warning("DINO ViT-B/8 运行时补齐失败", exc_info=True)
        return None


def ensure_dinov2(notify=None) -> Path | None:
    """补齐 DINOv2 HF 快照到 WEIGHTS_DIR/dinov2；失败返回 None。

    镜像口径：保留 subspacead 的 HF_ENDPOINT 机制——环境未显式设置时缺省
    走 hf-mirror（首次 import huggingface_hub 前生效），已设置则以用户为准。
    本地已就绪（发现口径命中）直接返回，不联网。
    """
    found = dinov2_discovered_dir()
    if found is not None:
        return found
    dest = hf_local_dir()
    log.info("DINOv2 骨干缺失，经统一下载器补齐（HF 快照）：%s", dest)
    os.environ.setdefault("HF_ENDPOINT", HF_MIRROR_DEFAULT)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        log.warning("huggingface_hub 缺失，DINOv2 无法在线补齐"
                    "（pip install huggingface_hub 后重试）")
        return None
    if notify:
        notify(f"正在下载 {DINOV2_REPO_ID}（HuggingFace 快照）…", None)
    try:
        snapshot_download(repo_id=DINOV2_REPO_ID, local_dir=str(dest),
                          allow_patterns=["config.json",
                                          "preprocessor_config.json",
                                          "model.safetensors"])
    except Exception:  # noqa: BLE001（接管失败回退原 hub id 路径）
        log.warning("DINOv2 快照下载失败（可设 HF_ENDPOINT 更换镜像）",
                    exc_info=True)
        return None
    if notify:
        notify(f"就绪 {DINOV2_REPO_ID}", 1.0)
    return dest if _hf_ready(dest) else None


def _rcnn50_backbone_url() -> str:
    """按当前 torchvision 检测模型实际默认 backbone 枚举解析 URL（随版本适配）。"""
    try:
        import inspect

        from torchvision.models.detection import maskrcnn_resnet50_fpn
        d = inspect.signature(maskrcnn_resnet50_fpn).parameters[
            "weights_backbone"].default
        if d is not None and not isinstance(d, str):
            return d.url
    except Exception:  # noqa: BLE001（签名形态随版本变化，统一退回 V1）
        log.warning("torchvision 检测模型默认骨干解析失败，退回 ResNet50 V1",
                    exc_info=True)
    from torchvision.models import ResNet50_Weights

    return ResNet50_Weights.IMAGENET1K_V1.url


def ensure_rcnn_backbones(arch: str, slim_head: bool = False,
                          use_pretrained: bool = True, notify=None,
                          fetcher=None) -> None:
    """rcnn._build 挂钩：按架构预取 ImageNet 骨干初始权重。

    fasterrcnn/maskrcnn → torchvision 检测模型默认 backbone 枚举；
    maskrcnn18 → ResNet18 IMAGENET1K_V1。失败仅记日志（回退 torchvision
    自带下载路径，不阻塞建模）；slim 自建初始权重（maskrcnn18_slim_init.pt）
    不在线下载。
    """
    if not use_pretrained:
        return
    try:
        if arch in ("fasterrcnn", "maskrcnn"):
            targets = [("resnet50 骨干（检测模型默认）", _rcnn50_backbone_url())]
        elif arch == "maskrcnn18":
            from torchvision.models import ResNet18_Weights

            targets = [("resnet18 ImageNet V1",
                        ResNet18_Weights.IMAGENET1K_V1.url)]
        else:
            return
    except ImportError:
        log.warning("torchvision 缺失，跳过 R-CNN 骨干权重预取")
        return
    for label, url in targets:
        fname = url.rsplit("/", 1)[-1]
        e = WeightEntry("rcnn", fname, label,
                        subpath="torch/hub/checkpoints", urls=(url,),
                        min_bytes=TV_PTH_MIN, license=RCNN_LICENSE)
        try:
            download_entry(e, notify, fetcher)
        except Exception:  # noqa: BLE001（接管失败回退 torchvision 自带下载）
            log.warning("R-CNN 骨干权重 %s 预取失败（回退 torchvision 自带下载）",
                        fname, exc_info=True)
