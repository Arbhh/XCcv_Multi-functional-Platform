"""SubspaceAD 无监督少样本异常检测后端（训练免梯度，PCA 子空间）。

来源：SubspaceAD（CVPR 2026，arXiv 2602.23013）Training-Free Few-Shot
Anomaly Detection via Subspace Modeling；平台按论文口径复现并适配本平台
数据集与推理契约。k-shot 可调但设上限 12，避免脱离免训练范畴；支持集
张数/增强样本数替代轮次类参数，与 UI 适配同步。

依赖：transformers（DINOv2-with-registers 特征，已随平台环境安装，不新增
安装）；PCA 为纯 torch 两遍流式 fp64 实现（与参考实现 PCAModel 同式，不
依赖 sklearn）；增强走 torchvision T（缺失回退 cv2 旋转）。

管线（论文主表口径：无显著性掩码/docrop/CLAHE）：
1. k-shot 采样：OK 池按种子随机取 k 张（k∈[1,12]，不足取全部）；
2. 增强：每张支持图 aug_count 张随机旋转（默认 30，0=关闭；旋转增强
   与参考实现同族）；
3. 特征：DINOv2 patch token，末段层 mean 聚合（骨干深度 ≥18 时取
   hidden states -12..-18 与论文一致；浅骨干退末 7 层）；
4. PCA：两遍流式（均值→协方差→eigh）fp64，按 pca_ev=0.99 选主成分数；
5. 打分：重构残差 Σ(x−(Z·Cᵀ+μ))² → patch 网格上采样 → 高斯平滑 → 图像分 = top1% 均值（mtop1p）；
6. z 标定（与 patchcore/CIF 同链）：OK 图像分 mu/sigma + 像素图
   mu/sigma；conf→z=1/conf−1；连通域出框/掩码 + 图像级 z 分数。

权重红线（"用户自备"，与 dino_vit.py 同惯例）：DINOv2 骨干不自动下载
大权重、不随包分发（只记指纹）。发现顺序：
①环境变量 XCCV_DINOV2_CKPT（本地 HF 模型目录，含 config.json）；
②<app_root>/weights/dinov2/ 同结构目录；③HF 本地缓存（优先大模型）；
④默认 hub id facebook/dinov2-with-registers-small（需要网络或
HF_ENDPOINT=https://hf-mirror.com 镜像）。
TensorRT .engine：PCA 子空间后端不支持（与 patchcore/CIF 同口径）。
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..dataset.base import TaskType
from ..dataset.reader import is_negative_dir_name
from ..io_utils import imread_unicode
from .anomaly import _z_from_conf, image_ap, image_auroc
from .base import BaseModel, Prediction, TrainConfig, TrainReporter

log = logging.getLogger(__name__)

DEFAULT_CKPT_ID = "facebook/dinov2-with-registers-small"
_REGISTER_FAMILY = ("giant", "large", "base", "small")   # 缓存择优顺序
_PCA_EV = 0.99              # 主成分方差覆盖率（论文 Table 4 最优档）
_MAX_LAYER_SPAN = 7         # 末段聚合层数（论文 -12..-18 共 7 层）
_IMG_AGG_TOP_FRAC = 0.01    # 图像分 = top1% 均值（参考实现 mtop1p）
_CALIB_CAP = 16             # 标定图上限（支持图+增强子集）
_VAL_CAP = 64               # 验证 NG/OK 各自上限（训练时长护栏）
_EXTRACT_BATCH = 4


def _feature_dim(config) -> int:
    return int(getattr(config, "hidden_size", 0) or 0)


def _layer_indices(depth: int) -> list[int]:
    """末段聚合层（负索引）：深度 ≥18 取 -12..-18（论文口径）；浅骨干退末 7 层。"""
    if depth >= 18:
        return list(range(-12, -18 - 1, -1))
    return list(range(-1, -min(_MAX_LAYER_SPAN, depth) - 1, -1))


def _find_backbone() -> tuple[str, str]:
    """定位 DINOv2 骨干 → (from_pretrained 参数, 指纹 hex16)。缺失抛
    FileNotFoundError（创建模型时即提示，训练前不空转）。"""
    env = os.environ.get("XCCV_DINOV2_CKPT", "").strip()
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    from ...app_paths import app_root

    local_root = app_root() / "weights" / "dinov2"
    if local_root.is_dir():
        if (local_root / "config.json").is_file():
            candidates.append(local_root)
        for sub in sorted(local_root.iterdir()):
            if sub.is_dir() and (sub / "config.json").is_file():
                candidates.append(sub)
    # HF 本地缓存（hub/models--facebook--dinov2-with-registers-<size>）
    hf_hub = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_hub.is_dir():
        for size in _REGISTER_FAMILY:
            d = hf_hub / f"models--facebook--dinov2-with-registers-{size}"
            if d.is_dir():
                candidates.append(d)
    for c in candidates:
        # hub 缓存目录解析最新快照；本地目录直接用
        snap = c
        if c.name.startswith("models--"):
            refs = c / "refs" / "main"
            try:
                commit = refs.read_text(encoding="utf-8").strip()
                snap = c / "snapshots" / commit
            except OSError:
                snaps = sorted((c / "snapshots").glob("*")) if (c / "snapshots").is_dir() else []
                snap = snaps[-1] if snaps else c
        if (snap / "config.json").is_file():
            has_weights = any(
                (snap / f).is_file() and (snap / f).stat().st_size > 1e6
                for f in ("model.safetensors", "pytorch_model.bin",
                          "model.safetensors.index.json"))
            digest_src = (snap / "config.json").read_bytes()
            if has_weights:
                for wf in sorted(snap.glob("*.safetensors")) + sorted(snap.glob("*.bin")):
                    digest_src += wf.name.encode()
                    digest_src += str(wf.stat().st_size).encode()
            digest = hashlib.sha256(digest_src).hexdigest()[:16]
            return str(snap), digest
    # 兜底：默认 hub id（需要网络；本机直连被墙时用 HF_ENDPOINT 镜像）
    return DEFAULT_CKPT_ID, hashlib.sha256(
        DEFAULT_CKPT_ID.encode()).hexdigest()[:16]


def _augment_fn(aug_count: int):
    """k-shot 增强工厂（旋转增强与参考实现同族）。

    返回 fn(pil_img) -> pil_img；aug_count=0 返回 None。torchvision 缺失
    回退 cv2 随机旋转（边界复制填充）。随机性由 BaseModel.train 的全局
    种子驱动（与参考实现 --seed 语义一致）。
    """
    if aug_count <= 0:
        return None
    try:
        from torchvision import transforms as T

        rot = T.RandomRotation(degrees=(0, 345))
        return rot
    except ImportError:
        def _cv2_rot(pil_img):
            import random as _r

            angle = _r.uniform(0.0, 345.0)
            h, w = pil_img.size[1], pil_img.size[0]
            m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            arr = cv2.warpAffine(np.array(pil_img)[:, :, ::-1], m, (w, h),
                                 borderMode=cv2.BORDER_REPLICATE)
            from PIL import Image

            return Image.fromarray(arr[:, :, ::-1])
        return _cv2_rot


def _topk_mean(arr: np.ndarray, frac: float = _IMG_AGG_TOP_FRAC) -> float:
    """图像分聚合（参考实现 mtop1p 同式：top k 均值，k=1% 像素数）。"""
    flat = arr.ravel()
    k = max(1, int(len(flat) * frac))
    idx = np.argpartition(flat, -k)[-k:]
    return float(np.mean(flat[idx]))


def _post_process(anomaly_map: np.ndarray, res: int) -> np.ndarray:
    """参考实现 post_process_map 同式：上采样到 res → 高斯平滑（k=3, σ=4）。"""
    if anomaly_map.dtype != np.float32:
        anomaly_map = anomaly_map.astype(np.float32)
    out = cv2.resize(anomaly_map, (res, res), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(out, (3, 3), 4.0)


class SubspaceAdModel(BaseModel):
    """SubspaceAD：DINOv2 子空间建模 + 重构残差打分（免训练，OK 图）。"""

    name = "subspacead"
    family = "folder"
    supported_tasks = (TaskType.ANOMALY,)
    default_image_size = 448                     # 14 的整倍数（patch 14）

    def __init__(self, num_classes: int = 1, device: str = "auto",
                 use_pretrained: bool = True):
        super().__init__(num_classes, device)
        self.task = TaskType.ANOMALY
        # 图像分 = OK 分布标定 z（无上界），conf→z=1/conf−1
        self.score_semantics = {
            "key": "z", "name": "图像分数", "range": [0.0, None],
            "higher": "分数越高越异常（OK 分布 z 标定）",
        }
        self.train_mode = "free"                 # 免训练范式声明
        self.image_size = self.default_image_size
        self.params = {
            "k_shot": 4, "aug_count": 30, "ev": _PCA_EV,
            "layers": "auto", "res": self.default_image_size,
            "agg": "mean", "score": "reconstruction", "img_agg": "mtop1p",
        }
        self.classes = ["异常"]
        self._bb = None                          # DINOv2（惰性加载）
        self._proc = None
        self._bb_id: str | None = None
        self._bb_sha: str = ""
        self._bb_loaded = False
        self._bb_id, self._bb_sha = _find_backbone()
        # PCA 参数（fit 后填充；容器加载恢复）
        self.mu = None                           # (D,) fp32
        self.components = None                   # (k, D) fp32
        self.eigvals = None                      # (k,) fp64
        self.k_components = 0
        self.map_mu = 0.0
        self.map_sigma = 1.0
        self.score_mu = 0.0
        self.score_sigma = 1.0

    # ---- 骨干 ----
    def _ensure_backbone(self):
        """惰性加载 DINOv2（首次 fit/predict 时；权重缺失给明确报错）。"""
        if self._bb_loaded:
            return
        try:
            from transformers import AutoImageProcessor, AutoModel

            self._proc = AutoImageProcessor.from_pretrained(self._bb_id)
            bb = AutoModel.from_pretrained(self._bb_id)
        except Exception as e:  # noqa: BLE001  网络/依赖问题 → 用户可读信息
            raise RuntimeError(
                "SubspaceAD 需要 DINOv2-with-registers 特征骨干（当前源："
                f"{self._bb_id}）。加载失败：{type(e).__name__}: {e}；"
                "请将权重目录放到 weights/dinov2/，或用环境变量 "
                "XCCV_DINOV2_CKPT 指向本地模型目录，或设置 "
                "HF_ENDPOINT=https://hf-mirror.com 后重试") from e
        bb.eval().to(self.device)
        for p in bb.parameters():
            p.requires_grad_(False)
        self._bb = bb
        self._bb_loaded = True
        log.info("DINOv2 骨干加载完成: %s（hidden=%d, patch=%d, registers=%d）",
                 self._bb_id, _feature_dim(bb.config),
                 getattr(bb.config, "patch_size", 14),
                 getattr(bb.config, "num_register_tokens", 0))

    @torch.no_grad()
    def _extract(self, imgs_rgb: list[np.ndarray]) -> tuple[torch.Tensor, tuple]:
        """RGB ndarray 批 → (tokens (B,C,h*w) fp32 CPU, (h, w) 网格)。

        末段层 mean 聚合（_layer_indices）；不请求 attentions
        （省约 13.6GB 显存，主流程不使用注意力图）。"""
        from PIL import Image

        self._ensure_backbone()
        res = int(self.params.get("res") or self.default_image_size)
        pil = [Image.fromarray(im[:, :, ::-1]) for im in imgs_rgb]
        inputs = self._proc(images=pil, return_tensors="pt", do_resize=True,
                            size={"height": res, "width": res},
                            do_center_crop=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self._bb(**inputs, output_hidden_states=True)
        hs = outputs.hidden_states               # tuple(L+1)×(B,N,C)
        cfg = self._bb.config
        ps = int(getattr(cfg, "patch_size", 14))
        num_reg = int(getattr(cfg, "num_register_tokens", 0) or 0)
        drop = 1 + num_reg                       # CLS + registers
        hp, wp = res // ps, res // ps
        n_expected = hp * wp
        idx = _layer_indices(len(hs) - 1)
        feats = torch.stack(
            [hs[li][:, drop:drop + n_expected, :] for li in idx], dim=0
        ).mean(dim=0)                            # (B,N,C)
        return (feats.permute(0, 2, 1).cpu().contiguous(), (hp, wp))

    # ---- PCA ----
    @staticmethod
    def _fit_pca(batches: list[torch.Tensor], dim: int, total: int,
                 device: str, ev: float) -> dict:
        """两遍流式 PCA（参考实现 PCAModel 同式，fp64）：均值→协方差→eigh→按
        方差覆盖率选 k。

        batches：每项 (N_i, C) token 矩阵（已摊平）；返回
        mu/components/eigvals/k（fp32/fp64 CPU）。"""
        mu = torch.zeros(dim, dtype=torch.float64, device=device)
        for b in batches:
            mu += b.to(device, dtype=torch.float64).sum(dim=0)
        mu /= max(total, 1)
        cov = torch.zeros((dim, dim), dtype=torch.float64, device=device)
        for b in batches:
            x = b.to(device, dtype=torch.float64) - mu
            cov += x.T @ x
        cov /= max(total - 1, 1)
        evals, evecs = torch.linalg.eigh(cov)
        order = torch.argsort(evals, descending=True)
        evals = evals[order]
        evecs = evecs[:, order]
        cum = torch.cumsum(evals, dim=0) / torch.clamp(evals.sum(), min=1e-12)
        k = int(torch.searchsorted(
            cum, torch.tensor([ev], dtype=torch.float64, device=device)
        ).item() + 1)
        k = max(1, min(k, evecs.shape[1]))
        return {"mu": mu.cpu().to(torch.float32),
                "components": evecs[:, :k].T.cpu().to(torch.float32),
                "eigvals": evals[:k].cpu(),
                "k": k}

    @staticmethod
    def _residual_scores(x: torch.Tensor, mu: torch.Tensor,
                         components: torch.Tensor) -> torch.Tensor:
        """重构残差（参考实现 _calculate_pca_scores("reconstruction") 同式）：
        x (N,D) fp32、mu (D,)、components (k,D) → (N,) 每行残差平方和。"""
        x0 = x - mu
        z = x0 @ components.T                    # (N,k)
        recon = (z @ components) + mu
        d = x - recon
        return (d * d).sum(dim=1)

    # ---- 训练 ----
    def fit(self, data, cfg: TrainConfig, reporter: TrainReporter | None = None) -> dict:
        reporter = reporter or TrainReporter()
        started = datetime.now()
        self.image_size = self.default_image_size
        params = dict(self.params)
        params.update(dict(getattr(cfg, "model_params", None) or {}))
        self.params.update(params)
        k_shot = int(min(max(int(params.get("k_shot") or 4), 1), 12))
        aug_count = int(min(max(int(params.get("aug_count") or 0), 0), 60))
        ev = float(params.get("ev") or _PCA_EV)
        ok_labels = {c for c in (data.classes or [])
                     if is_negative_dir_name(c)}
        pool = ([s for s in data.train_samples if s.label in ok_labels]
                if ok_labels else list(data.train_samples))
        if not pool:
            pool = list(data.train_samples)
        if not pool:
            raise ValueError("异常检测需要 OK/良品图（NEGATIVE_DIR_NAMES 目录）"
                             "或至少一组正常样本图")
        val_ok = [s for s in data.val_samples
                  if (ok_labels and s.label in ok_labels) or not ok_labels]
        val_ng = [s for s in data.val_samples
                  if ok_labels and s.label not in ok_labels]
        rng = np.random.default_rng(cfg.seed)
        order = rng.permutation(len(pool))
        k_eff = min(k_shot, len(pool))
        support = [pool[i] for i in order[:k_eff]]
        rest = [pool[i] for i in order[k_eff:]]
        log.info("SubspaceAD: k-shot=%d（池 %d 张）aug=%d ev=%.2f 支持图=%s",
                 k_eff, len(pool), aug_count, ev,
                 ",".join(Path(s.image_path).name for s in support))

        # 1) 特征抽取（支持图 + 增强变体）→ fp32 缓存（两遍流式共用）
        aug = _augment_fn(aug_count)
        from PIL import Image

        def _pil_of(s):
            img = imread_unicode(s.image_path)
            if img is None:
                return None
            return Image.fromarray(img[:, :, ::-1])

        batches: list[torch.Tensor] = []
        feat_dim = 0
        n_images = 0
        batch_imgs: list[Image.Image] = []
        for si, s in enumerate(support, start=1):
            pil = _pil_of(s)
            if pil is None:
                continue
            variants = [pil]
            if aug is not None:
                variants.extend(aug(pil) for _ in range(aug_count))
            for v in variants:
                arr = np.asarray(v)[:, :, ::-1]
                batch_imgs.append(arr)
                n_images += 1
                if len(batch_imgs) >= _EXTRACT_BATCH:
                    toks, _grid = self._extract(batch_imgs)
                    feat_dim = int(toks.shape[1])
                    # (B,C,N) → (B*N,C) token 矩阵（PCA 两遍流式契约）
                    batches.append(toks.permute(0, 2, 1).reshape(
                        -1, feat_dim))
                    batch_imgs = []
            if reporter is not None:
                reporter.on_train_progress(1, si, len(support))
        if batch_imgs:
            toks, _grid = self._extract(batch_imgs)
            feat_dim = int(toks.shape[1])
            batches.append(toks.permute(0, 2, 1).reshape(-1, feat_dim))
            batch_imgs = []
        if not batches:
            raise ValueError("支持图特征抽取失败（图片全部无法读取）")
        total_tokens = int(sum(b.shape[0] for b in batches))
        log.info("SubspaceAD 特征: %d 图（含增强）× %d token，dim=%d",
                 n_images, total_tokens // max(n_images, 1), feat_dim)

        # 2) PCA 两遍流式拟合
        pca = self._fit_pca(batches, feat_dim, total_tokens, self.device, ev)
        self.mu = pca["mu"]
        self.components = pca["components"]
        self.eigvals = pca["eigvals"]
        self.k_components = int(pca["k"])
        log.info("SubspaceAD PCA: k=%d（覆盖率 %.2f）", self.k_components, ev)
        del batches

        # 3) 标定（阶段 2）：留出 OK 图优先，不足退支持图+增强子集
        calib_sources = rest[:_CALIB_CAP] if len(rest) >= 3 else []
        if not calib_sources:
            calib_sources = support[:_CALIB_CAP]
        map_vals, ok_scores = [], []
        n_cal = len(calib_sources)
        for ci, s in enumerate(calib_sources, start=1):
            img = imread_unicode(s.image_path)
            if img is None:
                continue
            smap, sc = self._score_map(img)
            map_vals.append(smap.reshape(-1))
            ok_scores.append(sc)
            if reporter is not None:
                reporter.on_train_progress(2, ci, n_cal)
        if map_vals:
            all_map = np.concatenate(map_vals)
            self.map_mu = float(np.mean(all_map))
            self.map_sigma = max(float(np.std(all_map)), 1e-6)
        if len(ok_scores) >= 3:
            self.score_mu = float(np.mean(ok_scores))
            self.score_sigma = max(float(np.std(ok_scores)), 1e-6)

        # 4) 验证指标（阶段 3；AUROC/AP：NG + OK）
        scores, labels = [], []
        n_val = min(len(val_ng), _VAL_CAP) + min(len(val_ok), _VAL_CAP)
        done = 0
        for s in val_ng[:_VAL_CAP]:
            img = imread_unicode(s.image_path)
            if img is not None:
                scores.append(self._score_map(img)[1])
                labels.append(1)
            done += 1
            if reporter is not None:
                reporter.on_train_progress(3, done, max(n_val, 1))
        for s in val_ok[:_VAL_CAP]:
            img = imread_unicode(s.image_path)
            if img is not None:
                scores.append(self._score_map(img)[1])
                labels.append(0)
            done += 1
            if reporter is not None:
                reporter.on_train_progress(3, done, max(n_val, 1))
        auroc = image_auroc(scores, labels)
        ap = image_ap(scores, labels)
        z25 = _z_from_conf(0.25)
        n_fp = sum(1 for s, l in zip(scores, labels)
                   if l == 0 and (s - self.score_mu) / self.score_sigma >= z25)
        n_ok_v = sum(1 for l in labels if l == 0)
        # 权重落盘：明文 best.pt
        best_pt = Path(cfg.output_dir) / "best.pt"
        self.save_weights(best_pt)
        metrics = {
            "final": {"auroc": round(auroc, 4), "ap": round(ap, 4),
                      "ok_fp_rate": (round(n_fp / n_ok_v, 4) if n_ok_v else None)},
            "classes": self.classes,
            "k_shot": k_eff, "aug_count": aug_count,
            "pca_components": self.k_components,
            "support_images": len(support),
            "feature_dim": feat_dim,
            "train_ok_images": len(pool),
            "val_ng_images": len(val_ng), "val_ok_images": len(val_ok),
            "image_size": self.image_size,
            "backbone": self._bb_id, "backbone_fingerprint": self._bb_sha,
            "params": dict(self.params),
            "history": [{"epoch": 1, "train_loss": 0.0, "auroc": round(auroc, 4)}],
        }
        metrics["start_time"] = started.strftime("%Y-%m-%d %H:%M:%S")
        metrics["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics["status"] = "completed"
        reporter.on_epoch(1, {"auroc": auroc})
        reporter.on_finish(True)
        return metrics

    # ---- 单图评分（标定与预测共用）----
    def _score_map(self, img_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """BGR 图 → (平滑异常图 (res,res) float, 图像级原始残差分)。"""
        res = int(self.params.get("res") or self.default_image_size)
        toks, grid = self._extract([img_bgr])            # (1,C,h*w)
        hp, wp = grid
        x = toks[0].permute(1, 0)                        # (N,C)
        patch = self._residual_scores(
            x.to(self.device), self.mu.to(self.device),
            self.components.to(self.device)).reshape(hp, wp).cpu().numpy()
        smap = _post_process(patch, res)
        return smap, _topk_mean(smap)

    # ---- 推理 ----
    def predict(self, image_bgr: np.ndarray, conf: float = 0.25,
                iou: float = 0.45) -> Prediction:
        if self.mu is None or self.components is None:
            raise RuntimeError("异常检测模型未训练（PCA 子空间为空）")
        smap, img_score = self._score_map(image_bgr)
        z_map = (smap - self.map_mu) / self.map_sigma
        img_z = (img_score - self.score_mu) / self.score_sigma
        z_thr = _z_from_conf(conf)
        h_img, w_img = image_bgr.shape[:2]
        mask = (z_map > z_thr).astype(np.uint8)
        mask_full = cv2.resize(mask, (w_img, h_img),
                               interpolation=cv2.INTER_NEAREST)
        boxes, scores, masks = [], [], []
        contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        gh, gw = z_map.shape
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < 2 or bh < 2:
                continue
            boxes.append([float(x), float(y), float(x + bw), float(y + bh)])
            gx1 = max(0, min(gw - 1, int(x * gw / w_img)))
            gy1 = max(0, min(gh - 1, int(y * gh / h_img)))
            gx2 = max(gx1 + 1, min(gw, int((x + bw) * gw / w_img) + 1))
            gy2 = max(gy1 + 1, min(gh, int((y + bh) * gh / h_img) + 1))
            scores.append(round(float(z_map[gy1:gy2, gx1:gx2].max()), 4))
            m = np.zeros((h_img, w_img), dtype=np.uint8)
            cv2.drawContours(m, [cnt], -1, 1, -1)
            masks.append(m.astype(bool))
        return Prediction(labels=[0] * len(boxes), scores=scores,
                          boxes=boxes, masks=masks or None,
                          probs=[round(float(img_z), 4)])

    # ---- 权重 ----
    def save_weights(self, path: Path) -> None:
        """xccv-subspacead 容器（明文 torch.save）。

        DINOv2 骨干不进容器：只记来源 id + 指纹，加载时按发现链重建。"""
        ckpt = {
            "format": "xccv-subspacead",
            "classes": self.classes,
            "image_size": self.image_size,
            "mu": self.mu,
            "components": self.components,
            "eigvals": self.eigvals,
            "k_components": self.k_components,
            "map_mu": self.map_mu, "map_sigma": self.map_sigma,
            "score_mu": self.score_mu, "score_sigma": self.score_sigma,
            "params": dict(self.params),
            "backbone_id": self._bb_id,
            "backbone_sha": self._bb_sha,
        }
        torch.save(ckpt, path)

    def load_weights(self, source) -> None:
        """source：权重路径 / ckpt dict（registry 统一入口语义）。"""
        if isinstance(source, dict):
            ckpt = source
        else:
            ckpt = torch.load(source, map_location="cpu", weights_only=False)
        if ckpt.get("format") != "xccv-subspacead":
            raise ValueError("不是 xccv-subspacead 权重文件")
        self.classes = list(ckpt.get("classes") or ["异常"])
        self.image_size = int(ckpt.get("image_size") or self.default_image_size)
        params = dict(ckpt.get("params") or {})
        self.params.update(params)
        self.params["res"] = self.image_size
        self.mu = ckpt["mu"]
        self.components = ckpt["components"]
        self.eigvals = ckpt.get("eigvals")
        self.k_components = int(ckpt.get("k_components") or 0)
        self.map_mu = float(ckpt.get("map_mu", 0.0))
        self.map_sigma = max(float(ckpt.get("map_sigma", 1.0)), 1e-6)
        self.score_mu = float(ckpt.get("score_mu", 0.0))
        self.score_sigma = max(float(ckpt.get("score_sigma", 1.0)), 1e-6)
        sha = str(ckpt.get("backbone_sha") or "")
        if sha and sha != self._bb_sha:
            log.warning("DINOv2 权重指纹与训练时不一致（%s ≠ 当前 %s），"
                        "特征空间可能漂移，建议用同骨干复训",
                        sha[:16], self._bb_sha[:16])


