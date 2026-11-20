"""DINO ViT-B/8 特征器（CIF 后端特征底座；V0.7.3）。

主路径：timm `vit_base_patch8_224_dino`（与复现工程 `E:\Zcode\Study\CIF`
逐行同款调用——依赖已对齐；timm 已随复现工作装在平台
venv，不新增安装）。回退：纯 torch 重建的 DinoViT（键名与 timm
VisionTransformer 一致：150 键 cls_token/pos_embed/patch_embed.proj/12×
block/norm，探针 runs/probe_dino_ckpt.py 全部核对；保留零依赖能力）。

权重路径（"用户自备/可选下载"红线，SAM weights/sam 同款惯例）：
  1) 环境变量 XCCV_DINO_CKPT 指向权重文件；
  2) <app_root>/weights/dino/dino_vitbase8_pretrain.pth
不自动下载、不随包分发。
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

_EMBED_DIM = 768
_NUM_HEADS = 12
_DEPTH = 12
_MLP_RATIO = 4
_PATCH = 8
_IMAGE_SIZE = 224          # pos_embed 定长 785 token，输入固定 224×224

_CKPT_FILENAME = "dino_vitbase8_pretrain.pth"

_DINO_MISSING_MSG = (
    "CIF 模型需要 DINO ViT-B/8 特征骨干权重（dino_vitbase8_pretrain.pth）。"
    "请将该文件放到 weights/dino/ 目录，或用环境变量 XCCV_DINO_CKPT 指向它。"
)


class _Attention(nn.Module):
    """多头自注意力（qkv 融合线性层，与 checkpoint 布局一致）。"""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, -1)
        return self.proj(out)


class _Mlp(nn.Module):
    """FFN（fc1 → GELU → fc2，与 checkpoint 键名一致）。"""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class _PatchEmbed(nn.Module):
    """patch 嵌入容器（内部 conv 命名 proj，与 checkpoint 键 patch_embed.proj 对齐）。"""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=_PATCH, stride=_PATCH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _Block(nn.Module):
    """Transformer 块（pre-norm；残差连接；LN eps=1e-6 与 timm 一致）。"""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _Mlp(dim, dim * mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DinoViT(nn.Module):
    """DINO ViT-B/8 推理特征器：forward 返回 (B, 768, 784) patch 特征。"""

    def __init__(self, depth: int = _DEPTH):
        super().__init__()
        dim = _EMBED_DIM
        self.patch_embed = _PatchEmbed(dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, (_IMAGE_SIZE // _PATCH) ** 2 + 1, dim))
        self.blocks = nn.ModuleList(
            _Block(dim, _NUM_HEADS, _MLP_RATIO) for _ in range(depth))
        self.norm = nn.LayerNorm(dim, eps=1e-6)   # timm.layers.LayerNorm 同款

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """归一化输入 (B,3,224,224) → patch 特征 (B,768,784)。"""
        assert x.shape[-2:] == (_IMAGE_SIZE, _IMAGE_SIZE), \
            f"CIF 特征输入固定 {_IMAGE_SIZE}×{_IMAGE_SIZE}，得到 {tuple(x.shape[-2:])}"
        tokens = self.patch_embed(x)                       # (B,768,28,28)
        tokens = tokens.flatten(2).transpose(1, 2)         # (B,784,768)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        x = torch.cat([cls, tokens], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 1:].transpose(1, 2).contiguous()       # 丢 cls → (B,768,784)

    def load_dino_state(self, state: dict) -> None:
        """加载 DINO state_dict（键名与 timm VisionTransformer 一致，strict）。"""
        self.load_state_dict(state, strict=True)


def find_dino_checkpoint() -> Path | None:
    """定位 DINO 权重：环境变量 XCCV_DINO_CKPT → weights/dino 默认落点。"""
    env = os.environ.get("XCCV_DINO_CKPT", "").strip()
    if env:
        p = Path(env)
        if p.is_file():
            return p
        log.warning("XCCV_DINO_CKPT 指向的文件不存在: %s", p)
    from ...app_paths import app_root

    p = app_root() / "weights" / "dino" / _CKPT_FILENAME
    return p if p.is_file() else None


class _TimmViT(nn.Module):
    """timm VisionTransformer 包装：forward 与复现工程 ViTExtractor 逐行同构
    （patch_embed → _pos_embed → norm_pre → blocks → norm → 丢 cls）。"""

    def __init__(self, backbone: nn.Module, depth: int):
        super().__init__()
        if depth < 12:
            backbone.blocks = nn.Sequential(*list(backbone.blocks)[:depth])
        self._bb = backbone

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """归一化输入 (B,3,224,224) → patch 特征 (B,768,784)。"""
        b = self._bb
        y = b.patch_embed(x)
        y = b._pos_embed(y)
        y = b.norm_pre(y)
        y = b.blocks(y)
        y = b.norm(y)
        return y[:, 1:].permute(0, 2, 1).contiguous()


def load_dino_vit(device: str = "cpu", depth: int = _DEPTH) -> tuple[nn.Module, str]:
    """构建 ViT 并加载权重；缺失/损坏时抛 FileNotFoundError/RuntimeError。

    主路径 = timm `vit_base_patch8_224_dino`（与复现工程同款调用，依赖已
    对齐；timm 已随复现装在平台 venv）；timm 缺失时回退本文件
    的纯 torch DinoViT（零依赖能力保留，键映射已探针核对）。
    返回 (模型, 权重指纹 hex16)。
    """
    ckpt = find_dino_checkpoint()
    if ckpt is None:
        raise FileNotFoundError(_DINO_MISSING_MSG)
    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()[:16]
    try:
        import timm

        backbone = timm.create_model(
            "vit_base_patch8_224_dino", pretrained=False,
            checkpoint_path=str(ckpt))
        vit: nn.Module = _TimmViT(backbone, depth)
        source = "timm"
    except ImportError:
        log.warning("timm 不可用，回退纯 torch ViT-B/8（特征与 timm 等价）")
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        hand = DinoViT(depth=depth)
        try:
            hand.load_dino_state(state)
        except Exception as e:
            raise RuntimeError(
                f"DINO 权重与 ViT-B/8 结构不匹配（{ckpt.name}）: {e}") from e
        vit = hand
        source = "pure-torch"
    vit.to(device).eval()
    for p in vit.parameters():
        p.requires_grad_(False)
    log.info("DINO ViT-B/8 权重加载: %s (%.0f MB, 指纹 %s, 路径=%s)",
             ckpt, ckpt.stat().st_size / 1e6, digest, source)
    return vit, digest


def fingerprint_matches(digest_hex16: str) -> bool:
    """容器内记录的权重指纹是否与当前权重文件一致（不一致 = 特征空间漂移）。"""
    try:
        ckpt = find_dino_checkpoint()
        if ckpt is None:
            return False
        return hashlib.sha256(ckpt.read_bytes()).hexdigest()[:16] == digest_hex16
    except Exception:
        return False
