"""CIF 无监督异常检测后端（RGB-only，论文口径 L=1/α=0.9/K=4）。

来源：CIF（AAAI 2026，arXiv 2511.05966）少样本多模态异常检测；本地复现
`E:\Zcode\Study\CIF`（B+）。平台集成口径按集成设计落地。
依赖对齐：主路径与参考实现一致——timm ViT 特征
（dino_vit.py）、sklearn.KMeans(random_state=0)、anomalib SRP(eps=0.9)
（三包均已随复现装在平台 venv，不新增安装）；anomalib/timm/sklearn 缺失
时各自回退（cv2.kmeans / 纯 torch ViT / 原始特征直算），零依赖能力保留。
torchmetrics 仍不需要（平台自有 image_auroc/image_ap）。

管线（RGB 分支）：
1. 特征：DINO ViT-B/8（224 输入 → 28×28=784 patch token，768 维，全 12 层）；
2. SAHC 模糊超图：K-Means(K=4) 质心 → patch×质心余弦相似度按行 min-max
   归一 → ≥0.9 为邻接（平台 2D 图无前景掩码，掩码恒全 1——上游 MVTec 2D
   数据集对 RGB 即 torch.ones，口径一致）；
3. 记忆库：节点=patch 特征+所属超边 id（(N,769)），逐图累积后按簇
   K-Center-Greedy 共采样收缩（整体 10%，绝对上限 MAX_MEMORY_NODES）；
4. 推理：测试图超图与记忆库超图拼联合超图做 Bi-TF-MP（L=1、α=0.9、topk=1、
   双向）→ 传播后节点 1-NN 距离 × 超边级检索（topk=3）分数相乘 → 上采样
   224 → ^1.5 → 均值卷积平滑（5×5×5 + 7×7×3）→ 与 patchcore 一致的 z 标定
   阈值语义（conf→z=1/conf−1）→ 连通域出框/掩码 + 图像级 z 分数。

------------------------------------------------------------------------
ANOMALY 后端契约（后续 SubspaceAD/RAD 等接入模板；本类即参照实现）：
- 注册：engine/models/registry.py MODEL_SPECS + load_model 容器分支（模型层
  唯一触碰点）；supported_tasks=(TaskType.ANOMALY,)，family="folder"；
- fit(data, cfg, reporter) -> metrics：metrics.json 键与 patchcore 同构
  （final.auroc/ap/ok_fp_rate、classes、history 单条；additive 允许）；
- predict(image_bgr, conf, iou) -> Prediction：labels 全 0、boxes/scores
  （框内最大 z）、masks（与框一一对应）、probs=[图像级 z]——测试平台/
  复盘/回流零改动即可消费；
- save_weights/load_weights："xccv-<backend>" 明文容器（fit 收尾写
  best.pt）；冻结骨干
  （DINO 等）不进容器，只记指纹，加载时按 weights/ 落点重建；
- TensorRT .engine：记忆库/检索类后端一律不支持（与 patchcore 同口径，
  文档注明）。
------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..dataset.base import TaskType
from ..dataset.reader import is_negative_dir_name
from ..io_utils import imread_unicode
from .anomaly import image_ap, image_auroc
from .base import BaseModel, Prediction, TrainConfig, TrainReporter
from .dino_vit import fingerprint_matches, load_dino_vit

log = logging.getLogger(__name__)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# ---- 论文/复现口径超参（runs 评估 bat 显式传参，见设计文档 §二）----
N_CLUSTERS = 4            # 超边数 K
SAHC_THRESHOLD = 0.9      # SAHC 余弦邻接阈值
SAMPLING_RATIO = 0.1      # 记忆库共采样比例
BITFMP_LAYERS = 1         # Bi-TF-MP 传播层数 L
BITFMP_ALPHA = 0.9        # 自信息保留比 α
EDGE_TOPK = 3             # 超边级检索 topk
N_NEIGHBORS = 9           # 图像级分数 softmax 修正近邻数
MAX_MEMORY_NODES = 4000   # 记忆库绝对上限（联合超图 W 矩阵规模可控，设计文档 §三）
_CALIB_RATIO = 0.10       # 留出标定的 OK 图比例（图像级/图级 z 标定源）

_SMOOTH_L = (5, 2, 5)     # 平滑链：(kernel, pad, 次数)——5×5 均值卷积 ×5
_SMOOTH_U = (7, 3, 3)     # 7×7 均值卷积 ×3


def _z_from_conf(conf: float) -> float:
    """conf∈(0,1] → z 阈值（与 patchcore 同语义：conf 0.25 ⇒ z=3.0）。"""
    conf = min(max(float(conf), 1e-3), 1.0)
    return 1.0 / conf - 1.0


def _euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """成对欧氏距离（|x|²−2x·yᵀ+|y|² 形式；上游 search.euclidean_dist 同式）。"""
    x_norm = x.pow(2).sum(dim=-1, keepdim=True)
    y_norm = y.pow(2).sum(dim=-1, keepdim=True)
    res = x_norm - 2 * torch.matmul(x, y.transpose(-2, -1)) + y_norm.transpose(-2, -1)
    return res.clamp_min_(0).sqrt_()


def _kmeans(feats: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """K-Means（返回 (labels(M,), centers(k,D))）。

    主路径 = sklearn.KMeans(n_clusters=k, random_state=0, n_init='auto')，
    与参考实现逐参一致。
    回退 = cv2.kmeans（kmeans++ 同语义；内部 RNG 随进程推进，先
    setRNGSeed(0) 复位保持同输入 ⇒ 同输出）。
    """
    data = np.ascontiguousarray(feats, dtype=np.float32)
    k = max(1, min(k, data.shape[0]))
    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(data)
        return (km.labels_.reshape(-1).astype(np.int64),
                km.cluster_centers_.astype(np.float32))
    except ImportError:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
        cv2.setRNGSeed(0)
        _compact, labels, centers = cv2.kmeans(
            data, k, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
        return labels.reshape(-1).astype(np.int64), centers.astype(np.float32)


def _make_srp(seed: int):
    """anomalib SparseRandomProjection(eps=0.9)（参考实现 KCenter 算法）。

    返回工厂函数；anomalib 缺失回退复现工程 anomalib_compat 的本地 JL 实现；
    两者都缺返回 None（调用方退原始特征直算）。
    """
    try:
        from anomalib.models.components.dimensionality_reduction import (
            SparseRandomProjection,
        )

        return lambda: SparseRandomProjection(eps=0.9, random_state=int(seed))
    except ImportError:
        pass
    try:
        from sklearn.random_projection import johnson_lindenstrauss_min_dim

        class _LocalSRP:
            """anomalib 兼容层的 fallback 实现（sklearn-JL）。"""

            def __init__(self, eps: float, random_state: int | None):
                self.eps = eps
                self.random_state = random_state
                self.n_samples = 0
                self.projection_: np.ndarray | None = None

            def _sparse_random_matrix(self, n_features: int) -> np.ndarray:
                rng = np.random.default_rng(self.random_state)
                density = min(1.0, 1.0 / np.sqrt(n_features))
                n_comp = johnson_lindenstrauss_min_dim(
                    n_samples=max(self.n_samples, 2), eps=self.eps)
                n_comp = int(min(max(n_comp, 1), n_features))
                mask = rng.random((n_features, n_comp)) < density
                signs = rng.choice([-1.0, 1.0], size=(n_features, n_comp))
                return (np.where(mask, signs, 0.0) * np.sqrt(1.0 / density)
                        / np.sqrt(n_comp))

            def fit(self, features):
                self.n_samples = int(features.shape[0])
                self.projection_ = self._sparse_random_matrix(int(features.shape[1]))
                return self

            def transform(self, features):
                proj = torch.from_numpy(self.projection_).to(features.dtype).to(features.device)
                return features @ proj

        return lambda: _LocalSRP(0.9, int(seed))
    except ImportError:
        return None


def _construct_hypergraph(features: torch.Tensor, mask: torch.Tensor | None,
                          num_clusters: int = N_CLUSTERS,
                          threshold: float = SAHC_THRESHOLD,
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """SAHC 模糊超图（上游 hypergraph.construct_hypergraph 移植）。

    features (B,C,N)，mask (B,1,side,side)（None=全前景）。
    返回 (adj (B,N,K), X_E (B,K,C))。
    """
    bs, dim, num_patches = features.shape
    side = int(num_patches ** 0.5)
    if mask is None:
        mask = torch.ones(bs, 1, side, side, device=features.device)
    mask_flat = mask.reshape(bs, -1)
    adj = torch.zeros(bs, num_patches, num_clusters, device=features.device)
    for b in range(bs):
        feat = features[b].T                                  # (N, C)
        valid = mask_flat[b].bool()
        valid_feat = feat[valid]
        if valid_feat.shape[0] == 0:                          # 空图兜底：全图参与
            valid = torch.ones_like(valid)
            valid_feat = feat
        labels, centers_np = _kmeans(valid_feat.cpu().numpy(), num_clusters)
        k_eff = centers_np.shape[0]
        centers = torch.from_numpy(centers_np).to(features.device)
        norm_feat = F.normalize(feat, p=2, dim=1)
        norm_centers = F.normalize(centers, p=2, dim=1)
        sim = torch.mm(norm_feat, norm_centers.T)             # (N, k_eff)
        min_v = sim.min(dim=1, keepdim=True).values
        max_v = sim.max(dim=1, keepdim=True).values
        denom = (max_v - min_v)
        denom[denom == 0] = 1e-8
        sim_n = (sim - min_v) / denom
        assign = torch.zeros_like(sim)
        assign[sim_n >= threshold] = 1.0
        adj[b, :, :k_eff] = assign
        adj[b, ~valid, :] = 0
    d_he = torch.diag_embed(1.0 / torch.clamp(adj.sum(dim=1), min=1e-12))
    d_hv = torch.diag_embed(1.0 / torch.sqrt(torch.clamp(adj.sum(dim=2), min=1e-12)))
    x_e = d_he @ adj.transpose(1, 2) @ d_hv @ features.transpose(1, 2)
    return adj, x_e


class _BiTFMP:
    """双向免训练消息传递（上游 hypergraph.BiTFMP 移植，topk=1 双向）。"""

    def __init__(self, layers: int = BITFMP_LAYERS, alpha: float = BITFMP_ALPHA,
                 topk: int = 1, bidirectional: bool = True, eps: float = 1e-8):
        self.L = max(1, int(layers))
        self.alpha = float(alpha)
        self.topk = max(1, int(topk))
        self.bidirectional = bool(bidirectional)
        self.eps = eps

    def _cross_incidence(self, x: torch.Tensor, mem_x: torch.Tensor,
                         ) -> torch.Tensor:
        """跨库超边关联矩阵 (B, N+A, N+A)（topk 最近邻，双向时反向各一条）。

        上游逐节点 python scatter 循环等价向量化（语义一致：无向二值边）。
        """
        b, c, n = x.shape
        a = mem_x.size(1)
        dev = x.device
        x_n = F.normalize(x, dim=1)                    # (B,C,N)
        m_n = F.normalize(mem_x, dim=2).permute(0, 2, 1)  # (B,C,A)
        sim = torch.bmm(x_n.permute(0, 2, 1), m_n)     # (B,N,A)
        knn = sim.topk(k=self.topk, dim=-1).indices    # (B,N,topk)
        e_cross = n + (a if self.bidirectional else 0)
        h = torch.zeros(b, n + a, e_cross, device=dev)
        idx = torch.arange(n, device=dev)
        for bi in range(b):
            h[bi, idx, idx] = 1.0                      # 自环边（与上游实现一致）
            for k in range(self.topk):
                # 前向边 j（列 j）：行 = j + knn[j,k]（上游 scatter_ 语义：
                # 边含测试节点 j 与其最近记忆节点，列=边 id、行=成员）
                h[bi, n + knn[bi, :, k], idx] = 1.0
        if self.bidirectional:
            rev = sim.transpose(1, 2).topk(k=self.topk, dim=-1).indices  # (B,A,topk)
            a_idx = torch.arange(a, device=dev)
            for bi in range(b):
                h[bi, n + a_idx, n + a_idx] = 1.0      # 记忆节点自环边
                for k in range(self.topk):
                    h[bi, rev[bi, :, k], n + a_idx] = 1.0  # 记忆节点→最近测试节点
        return h

    def _tf_mp(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """对称归一 clique 展开 + 幂次和传播（上游 tf_mp 逐式移植）。"""
        bs, c, n = x.shape
        d_e = h.sum(dim=1) + self.eps
        h_w = torch.bmm(h, torch.diag_embed(1.0 / d_e))
        w = torch.bmm(h_w, h.transpose(1, 2))          # (B,N,N)
        eye = torch.eye(n, device=x.device).unsqueeze(0).expand(bs, -1, -1)
        w_tilde = w + eye
        d_inv_sqrt = 1.0 / torch.sqrt(w_tilde.sum(dim=2) + self.eps)
        d_diag = torch.diag_embed(d_inv_sqrt)
        a_sym = torch.bmm(torch.bmm(d_diag, w_tilde), d_diag)
        # S = α·I + Σ α(1−α)^l·A^l + (1−α)^L·A^L（上游幂次和同式）
        a_power = eye.clone()
        s = self.alpha * a_power.clone()
        for l in range(1, self.L):
            a_power = torch.bmm(a_power, a_sym)
            s += self.alpha * (1 - self.alpha) ** l * a_power
        a_power = torch.bmm(a_power, a_sym)
        s += (1 - self.alpha) ** self.L * a_power
        return torch.bmm(s, x.transpose(1, 2)).transpose(1, 2)

    def forward(self, features: torch.Tensor, adj: torch.Tensor,
                total_feats: torch.Tensor, mem_adj: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """联合超图传播：features (B,C,N)、adj (B,N,K)、total_feats (A,C)、
        mem_adj (A,K)；返回 (X_hat (B,C,N), X_E (B,K,C))。"""
        b, c, n = features.shape
        mem_x = total_feats.unsqueeze(0).expand(b, -1, -1).to(features.dtype)
        mem_h = mem_adj.unsqueeze(0).expand(b, -1, -1).to(features.dtype)
        h_cross = self._cross_incidence(features, mem_x)
        # 联合关联矩阵：测试超图 | 记忆超图 | 跨库边（上游 make_joint_H 同式）
        pad_mem = torch.zeros(b, mem_x.size(1), adj.shape[2], device=adj.device)
        h_test = torch.cat([adj, pad_mem], dim=1)
        pad_test = torch.zeros(b, n, mem_adj.shape[1], device=adj.device)
        h_mem = torch.cat([pad_test, mem_h], dim=1)
        h_joint = torch.cat([h_test, h_mem, h_cross], dim=2)
        joint_x = torch.cat([features, mem_x.permute(0, 2, 1)], dim=2)
        joint_hat = self._tf_mp(joint_x, h_joint)
        x_hat = joint_hat[:, :, :n]
        d_he = torch.diag_embed(1.0 / torch.clamp(adj.sum(dim=1), min=self.eps))
        d_hv = torch.diag_embed(1.0 / torch.sqrt(torch.clamp(adj.sum(dim=2), min=self.eps)))
        x_e = d_he @ adj.transpose(1, 2) @ d_hv @ x_hat.transpose(1, 2)
        return x_hat, x_e


def _build_adj(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """记忆库关联矩阵（上游 eval.build_adj 移植）：(M,769) → (adj (M,E), 新 id)。"""
    feats = x[:, :-1]
    ids = x[:, -1].long()
    unique, inverse = torch.unique(ids, return_inverse=True)
    adj = torch.zeros(x.shape[0], unique.shape[0], device=x.device)
    adj[torch.arange(x.shape[0]), inverse] = 1.0
    return adj, torch.cat([feats, inverse.unsqueeze(1).float()], dim=1)


def _nearest_neighbors(embedding: torch.Tensor, memory: torch.Tensor,
                       fore_mask: torch.Tensor | None = None,
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """前景 patch 对记忆库 1-NN 距离（上游 nearest_neighbors(n=1) 移植）。

    embedding (N,D)、memory (M,D)、fore_mask (N,) bool；背景 patch 分数置 0。
    """
    if fore_mask is None:
        fore_mask = torch.ones(embedding.shape[0], dtype=torch.bool,
                               device=embedding.device)
    fore = fore_mask.reshape(-1).bool()
    fg_idx = torch.nonzero(fore, as_tuple=True)[0]
    scores = torch.zeros(embedding.shape[0], device=embedding.device)
    locations = torch.zeros(embedding.shape[0], dtype=torch.long,
                            device=embedding.device)
    if fg_idx.numel():
        dist = _euclidean_dist(embedding[fg_idx], memory)
        scores_fg, locs = dist.min(1)
        scores[fg_idx] = scores_fg
        locations[fg_idx] = locs
    return scores, locations


def _hyper_search(features_test: torch.Tensor, adj: torch.Tensor,
                  hyperedge: torch.Tensor, total_feats: torch.Tensor,
                  total_edges: torch.Tensor, topk: int = EDGE_TOPK,
                  ) -> torch.Tensor:
    """超边级节点异常分（上游 hyper_search 移植）：测试图每簇节点对记忆库
    topk 匹配簇内节点的最小欧氏距离；(B,N)。"""
    bs = features_test.shape[0]
    n_clusters = hyperedge.shape[1]
    total_ids = total_feats[:, -1:]
    scores = torch.zeros(bs, features_test.shape[2], device=features_test.device)
    for b in range(bs):
        sim = torch.matmul(F.normalize(hyperedge[b], dim=1),
                           F.normalize(total_edges[:, :-1], dim=1).T)
        sim = (sim + 1) / 2
        _, topk_idx = torch.topk(sim, min(topk, total_edges.shape[0]), dim=1)
        real_topk = total_edges[:, -1][topk_idx]        # (K, topk) 记忆簇 id
        node_feats = features_test[b].permute(1, 0).contiguous()
        node_edge_ids = torch.argmax(adj[b], dim=-1)
        no_conn = adj[b].sum(dim=-1) == 0
        node_edge_ids = torch.where(no_conn, torch.full_like(node_edge_ids, -1),
                                    node_edge_ids)
        for eid in range(n_clusters):
            mask_e1 = node_edge_ids == eid
            n1 = node_feats[mask_e1]
            e2_ids = real_topk[eid]
            mask_e2 = torch.isin(total_ids.squeeze(1), e2_ids)
            n2 = total_feats[mask_e2, :-1]
            if n1.shape[0] > 0 and n2.shape[0] > 0:
                dist = _euclidean_dist(n1, n2)
                scores[b, mask_e1] = dist.min(dim=1).values
            elif n1.shape[0] > 0:
                scores[b, mask_e1] = 1e6                # 无匹配簇（与上游实现一致）
    return scores


def _compute_anomaly_score(patch_scores: torch.Tensor, locations: torch.Tensor,
                           embedding: torch.Tensor, memory: torch.Tensor,
                           ) -> torch.Tensor:
    """图像级分数（上游 compute_anomaly_score 的 n_neighbors=9 分支移植）。

    patch_scores (B,N)、locations (B,N)、embedding (B·N,D)、memory (M,D)。
    """
    b, num = patch_scores.shape
    max_idx = torch.argmax(patch_scores, dim=1)
    max_feats = embedding.reshape(b, num, -1)[torch.arange(b), max_idx]
    score = patch_scores[torch.arange(b), max_idx]
    nn_index = locations[torch.arange(b), max_idx]
    nn_sample = memory[nn_index]                        # (B,D)
    k = min(N_NEIGHBORS, memory.shape[0])
    _, support = _euclidean_dist(nn_sample, memory).topk(k=k, largest=False, dim=1)
    dist = _euclidean_dist(max_feats.unsqueeze(1), memory[support])
    weights = (1 - F.softmax(dist.squeeze(1), dim=1))[..., 0]
    return weights * score


def _kcenter_greedy(feats: torch.Tensor, n_select: int,
                    rng: np.random.Generator) -> torch.Tensor:
    """K-Center-Greedy coreset（原始特征直算，SRP 不可用时的回退路径）。

    feats (M,D)；返回选中索引。选 1 个时取 medoid（最小化最大距离，与上游
    n_sample==1 分支一致）；多个时逐中心增量更新最小距离，起始点由 rng
    抽取。
    """
    m = feats.shape[0]
    n_select = max(1, min(n_select, m))
    if n_select == 1 or m == 1:
        dists = torch.cdist(feats, feats)
        return torch.tensor([int(torch.argmin(dists.max(dim=1).values).item())],
                            dtype=torch.long)
    min_dist: torch.Tensor | None = None
    selected: list[int] = []
    idx = int(rng.integers(m))
    for _ in range(n_select):
        d = torch.cdist(feats[idx:idx + 1], feats).reshape(-1)
        min_dist = d if min_dist is None else torch.minimum(min_dist, d)
        selected.append(idx)
        min_dist[selected[-1]] = 0
        idx = int(torch.argmax(min_dist).item())
        if min_dist[idx].item() <= 0:                   # 全覆盖，提前收尾
            break
    return torch.tensor(sorted(set(selected)), dtype=torch.long)


def _kcenter_greedy_srp(feats: torch.Tensor, n_select: int, seed: int,
                        ) -> torch.Tensor:
    """上游 sampling.KCenterGreedy 对齐版：SRP(eps=0.9) 降维后贪心。

    起始点 = torch.randint（与上游实现一致，全局 torch 种子由 BaseModel.train
    的 manual_seed(cfg.seed) 驱动）；SRP 由 seed 显式定随机矩阵（确定）。
    SRP 库缺失时退 _kcenter_greedy 原始特征路径。
    """
    m = feats.shape[0]
    n_select = max(1, min(n_select, m))
    factory = _make_srp(seed)
    if factory is None:
        return _kcenter_greedy(feats, n_select, np.random.default_rng(seed))
    proj = factory().fit(feats).transform(feats)        # (M, D')
    min_dist: torch.Tensor | None = None
    selected: list[int] = []
    idx = int(torch.randint(high=m, size=(1,)).item())  # 与上游实现一致的起始
    for _ in range(n_select):
        d = torch.cdist(proj[idx:idx + 1], proj).reshape(-1)
        min_dist = d if min_dist is None else torch.minimum(min_dist, d)
        selected.append(idx)
        min_dist[selected[-1]] = 0
        idx = int(torch.argmax(min_dist).item())
        if min_dist[idx].item() <= 0:
            break
    return torch.tensor(sorted(set(selected)), dtype=torch.long)


def _sample_nodes(node_bank: torch.Tensor, num_clusters: int,
                  sampling_ratio: float, seed: int,
                  max_nodes: int = MAX_MEMORY_NODES,
                  ) -> torch.Tensor:
    """按簇 K-Center-Greedy 共采样（上游 sampling.sample_nodes 对齐版）。

    node_bank (M,769) 末列簇 id；目标 = min(总量×ratio, max_nodes)；
    n_sel==1 取 medoid（与上游原始 cdist 分支一致），多选走 SRP 贪心。
    """
    total = node_bank.shape[0]
    target = min(max(1, int(total * sampling_ratio)), max_nodes, total)
    picked: list[torch.Tensor] = []
    for cid in range(num_clusters):
        cluster = node_bank[node_bank[:, -1] == float(cid)]
        if cluster.shape[0] == 0:
            continue
        n_sel = max(1, int(round(cluster.shape[0] / total * target)))
        if cluster.shape[0] == 1 or n_sel == 1:
            idx = _kcenter_greedy(cluster[:, :-1], 1,
                                  np.random.default_rng(seed))
        else:
            idx = _kcenter_greedy_srp(cluster[:, :-1], n_sel, seed)
        picked.append(cluster[idx])
    return torch.cat(picked, dim=0) if picked else node_bank


class _HyperMemoryBank:
    """超图记忆库（上游 memory.HyperMemoryBank 移植：节点+簇 id、边特征）。"""

    def __init__(self, dim: int, num_clusters: int):
        self.dim = dim
        self.num_clusters = num_clusters
        self.node_bank: list[torch.Tensor] = []
        self.edge_bank: torch.Tensor | None = None

    def update(self, node_feat: torch.Tensor, edge_feat: torch.Tensor,
               adj: torch.Tensor) -> None:
        """node_feat (C,N)、edge_feat (K,C)、adj (N,K)——上游 update 同语义。"""
        node_feat = node_feat.T
        valid = adj.max(dim=1).values > 0
        node_feat = node_feat[valid]
        adj = adj[valid]
        if node_feat.shape[0] == 0:
            return
        if self.edge_bank is None:
            ids = torch.argmax(adj, dim=1).unsqueeze(1).float()
            self.node_bank.append(torch.cat([node_feat, ids], dim=1))
            self.edge_bank = edge_feat.clone()
        else:
            sim = F.cosine_similarity(edge_feat.unsqueeze(1),
                                      self.edge_bank.unsqueeze(0), dim=2)
            matched = torch.argmax(sim, dim=1)
            old_ids = torch.argmax(adj, dim=1)
            ids = matched[old_ids].unsqueeze(1).float()
            self.node_bank.append(torch.cat([node_feat, ids], dim=1))
            self.edge_bank = self._edge_from_memory()

    def _edge_from_memory(self, eps: float = 1e-6) -> torch.Tensor:
        """由当前记忆库重算边特征（上游 _compute_edge_features_from_memory）。"""
        bank = torch.cat(self.node_bank, dim=0)
        node_feat = bank[:, :-1]
        ids = bank[:, -1].long()
        h = torch.zeros(node_feat.shape[0], self.num_clusters)
        h[torch.arange(node_feat.shape[0]), ids] = 1.0
        d_he = 1.0 / torch.clamp(h.sum(dim=0), min=eps)
        d_hv = 1.0 / torch.sqrt(torch.clamp(h.sum(dim=1), min=eps))
        return torch.diag(d_he) @ h.T @ torch.diag(d_hv) @ node_feat

    def sample(self, sampling_ratio: float, seed: int) -> None:
        """共采样收缩（上游 mem_sampling）。"""
        if not self.node_bank:
            return
        self.node_bank = [_sample_nodes(torch.cat(self.node_bank, dim=0),
                                        self.num_clusters, sampling_ratio, seed)]

    def get_nodes(self) -> torch.Tensor:
        return torch.cat(self.node_bank, dim=0)

    def __len__(self) -> int:
        return sum(t.shape[0] for t in self.node_bank)


class CifModel(BaseModel):
    """CIF：DINO 特征 + 模糊超图记忆库 + Bi-TF-MP 检索（免训练式，OK 图）。"""

    name = "cif"
    family = "folder"
    supported_tasks = (TaskType.ANOMALY,)
    default_image_size = 224                      # pos_embed 定长，固定 224

    def __init__(self, num_classes: int = 1, device: str = "auto",
                 use_pretrained: bool = True):
        super().__init__(num_classes, device)
        self.task = TaskType.ANOMALY
        # §三（V0.7.3）：图像分 = OK 分布标定 z（无上界），conf→z=1/conf−1
        self.score_semantics = {
            "key": "z", "name": "图像分数", "range": [0.0, None],
            "higher": "分数越高越异常（OK 分布 z 标定）",
        }
        # §四（V0.7.3）：免训练范式声明（UI 隐藏轮次类参数、阶段化进度）
        self.train_mode = "free"
        self.image_size = self.default_image_size
        # DINO 权重缺失在此即抛（创建模型时用户即见提示，训练前不空转）
        self.vit, self.dino_sha = load_dino_vit(device=self.device)
        self.memory: torch.Tensor | None = None   # (M, 769) fp32（末列簇 id）
        self.map_mu = 0.0
        self.map_sigma = 1.0
        self.score_mu = 0.0
        self.score_sigma = 1.0
        self.params = {
            "n_clusters": N_CLUSTERS, "threshold": SAHC_THRESHOLD,
            "sampling_ratio": SAMPLING_RATIO, "layers": BITFMP_LAYERS,
            "alpha": BITFMP_ALPHA, "edge_topk": EDGE_TOPK,
            "n_neighbors": N_NEIGHBORS,
        }
        self.classes = ["异常"]
        self._mem_cache: tuple | None = None      # (mem_feats, mem_adj, mem_edges)
        self.bi_tfmp = _BiTFMP(layers=BITFMP_LAYERS, alpha=BITFMP_ALPHA, topk=1)

    # ---- 预处理 ----
    def _prepare(self, img_bgr: np.ndarray) -> torch.Tensor:
        size = self.image_size
        rgb = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
        x = rgb[:, :, ::-1].astype(np.float32) / 255.0
        x = (x - np.array(_IMAGENET_MEAN)) / np.array(_IMAGENET_STD)
        t = torch.from_numpy(x.astype(np.float32)).permute(2, 0, 1)
        return t.unsqueeze(0).to(self.device)

    def _extract(self, img_bgr: np.ndarray) -> torch.Tensor:
        """(BGR 图) → (1,768,784) patch 特征。"""
        return self.vit(self._prepare(img_bgr))

    # ---- 记忆库派生量（fit 后/加载后建一次）----
    def _memory_views(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(mem_feats (M,768) fp32 CPU, mem_adj (M,E), mem_edges (E,768))。

        缓存常驻 CPU（省显存）；_score_map 用时搬设备。
        """
        if self._mem_cache is None:
            if self.memory is None or self.memory.shape[0] == 0:
                raise RuntimeError("异常检测模型未训练（记忆库为空）")
            bank = self.memory.float()
            mem_adj, bank_new = _build_adj(bank)
            mem_feats_col = bank_new[:, :-1].T                 # (C,M)
            d_he = torch.diag_embed(1.0 / torch.clamp(mem_adj.sum(dim=0), min=1e-12))
            d_hv = torch.diag_embed(1.0 / torch.sqrt(torch.clamp(mem_adj.sum(dim=1), min=1e-12)))
            mem_edges = d_he @ mem_adj.T @ d_hv @ mem_feats_col.T   # (E,C)
            self._mem_cache = (bank[:, :-1], mem_adj, mem_edges)
        return self._mem_cache

    # ---- 单图评分（标定与预测共用）----
    def _score_map(self, img_bgr: np.ndarray,
                   ) -> tuple[np.ndarray, float, torch.Tensor, torch.Tensor]:
        """返回 (平滑后异常图 (224,224) float, 图像级原始分, init 分数, locations)。"""
        feats = self._extract(img_bgr)                           # (1,768,N)
        adj, hyperedge = _construct_hypergraph(feats, None, N_CLUSTERS,
                                               SAHC_THRESHOLD)
        bank = self.memory.to(self.device).float()
        _, mem_adj, mem_edges = self._memory_views()
        mem_adj = mem_adj.to(self.device)
        edges_ids = torch.cat([mem_edges.to(self.device), torch.arange(
            mem_edges.shape[0], device=self.device).unsqueeze(1).float()], dim=1)
        feats_mp, hyperedge_mp = self.bi_tfmp.forward(
            feats, adj, bank[:, :-1], mem_adj)
        flat = feats.permute(0, 2, 1).reshape(-1, feats.shape[1])
        init_scores, locations = _nearest_neighbors(
            feats_mp.permute(0, 2, 1).reshape(-1, feats.shape[1]),
            bank[:, :-1])
        patch_scores = _hyper_search(feats_mp, adj, hyperedge_mp,
                                     bank, edges_ids, topk=EDGE_TOPK)
        score_mix = (patch_scores * init_scores.unsqueeze(0)).reshape(1, 1, 28, 28)
        img_score = _compute_anomaly_score(
            patch_scores, locations.reshape(1, -1), flat, bank[:, :-1])
        score = F.interpolate(score_mix, size=self.image_size,
                              mode="bilinear", align_corners=True)
        score = score.pow(1.5)
        for kernel, pad, reps in (_SMOOTH_L, _SMOOTH_U):
            w = torch.ones(1, 1, kernel, kernel, device=score.device) / (kernel * kernel)
            for _ in range(reps):
                score = F.conv2d(score, w, padding=pad)
        return (score[0, 0].cpu().numpy(), float(img_score[0]),
                init_scores, locations)

    # ---- 训练 ----
    def fit(self, data, cfg: TrainConfig, reporter: TrainReporter | None = None) -> dict:
        reporter = reporter or TrainReporter()
        self.image_size = self.default_image_size     # CIF 固定 224（论文 §2.3 口径）
        ok_labels = {c for c in (data.classes or []) if is_negative_dir_name(c)}
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
        started = datetime.now()
        rng = np.random.default_rng(cfg.seed)

        # 1) 留出标定图（≥3 张才独立标定，否则样本内标定并打点）
        n_calib = max(0, int(round(len(pool) * _CALIB_RATIO)))
        if len(pool) >= 8 and n_calib < 3:
            n_calib = min(3, len(pool) // 4)
        order = rng.permutation(len(pool))
        calib = [pool[i] for i in order[:n_calib]]
        train_pool = [pool[i] for i in order[n_calib:]]

        # 2) 逐图特征 → SAHC → 记忆库累积
        bank = _HyperMemoryBank(dim=768, num_clusters=N_CLUSTERS)
        n_img = len(train_pool)
        for i, s in enumerate(train_pool, start=1):
            img = imread_unicode(s.image_path)
            if img is None:
                continue
            feats = self._extract(img)
            adj, x_e = _construct_hypergraph(feats, None, N_CLUSTERS,
                                             SAHC_THRESHOLD)
            bank.update(feats[0].cpu(), x_e[0].cpu(), adj[0].cpu())
            if reporter is not None:
                reporter.on_train_progress(1, i, n_img)
            if i % 20 == 0:
                log.info("CIF 特征抽取 %d/%d（记忆库 %d 节点）", i, n_img, len(bank))
        if not bank.node_bank:
            raise ValueError("OK 图特征抽取失败（图片全部无法读取）")
        bank.sample(SAMPLING_RATIO, int(cfg.seed))
        self.memory = bank.get_nodes()
        log.info("CIF 记忆库构建完成: %d 节点（%d 超边, 采样比例 %.2f）",
                 self.memory.shape[0], N_CLUSTERS, SAMPLING_RATIO)
        self._mem_cache = None

        # 3) z 标定（图像级 + 图级）：优先留出图，不足退训练池样本内
        calib_src = calib if len(calib) >= 3 else train_pool[:min(len(train_pool), 200)]
        if len(calib) < 3:
            log.info("CIF 标定源不足（留出 %d 张），退训练池样本内标定", len(calib))
        map_vals, ok_scores = [], []
        _n_cal = len(calib_src)
        for _ci, s in enumerate(calib_src, start=1):
            img = imread_unicode(s.image_path)
            if img is None:
                continue
            smap, sc, _, _ = self._score_map(img)
            map_vals.append(smap.reshape(-1))
            ok_scores.append(sc)
            if reporter is not None:      # §四 阶段化进度：2=标定
                reporter.on_train_progress(2, _ci, _n_cal)
        if map_vals:
            all_map = np.concatenate(map_vals)
            self.map_mu = float(np.mean(all_map))
            self.map_sigma = max(float(np.std(all_map)), 1e-6)
        if len(ok_scores) >= 5:
            self.score_mu = float(np.mean(ok_scores))
            self.score_sigma = max(float(np.std(ok_scores)), 1e-6)

        # 4) 验证集指标（NG + OK/留出 OK）
        scores, labels = [], []
        _n_val = max(len(val_ng) + len(val_ok), 1)
        _v_done = 0
        for s in val_ng:
            img = imread_unicode(s.image_path)
            if img is not None:
                scores.append(self._score_map(img)[1])
                labels.append(1)
            _v_done += 1
            if reporter is not None:      # §四 阶段化进度：3=验证评估
                reporter.on_train_progress(3, _v_done, _n_val)
        for s in val_ok:
            img = imread_unicode(s.image_path)
            if img is not None:
                scores.append(self._score_map(img)[1])
                labels.append(0)
            _v_done += 1
            if reporter is not None:
                reporter.on_train_progress(3, _v_done, _n_val)
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
            "memory_nodes": int(self.memory.shape[0]),
            "memory_clusters": N_CLUSTERS,
            "support_images": len(train_pool),
            "calib_images": len(calib),
            "val_ng_images": len(val_ng),
            "val_ok_images": len(val_ok),
            "image_size": self.image_size,
            "dino_fingerprint": self.dino_sha,
            "params": dict(self.params),
            "history": [{"epoch": 1, "train_loss": 0.0, "auroc": round(auroc, 4)}],
        }
        metrics["start_time"] = started.strftime("%Y-%m-%d %H:%M:%S")
        metrics["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics["status"] = "completed"
        reporter.on_epoch(1, {"auroc": auroc})
        reporter.on_finish(True)
        return metrics

    # ---- 推理 ----
    def predict(self, image_bgr: np.ndarray, conf: float = 0.25,
                iou: float = 0.45) -> Prediction:
        smap, img_score, _, _ = self._score_map(image_bgr)
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
        """xccv-cif 容器（明文 torch.save）。

        冻结骨干不进容器（330MB×每模型不可接受）：只记指纹，加载时按
        weights/dino 落点重建（设计文档 §四）。
        """
        ckpt = {
            "format": "xccv-cif",
            "classes": self.classes,
            "image_size": self.image_size,
            "memory": self.memory.cpu().to(torch.float16) if self.memory is not None else None,
            "map_mu": self.map_mu, "map_sigma": self.map_sigma,
            "score_mu": self.score_mu, "score_sigma": self.score_sigma,
            "params": dict(self.params),
            "dino_sha": self.dino_sha,
        }
        torch.save(ckpt, path)

    def load_weights(self, source) -> None:
        """source：权重路径 / ckpt dict（registry 统一入口语义）。"""
        if isinstance(source, dict):
            ckpt = source
        else:
            ckpt = torch.load(source, map_location="cpu", weights_only=False)
        if ckpt.get("format") != "xccv-cif":
            raise ValueError("不是 xccv-cif 权重文件")
        self.classes = list(ckpt.get("classes") or ["异常"])
        self.image_size = int(ckpt.get("image_size") or self.default_image_size)
        params = dict(ckpt.get("params") or {})
        self.params.update(params)
        self.bi_tfmp = _BiTFMP(layers=params.get("layers", BITFMP_LAYERS),
                               alpha=params.get("alpha", BITFMP_ALPHA), topk=1)
        mem = ckpt.get("memory")
        self.memory = mem.cpu().float() if mem is not None else None
        self.map_mu = float(ckpt.get("map_mu", 0.0))
        self.map_sigma = max(float(ckpt.get("map_sigma", 1.0)), 1e-6)
        self.score_mu = float(ckpt.get("score_mu", 0.0))
        self.score_sigma = max(float(ckpt.get("score_sigma", 1.0)), 1e-6)
        self._mem_cache = None
        sha = str(ckpt.get("dino_sha") or "")
        if sha and not fingerprint_matches(sha):
            log.warning("DINO 权重指纹与训练时不一致（%s ≠ 当前权重），"
                        "特征空间可能漂移，建议用同权重文件复训", sha[:16])



