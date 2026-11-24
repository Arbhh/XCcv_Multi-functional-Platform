"""训练平台测试：3 个历史测试模块的合并模块。

由以下模块按域合并而成，用例逐段原样迁移、断言未改：
- test_v073_cif
- test_v07_a1_anomaly
- test_v07_a2_ocr
"""


# == 来源模块：test_v073_cif ==
"""CIF 异常检测后端回归锁。

分层：
- 核心数学层：无 DINO 权重、CPU 秒级（合成小张量直接断言）；
- 集成层：skipUnless DINO 权重在位（weights/dino 或 XCCV_DINO_CKPT），
  合成纹理数据 fit → NG 分高 → 亮块出框 → best.pt 回读。
"""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from xccv.engine.dataset.base import TaskType
from xccv.engine.models.cif import (
    _BiTFMP,
    _HyperMemoryBank,
    _build_adj,
    _compute_anomaly_score,
    _construct_hypergraph,
    _euclidean_dist,
    _kcenter_greedy,
    _kmeans,
    _nearest_neighbors,
    _hyper_search,
    _sample_nodes,
    _z_from_conf,
)
from xccv.engine.models.dino_vit import find_dino_checkpoint
from xccv.engine.models.registry import list_models

_IMAGE_SIZE = 224


# ============================================================
# 核心数学层（无 DINO 权重）
# ============================================================
class CoreMathTest(unittest.TestCase):

    def test_z_from_conf(self):
        self.assertAlmostEqual(_z_from_conf(0.25), 3.0, places=6)
        self.assertGreater(_z_from_conf(0.1), _z_from_conf(0.5))

    def test_euclidean_dist(self):
        x = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
        d = _euclidean_dist(x, x)
        # 对角 ≈ 0（展开式 |x|²−2xyᵀ+|y|² 有 sqrt 浮点残差，上游同式）；
        # 非对角 = 3-4-5 三角形斜边
        self.assertTrue(bool((torch.abs(torch.diagonal(d)) < 1e-3).all()))
        self.assertAlmostEqual(float(d[0, 1]), 5.0, places=4)
        d2 = _euclidean_dist(torch.tensor([[0.0, 0.0]]),
                             torch.tensor([[3.0, 4.0]]))
        self.assertAlmostEqual(float(d2[0, 0]), 5.0, places=5)

    def test_kmeans_shapes_and_determinism(self):
        rng = np.random.default_rng(0)
        feats = np.vstack([rng.normal(0, .1, (30, 8)) + 10,
                           rng.normal(0, .1, (30, 8)) - 10]).astype(np.float32)
        labels, centers = _kmeans(feats, 2)
        self.assertEqual(labels.shape, (60,))
        self.assertEqual(centers.shape, (2, 8))
        labels2, centers2 = _kmeans(feats, 2)
        self.assertTrue(np.array_equal(labels, labels2))
        self.assertTrue(np.allclose(centers, centers2))
        # 两簇分别对应 +10 / -10 两组
        means = [float(feats[labels == i, 0].mean()) for i in range(2)]
        self.assertLess(min(means), -9.0)
        self.assertGreater(max(means), 9.0)

    def test_kmeans_k_larger_than_points(self):
        labels, centers = _kmeans(np.ones((2, 4), dtype=np.float32), 4)
        self.assertEqual(centers.shape, (2, 4))   # k 收缩到点数

    def test_construct_hypergraph_membership(self):
        rng = np.random.default_rng(1)
        feats = torch.randn(1, 8, 16)
        adj, x_e = _construct_hypergraph(feats, None, 4, 0.9)
        self.assertEqual(tuple(adj.shape), (1, 16, 4))
        self.assertEqual(tuple(x_e.shape), (1, 4, 8))
        # 按行 min-max 归一 ⇒ 每个有效 patch 至少属于 1 个超边
        self.assertTrue(bool((adj[0].sum(dim=1) >= 1).all()))

    def test_construct_hypergraph_background_zeroed(self):
        feats = torch.randn(1, 8, 16)
        mask = torch.zeros(1, 1, 4, 4)
        mask[0, 0, 0, 0] = 1.0                     # 仅 2 个前景 patch
        mask[0, 0, 0, 1] = 1.0
        adj, _ = _construct_hypergraph(feats, mask, 4, 0.9)
        self.assertEqual(float(adj[0, 2:].sum()), 0.0)        # 背景全零
        self.assertGreaterEqual(float(adj[0, :2].sum()), 2.0)  # 前景至少各 1

    def test_bitfmp_shapes_and_cross_propagation(self):
        torch.manual_seed(3)
        feats = torch.randn(1, 8, 16)
        adj, _ = _construct_hypergraph(feats, None, 2, 0.9)
        mem_x = torch.randn(24, 8)
        mem_adj = torch.zeros(24, 2)
        mem_adj[:12, 0] = 1.0
        mem_adj[12:, 1] = 1.0
        mp = _BiTFMP(layers=2, alpha=0.5, topk=1)
        x_hat, x_e = mp.forward(feats, adj, mem_x, mem_adj)
        self.assertEqual(tuple(x_hat.shape), feats.shape)
        self.assertEqual(tuple(x_e.shape), (1, 2, 8))
        self.assertTrue(bool(torch.isfinite(x_hat).all()))
        # 跨库传播应改变节点特征（记忆库特征独立随机 → 有信息流入）
        self.assertGreater(float((x_hat - feats).abs().mean()), 1e-4)

    def test_build_adj_remaps_ids(self):
        bank = torch.randn(5, 3)
        ids = torch.tensor([2.0, 2.0, 0.0, 7.0, 7.0]).unsqueeze(1)
        adj, new_bank = _build_adj(torch.cat([bank, ids], dim=1))
        self.assertEqual(tuple(adj.shape), (5, 3))
        self.assertTrue(bool((adj.sum(dim=1) == 1).all()))
        self.assertEqual(set(new_bank[:, -1].tolist()), {0.0, 1.0, 2.0})

    def test_nearest_neighbors_foreground(self):
        emb = torch.randn(4, 6)
        mem = torch.randn(10, 6)
        scores, locs = _nearest_neighbors(emb, mem)
        self.assertEqual(tuple(scores.shape), (4,))
        # 背景 patch（掩码 0）分数恒 0
        mask = torch.tensor([1, 1, 0, 1], dtype=torch.bool)
        s2, l2 = _nearest_neighbors(emb, mem, fore_mask=mask)
        self.assertEqual(float(s2[2]), 0.0)
        self.assertEqual(int(l2[2]), 0)

    def test_hyper_search_cluster_matching(self):
        torch.manual_seed(5)
        d = 4
        n1 = torch.randn(2, d)                # 簇 0 节点
        n2 = torch.randn(2, d)                # 簇 1 节点
        feats = torch.cat([n1, n2]).T.unsqueeze(0)          # (1,d,4)
        adj = torch.zeros(1, 4, 2)
        adj[0, :2, 0] = 1.0
        adj[0, 2:, 1] = 1.0
        # 超边/记忆边特征用正交基构造 → 检索确定性（簇 0↔记忆 0、簇 1↔记忆 1）
        hyperedge = torch.zeros(1, 2, d)
        hyperedge[0, 0, 0] = 1.0
        hyperedge[0, 1, 1] = 1.0
        mem0 = n1 + 0.01                      # 簇 0 记忆贴近（分小）
        mem1 = n2 + 50.0                      # 簇 1 记忆远离（分大）
        total_feats = torch.cat([
            torch.cat([mem0, torch.zeros(2, 1)], dim=1),
            torch.cat([mem1, torch.ones(2, 1)], dim=1)])
        edges = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0, 1.0]])    # (E=2, D+1) 末列簇 id
        scores = _hyper_search(feats, adj, hyperedge, total_feats, edges, topk=1)
        self.assertEqual(tuple(scores.shape), (1, 4))
        self.assertLess(float(scores[0, :2].mean()), float(scores[0, 2:].mean()))

    def test_compute_anomaly_score_bounds(self):
        torch.manual_seed(6)
        b, n, d, m = 1, 4, 8, 12
        patch_scores = torch.tensor([[0.1, 0.2, 5.0, 0.3]])
        locations = torch.randint(0, m, (b, n))
        embedding = torch.randn(b * n, d)
        memory = torch.randn(m, d)
        out = _compute_anomaly_score(patch_scores, locations, embedding, memory)
        self.assertEqual(tuple(out.shape), (b,))
        self.assertGreater(float(out[0]), 0.0)
        self.assertLessEqual(float(out[0]), 5.0)

    def test_kcenter_greedy_medoid_and_coverage(self):
        rng = np.random.default_rng(2)
        feats = torch.randn(20, 6)
        one = _kcenter_greedy(feats, 1, rng)
        self.assertEqual(one.numel(), 1)
        dists = torch.cdist(feats, feats)
        medoid = int(torch.argmin(dists.max(dim=1).values).item())
        self.assertEqual(int(one[0]), medoid)
        picks = _kcenter_greedy(feats, 5, rng)
        self.assertEqual(picks.numel(), len(set(picks.tolist())))
        # 选中集能覆盖最远点：任一点到选中集距离 ≤ 最大点对距离
        sel = feats[picks]
        dmax = float(torch.cdist(feats, sel).min(dim=1).values.max())
        self.assertLessEqual(dmax, float(dists.max()) + 1e-5)

    def test_sample_nodes_caps_and_clusters(self):
        torch.manual_seed(7)
        bank = torch.cat([
            torch.cat([torch.randn(300, 16), torch.zeros(300, 1)], dim=1),
            torch.cat([torch.randn(100, 16), torch.ones(100, 1)], dim=1)])
        out = _sample_nodes(bank, 2, 0.5, 3, max_nodes=100)
        self.assertLessEqual(out.shape[0], 100)
        self.assertEqual(set(out[:, -1].tolist()), {0.0, 1.0})

    def test_sample_nodes_deterministic(self):
        torch.manual_seed(7)
        bank = torch.cat([
            torch.cat([torch.randn(300, 16), torch.zeros(300, 1)], dim=1),
            torch.cat([torch.randn(100, 16), torch.ones(100, 1)], dim=1)])
        torch.manual_seed(7)      # a/b 均从"建库后+复位"的同一 RNG 状态起跑
        a = _sample_nodes(bank, 2, 0.5, 3, max_nodes=100)
        torch.manual_seed(7)
        b = _sample_nodes(bank, 2, 0.5, 3, max_nodes=100)
        self.assertTrue(torch.equal(a, b))           # 同种子 ⇒ 同 coreset

    def test_hyper_memory_bank_update_and_sample(self):
        torch.manual_seed(8)
        bank = _HyperMemoryBank(dim=8, num_clusters=2)
        for _ in range(3):
            node = torch.randn(16, 8)
            edge = torch.randn(2, 8)
            adj = torch.zeros(16, 2)
            adj[:8, 0] = 1.0
            adj[8:, 1] = 1.0
            bank.update(node.T, edge, adj)
        self.assertEqual(len(bank), 48)
        self.assertIsNotNone(bank.edge_bank)
        bank.sample(0.5, 4)
        self.assertLess(len(bank), 48)
        nodes = bank.get_nodes()
        self.assertEqual(nodes.shape[1], 9)          # 8 维 + 簇 id


# ============================================================
# 集成层（DINO 权重在位时跑；缺失则整类跳过）
# ============================================================
_HAS_DINO = find_dino_checkpoint() is not None


def _texture(rng: np.random.Generator) -> np.ndarray:
    base = rng.normal(0, 1, (_IMAGE_SIZE // 8, _IMAGE_SIZE // 8)) * 40 + 120
    img = cv2.resize(base.astype(np.float32), (_IMAGE_SIZE, _IMAGE_SIZE),
                     interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


@unittest.skipUnless(_HAS_DINO, "DINO ViT-B/8 权重不在位（weights/dino）")
class CifIntegrationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from xccv.engine.models.base import TrainConfig
        from xccv.engine.models.registry import create_model

        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        ok_dir, ng_dir = root / "OK", root / "NG"
        ok_dir.mkdir(); ng_dir.mkdir()
        rng = np.random.default_rng(7)
        for i in range(6):
            cv2.imwrite(str(ok_dir / f"ok_{i}.png"), _texture(rng))
        for i in range(3):
            img = _texture(rng)
            img[40:70, 60:90] = 245
            cv2.imwrite(str(ng_dir / f"ng_{i}.png"), img)
        cls.model = create_model("cif", TaskType.ANOMALY, 2,
                                 device="cpu", use_pretrained=True)
        cls.cfg = TrainConfig(
            task=TaskType.ANOMALY, dataset_dir=root,
            output_dir=root / "run", model_name="cif",
            epochs=1, device="cpu", workers=0, seed=7, use_pretrained=True)
        cls.metrics = cls.model.train(cls.cfg, reporter=None)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_registry_lists_cif(self):
        names = [m["name"] for m in list_models(TaskType.ANOMALY)]
        self.assertIn("cif", names)
        self.assertIn("patchcore", names)

    def test_metrics_structure(self):
        self.assertGreaterEqual(self.metrics["final"]["auroc"], 0.9)
        self.assertGreater(self.metrics["memory_nodes"], 0)
        self.assertEqual(self.metrics["memory_clusters"], 4)
        self.assertEqual(self.metrics["image_size"], 224)
        self.assertTrue(self.metrics.get("dino_fingerprint"))

    def test_weights_saved_and_reloaded(self):
        from xccv.engine.models import load_model

        xva = Path(self.cfg.output_dir) / "best.pt"
        self.assertTrue(xva.is_file(), "训练应收尾产出 best.pt")
        m2 = load_model(xva, device="cpu")
        self.assertEqual(type(m2).__name__, "CifModel")
        self.assertGreater(m2.memory.shape[0], 0)

    def test_ng_scores_higher_and_boxes(self):
        rng = np.random.default_rng(99)
        ok_scores = [self.model.predict(_texture(rng)).probs[0] for _ in range(2)]
        ng_scores, n_boxes = [], 0
        for _ in range(2):
            img = _texture(rng)
            img[40:70, 60:90] = 245
            pred = self.model.predict(img)
            ng_scores.append(pred.probs[0])
            n_boxes += len(pred.boxes)
        self.assertGreater(min(ng_scores), max(ok_scores),
                           f"NG 分数应全面高于 OK：ok={ok_scores} ng={ng_scores}")
        self.assertGreater(n_boxes, 0, "异常亮块应产出检出框")

    def test_predict_contract(self):
        img = _texture(np.random.default_rng(11))
        pred = self.model.predict(img, conf=0.25)
        self.assertEqual(pred.labels, [0] * len(pred.boxes))
        if pred.boxes:
            self.assertEqual(len(pred.masks), len(pred.boxes))
        self.assertEqual(len(pred.probs), 1)


class CollectStatsAnomalyTest(unittest.TestCase):
    """P252 回归锁：ANOMALY 数据集（OK/NG 子目录纯图）统计必须非零。"""

    def test_collect_stats_counts_plain_image_dirs(self):
        import tempfile

        from xccv.training.dataset_stats import collect_stats

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ok_dir, ng_dir = root / "OK", root / "NG"
            ok_dir.mkdir(); ng_dir.mkdir()
            rng = np.random.default_rng(5)
            for i in range(3):
                cv2.imwrite(str(ok_dir / f"ok_{i}.png"), _texture(rng))
            img = _texture(rng)
            img[40:70, 60:90] = 245
            cv2.imwrite(str(ng_dir / "ng_0.png"), img)
            stats = collect_stats("anomaly", root)
            self.assertEqual(stats.n_images, 4)
            self.assertEqual(sorted(stats.classes), ["NG", "OK"])
            self.assertEqual(stats.class_counts["OK"], 3)
            self.assertEqual(stats.class_counts["NG"], 1)
            self.assertEqual(stats.n_ok, 3)
            self.assertEqual(stats.n_labeled, 0)

# == 来源模块：test_v07_a1_anomaly ==
"""V0.7 A1：无监督异常检测（PatchCore-lite）回归锁。

合成数据（纯噪声纹理 OK vs 同纹理+亮块 NG）+ 真实 resnet18 离线权重：
训练收敛 → OK 分低 NG 分高 → 异常区命中。CPU 跑得动（image_size 128）。
"""

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from xccv.engine.dataset.base import TaskType
from xccv.engine.dataset.prepare import prepare
from xccv.engine.metrics.batch_match import collect_match_stats  # noqa: F401
from xccv.engine.models.anomaly import (
    AnomalyModel,
    _z_from_conf,
    image_ap,
    image_auroc,
)
from xccv.engine.models.registry import create_model, list_models

_IMAGE_SIZE__v07_a1_anomaly = 128
_N_OK = 6
_N_NG = 4


def _texture__v07_a1_anomaly(rng) -> np.ndarray:
    """固定纹理（平滑噪声，模拟均质工业表面，BGR 3 通道）。"""
    base = rng.normal(0, 1, (_IMAGE_SIZE__v07_a1_anomaly // 8, _IMAGE_SIZE__v07_a1_anomaly // 8)) * 40 + 120
    img = cv2.resize(base.astype(np.float32), (_IMAGE_SIZE__v07_a1_anomaly, _IMAGE_SIZE__v07_a1_anomaly),
                     interpolation=cv2.INTER_CUBIC)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _make_dataset(root: Path, seed: int = 7):
    rng = np.random.default_rng(seed)
    ok_dir = root / "OK"
    ng_dir = root / "NG"
    ok_dir.mkdir(parents=True)
    ng_dir.mkdir(parents=True)
    for i in range(_N_OK):
        img = _texture__v07_a1_anomaly(rng)
        cv2.imwrite(str(ok_dir / f"ok_{i}.png"), img)
    for i in range(_N_NG):
        img = _texture__v07_a1_anomaly(rng)
        x, y = 40 + i * 8, 40
        img[y:y + 28, x:x + 28] = 245          # 亮块异常
        cv2.imwrite(str(ng_dir / f"ng_{i}.png"), img)
    return ok_dir, ng_dir


class AnomalyEngineTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        _make_dataset(root)
        cls.model = create_model("patchcore", TaskType.ANOMALY, 2,
                                 device="cpu", use_pretrained=True)
        cfg = cls.model._make_test_cfg(root) if hasattr(
            cls.model, "_make_test_cfg") else None
        from xccv.engine.models.base import TrainConfig

        cfg = TrainConfig(
            task=TaskType.ANOMALY, dataset_dir=root,
            output_dir=root / "run", model_name="patchcore",
            epochs=1, image_size=_IMAGE_SIZE__v07_a1_anomaly, device="cpu", workers=0,
            seed=7, use_pretrained=True)
        cls.cfg = cfg
        cls.metrics = cls.model.train(cfg, reporter=None)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_registry_and_listing(self):
        self.assertIsInstance(self.model, AnomalyModel)
        names = [m["name"] for m in list_models(TaskType.ANOMALY)]
        self.assertIn("patchcore", names)

    def test_weights_saved_and_reloaded(self):
        xva = Path(self.cfg.output_dir) / "best.pt"
        self.assertTrue(xva.is_file(), "训练应收尾产出 best.pt")
        from xccv.engine.models import load_model

        m2 = load_model(xva, device="cpu")
        self.assertIsInstance(m2, AnomalyModel)
        self.assertEqual(m2.image_size, _IMAGE_SIZE__v07_a1_anomaly)
        self.assertGreater(m2.memory.shape[0], 0)

    def test_ok_scores_lower_than_ng(self):
        """验证：NG 图像级 z 分数显著高于 OK 图（异常检出能力）。"""
        rng = np.random.default_rng(99)
        ok_scores, ng_scores = [], []
        for i in range(3):
            ok_scores.append(self.model.predict(_texture__v07_a1_anomaly(rng)).probs[0])
            img = _texture__v07_a1_anomaly(rng)
            img[40:70, 60:90] = 245
            ng_scores.append(self.model.predict(img).probs[0])
        self.assertGreater(min(ng_scores), max(ok_scores),
                           f"NG 分数应全面高于 OK：ok={ok_scores} ng={ng_scores}")

    def test_predict_boxes_on_anomaly_region(self):
        img = _texture__v07_a1_anomaly(np.random.default_rng(1))
        img[40:80, 40:80] = 250
        pred = self.model.predict(img, conf=0.25)
        self.assertTrue(len(pred.boxes) > 0, "异常亮块应产出检出框")
        x1, y1, x2, y2 = pred.boxes[0]
        self.assertTrue(x1 < 100 and y1 < 100 and x2 > 30 and y2 > 30,
                        f"检出框应命中亮块区域: {pred.boxes[0]}")
        self.assertTrue(pred.masks is not None and len(pred.masks) == len(pred.boxes))
        self.assertEqual(pred.labels, [0] * len(pred.boxes))

    def test_metrics_have_auroc(self):
        self.assertGreaterEqual(self.metrics["final"]["auroc"], 0.999,
                                "合成数据 AUROC 应接近 1")
        self.assertIn("history", self.metrics)

    def test_prepare_anomaly_family(self):
        data = prepare(TaskType.ANOMALY, "folder",
                       Path(self.cfg.dataset_dir), val_ratio=0.25, seed=7)
        self.assertTrue(data.classes)            # OK/NG 目录名成类
        self.assertTrue(data.train_samples)
        self.assertTrue(data.val_samples)

    def test_z_from_conf_monotonic(self):
        self.assertAlmostEqual(_z_from_conf(0.25), 3.0, places=6)
        self.assertGreater(_z_from_conf(0.1), _z_from_conf(0.5))


class ScoreUtilsTest(unittest.TestCase):

    def test_auroc_perfect_and_degenerate(self):
        self.assertEqual(image_auroc([1, 2, 3, 4], [0, 0, 1, 1]), 1.0)
        self.assertEqual(image_auroc([1, 2, 3, 4], [1, 1, 1, 1]), 0.0)
        self.assertEqual(image_auroc([1.0, 1.0], [0, 1]), 0.5)   # 并列平均秩

    def test_ap_perfect(self):
        self.assertEqual(image_ap([9, 8, 1, 0.5], [1, 1, 0, 0]), 1.0)
        self.assertEqual(image_ap([1, 2], []), 0.0)

# == 来源模块：test_v07_a2_ocr ==
"""V0.7 A2：OCR 字符识别（CRNN+CTC）回归锁。

合成数据：PIL 默认字体在固定位置画文本 + 同位置 rectangle 标注
（label=字符串内容）。小字符集 + 固定渲染，短训练可记忆——
断言 loss 收敛 + 精确匹配率阈值 + best.pt 权重回环 + JSON text 键。
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from xccv.engine.dataset.base import TaskType
from xccv.engine.dataset.prepare import prepare
from xccv.engine.models.base import Prediction
from xccv.engine.models.registry import create_model, list_models, load_model
from xccv.engine.models.ocr import OcrModel, _levenshtein

_TEXTS = ["CAT", "DOG", "A12", "G55"]
_FONT = None


def _draw_sample(text: str, size=(200, 120)) -> tuple:
    """(BGR 图, rectangle 顶点)——文本画在 (40,40)。"""
    global _FONT
    if _FONT is None:
        _FONT = ImageFont.load_default()
    img = Image.new("RGB", size, (30, 30, 30))
    d = ImageDraw.Draw(img)
    d.text((40, 40), text, fill=(240, 240, 240), font=_FONT)
    bbox = d.textbbox((40, 40), text, font=_FONT)
    arr = np.array(img)[:, :, ::-1].copy()   # RGB→BGR
    points = [[float(bbox[0] - 2), float(bbox[1] - 2)],
              [float(bbox[2] + 2), float(bbox[1] - 2)],
              [float(bbox[2] + 2), float(bbox[3] + 2)],
              [float(bbox[0] - 2), float(bbox[3] + 2)]]
    return arr, points


class OcrEngineTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        rng = np.random.default_rng(3)
        for i in range(16):                       # 每文本 4 张
            text = _TEXTS[i % len(_TEXTS)]
            arr, pts = _draw_sample(text)
            img_path = root / f"img_{i:02d}.png"
            from xccv.engine.io_utils import imwrite_unicode

            imwrite_unicode(str(img_path), arr)
            (root / f"img_{i:02d}.json").write_text(json.dumps({
                "version": "5.8.1", "flags": {},
                "shapes": [{"label": text, "points": pts,
                            "shape_type": "rectangle"}],
                "imagePath": img_path.name, "imageData": None,
                "imageHeight": arr.shape[0], "imageWidth": arr.shape[1],
            }), encoding="utf-8")
        cls.model = create_model("crnn", TaskType.OCR, 1, device="cpu",
                                 use_pretrained=False)
        from xccv.engine.models.base import TrainConfig

        cfg = TrainConfig(task=TaskType.OCR, dataset_dir=root,
                          output_dir=root / "run", model_name="crnn",
                          epochs=30, learning_rate=2e-3, device="cpu",
                          workers=0, seed=3, use_pretrained=False)
        cls.cfg = cfg
        cls.metrics = cls.model.train(cfg, reporter=None)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_registry_and_listing(self):
        self.assertIsInstance(self.model, OcrModel)
        names = [m["name"] for m in list_models(TaskType.OCR)]
        self.assertIn("crnn", names)

    def test_training_converged(self):
        hist = self.metrics["history"]
        self.assertGreater(len(hist), 0)
        # epoch1 可能有负 CTC 数值噪声（BN+随机初始化，自愈型）——
        # 与第 3 轮比对收敛趋势，不与首轮比
        warm = hist[min(2, len(hist) - 1)]["train_loss"]
        self.assertLess(hist[-1]["train_loss"], warm,
                        f"CTC 损失应下降: {[h['train_loss'] for h in hist[:5]]}…")
        self.assertGreaterEqual(self.metrics["best_val_top1"] or 0.0, 0.75,
                                f"最优精确匹配率过低: {self.metrics['final']}")

    def test_weights_roundtrip(self):
        xva = Path(self.cfg.output_dir) / "best.pt"
        self.assertTrue(xva.is_file())
        m2 = load_model(xva, device="cpu")
        self.assertIsInstance(m2, OcrModel)
        self.assertEqual(m2.charset, self.model.charset)

    def test_predict_returns_text(self):
        arr, _ = _draw_sample("CAT")
        pred = self.model.predict(arr)
        self.assertEqual(len(pred.boxes), 1)
        self.assertIsInstance(pred.texts[0], str)
        self.assertGreater(len(pred.texts[0]), 0)

    def test_prepare_ocr_family(self):
        data = prepare(TaskType.OCR, "yolo", Path(self.cfg.dataset_dir),
                       val_ratio=0.25, seed=3)
        self.assertTrue(data.classes)            # 标注文本即"类别"

    def test_detection_rows_text_passthrough(self):
        from xccv.engine.result_json import build_result_json

        pred = Prediction(labels=[0], scores=[0.9],
                          boxes=[[0, 0, 10, 10]], texts=["CAT"])
        payload = build_result_json(task="ocr", source={"file": "a.png"},
                                    prediction=pred, classes=["文本"])
        self.assertEqual(payload["detections"][0]["text"], "CAT")

    def test_levenshtein(self):
        self.assertEqual(_levenshtein("CAT", "CAT"), 0)
        self.assertEqual(_levenshtein("CAT", "CAP"), 1)
        self.assertEqual(_levenshtein("", "AB"), 2)

if __name__ == '__main__':
    unittest.main()
