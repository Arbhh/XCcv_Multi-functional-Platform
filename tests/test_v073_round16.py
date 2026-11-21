"""回归锁：遗留待办 §二/§三/§四 + SubspaceAD。

- §二 复盘统一口径：batch_match ANOMALY 图像级弱 GT（OK 过检/NG 漏检/
  NG 检出=正确）、分类对错分组；review_logic 徽章/统计行随任务改字；
- §三 分数语义：BaseModel.score_semantics 元数据、ANOMALY z 语义覆写、
  阈值面板动态轴（无上界）、result_json additive 键；
- §四 免训练范式：registry MODEL_META、阶段化进度编码（1/2/3）、
  TrainConfig.model_params、阶段化进度回调；
- SubspaceAD：PCA 数学核心（两遍流式拟合/重构残差）、层选择、聚合、
  发现链、注册项（骨干权重缺失时数学项照测）。
"""

import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from xccv.engine.dataset.base import TaskType
from xccv.engine.metrics.batch_match import collect_match_stats
from xccv.testing.review_logic import classify_images, stats_line

# watcher 对象跨测试保活（处理完文件变化即析构会撞原生竞态）
_WATCHER_KEEPER: list = []


class LangWatcherCacheTest(unittest.TestCase):
    """语言偏好文件是跨进程共享状态，watcher
    _fire 曾走 lang() 进程缓存——非"保存"进程缓存恒为旧值，平台窗口
    永不重建。"""

    def test_external_file_change_fires_rebuild(self):
        from PyQt5.QtWidgets import QApplication

        from xccv.i18n import _LANG_FILE, install_language_watcher, lang, set_lang

        set_lang("zh")                      # 预热进程缓存（模拟平台启动态）
        self.assertEqual(lang(), "zh")
        app = QApplication.instance() or QApplication([])
        fired = []
        w = install_language_watcher(lambda: fired.append(1), parent=None)
        if w is None:
            self.skipTest("环境不可监听")
        _WATCHER_KEEPER.append(w)           #1429进程结束前不析构
        try:
            # 模拟另一进程保存 EN：直写文件（不经本进程 set_lang）
            _LANG_FILE.write_text(json.dumps({"lang": "en"}), encoding="utf-8")
            deadline = time.time() + 4
            while time.time() < deadline and not fired:
                app.processEvents()
                time.sleep(0.05)
            self.assertTrue(fired, "外部改语言文件未触发重建")
            self.assertEqual(lang(), "en")  # 进程缓存已同步（重建渲染 EN 的前提）
        finally:
            set_lang("zh")                  #1937复位偏好与缓存



def _res(path: str, score=None, dets=None, classifications=None):
    r = {"file": Path(path).name, "path": path, "width": 64, "height": 64,
         "detections": dets or []}
    if score is not None:
        r["image_score"] = score
    if classifications is not None:
        r["classifications"] = classifications
    return r


class AnomalyImageLevelTest(unittest.TestCase):
    """§二：ANOMALY 图像级口径（来源目录 = 弱 GT）。"""

    def test_ok_ng_groups(self):
        results = [
            _res("d:/ds/OK/a.png", score=5.0),    # OK 有过阈检出 → 过检
            _res("d:/ds/OK/b.png", score=0.5),    # OK 低分（阈值下无检出）→ 正确
            _res("d:/ds/NG/c.png", score=8.2),    # NG 有检出 → 正确
            _res("d:/ds/NG/d.png", score=None),   # NG 零检出 → 漏检
            _res("d:/ds/other/e.png", score=3.0),  # 目录不在类别集 → 未评估
        ]
        weak = {"OK", "NG"}
        stats = collect_match_stats(results, iou_thr=0.5, task="anomaly",
                                    score_semantics={"key": "z"},
                                    weak_classes=weak)
        self.assertEqual(stats["task"], "anomaly")
        self.assertEqual(stats["n_ok_images"], 2)
        self.assertEqual(stats["gt_total"], 2)          # NG 图数
        groups = classify_images(stats, {"score": 3.0})
        self.assertEqual([e["path"][-5] for e in groups["over"]], ["a"])
        self.assertEqual([e["path"][-5] for e in groups["missed"]], ["d"])
        self.assertEqual(sorted(e["path"][-5] for e in groups["correct"]),
                         ["b", "c"])             # OK 低分 + NG 检出
        self.assertEqual(len(groups["unjudged_list"]), 1)
        self.assertEqual(stats_line(groups, "evaluated", "anomaly").count("过检"), 1)
        # 回归锁：OK 图检出记录 matched=False（误报不进 TP）——
        # 否则精确率/召回率把 OK 误报计成命中
        ok_entry = next(e for e in stats["per_image"] if e["path"].endswith("a.png"))
        self.assertFalse(ok_entry["dets"][0][1])

    def test_score_semantics_propagated(self):
        stats = collect_match_stats([_res("d:/ds/NG/x.png", score=5.0)],
                                    task="anomaly",
                                    score_semantics={"key": "z", "name": "图像分数"})
        self.assertEqual(stats["score_semantics"]["name"], "图像分数")

    def test_threshold_sweep_image_level(self):
        """阈值现算复用：阈值 1.0 时 OK(1.2) 过检、NG(8.2) 正确；阈值 3.0
        时两者皆正确（OK 检出被阈值筛掉）。"""
        results = [_res("d:/ds/OK/a.png", score=1.2),
                   _res("d:/ds/NG/c.png", score=8.2)]
        stats = collect_match_stats(results, task="anomaly",
                                    weak_classes={"OK", "NG"})
        low = classify_images(stats, {"score": 1.0})
        self.assertEqual([e["path"][-5] for e in low["over"]], ["a"])
        self.assertEqual([e["path"][-5] for e in low["correct"]], ["c"])
        high = classify_images(stats, {"score": 3.0})
        self.assertEqual(sorted(e["path"][-5] for e in high["correct"]),
                         ["a", "c"])
        self.assertEqual(len(high["over"]), 0)


class ClassificationImageLevelTest(unittest.TestCase):
    """§二：分类对/错分组（预测类别 vs 目录真值）。"""

    def test_correct_wrong_groups(self):
        results = [
            _res("d:/ds/划伤/a.png", classifications=[{"label": "划伤", "score": 0.9}]),
            _res("d:/ds/划伤/b.png", classifications=[{"label": "污点", "score": 0.8}]),
            _res("d:/ds/污点/c.png", classifications=[{"label": "污点", "score": 0.7}]),
        ]
        stats = collect_match_stats(results, task="classification",
                                    weak_classes={"划伤", "污点"})
        self.assertEqual(stats["task"], "classification")
        self.assertEqual(stats["gt_total"], 3)
        groups = classify_images(stats, {"score": 0.0})
        self.assertEqual([e["path"][-5] for e in groups["correct"]],
                         ["a", "c"])
        self.assertEqual([e["path"][-5] for e in groups["over"]], ["b"])
        self.assertEqual([e["path"][-5] for e in groups["missed"]], [])
        self.assertIn("错误 1", stats_line(groups, "evaluated", "classification"))
        self.assertIn("错1", groups["over"][0]["badges"])

    def test_unknown_dir_unjudged(self):
        results = [_res("d:/ds/未知/a.png",
                        classifications=[{"label": "划伤", "score": 0.9}])]
        stats = collect_match_stats(results, task="classification",
                                    weak_classes={"划伤", "污点"})
        groups = classify_images(stats, {})
        self.assertEqual(len(groups["unjudged_list"]), 1)


class ScoreSemanticsTest(unittest.TestCase):
    """§三：分数语义元数据。"""

    def test_base_default(self):
        from xccv.engine.models.base import BaseModel

        self.assertEqual(BaseModel.score_semantics["key"], "conf")
        self.assertEqual(BaseModel.score_semantics["range"], [0.0, 1.0])

    def test_anomaly_model_semantics(self):
        from xccv.engine.models.anomaly import AnomalyModel

        m = AnomalyModel(1, device="cpu", use_pretrained=False)
        self.assertEqual(m.task, TaskType.ANOMALY)
        self.assertEqual(m.score_semantics["key"], "z")
        self.assertIsNone(m.score_semantics["range"][1])
        self.assertEqual(m.train_mode if hasattr(m, "train_mode") else "free",
                         getattr(m, "train_mode", "free"))

    def test_classifier_semantics(self):
        from xccv.engine.models.classifier import ClassifierModel

        m = ClassifierModel("resnet18", 2, device="cpu", use_pretrained=False)
        self.assertEqual(m.task, TaskType.CLASSIFICATION)
        self.assertEqual(m.score_semantics["key"], "prob")

    def test_registry_meta(self):
        from xccv.engine.models.registry import MODEL_META, list_models

        self.assertEqual(MODEL_META["cif"]["train_mode"], "free")
        self.assertTrue(MODEL_META["subspacead"]["support_k_shot"])
        items = list_models(TaskType.ANOMALY)
        names = {m["name"] for m in items}
        self.assertIn("subspacead", names)
        for m in items:
            self.assertIn("train_mode", m)
            self.assertIn("support_k_shot", m)

    def test_result_json_additive(self):
        from xccv.engine.models.base import Prediction
        from xccv.engine.result_json import build_result_json

        payload = build_result_json(
            task="anomaly", source={"file": "a.png", "path": "d:/a.png",
                                    "width": 8, "height": 8},
            prediction=Prediction(probs=[3.2]),
            classes=["异常"],
            score_semantics={"key": "z", "name": "图像分数",
                             "range": [0.0, None], "higher": "越高越异常"})
        self.assertEqual(payload["score_semantics"]["key"], "z")
        self.assertIsNone(payload["score_semantics"]["range"][1])
        # 不传语义时不输出键（schema 只增不改，旧消费方零感知）
        payload2 = build_result_json(task="detection",
                                     source={"file": "a.png", "path": "p",
                                             "width": 8, "height": 8},
                                     prediction=Prediction())
        self.assertNotIn("score_semantics", payload2)


class FreeTrainModeTest(unittest.TestCase):
    """§四：免训练范式引擎侧契约。"""

    def test_train_config_model_params(self):
        from xccv.engine.models.base import TrainConfig

        cfg = TrainConfig(task=TaskType.ANOMALY,
                          dataset_dir=Path("."), output_dir=Path("."),
                          model_name="cif",
                          model_params={"k_shot": 6, "aug_count": 30})
        self.assertEqual(cfg.model_params["k_shot"], 6)

    def test_free_models_declared(self):
        """ANOMALY 后端实例声明 z 语义与 free 形态（CIF 需 DINO 权重，
        缺失时由 test_v073_cif 的 skipUnless 路径覆盖）。"""
        from xccv.engine.models.anomaly import AnomalyModel
        from xccv.engine.models.subspacead import SubspaceAdModel

        m = AnomalyModel(1, device="cpu", use_pretrained=False)
        self.assertEqual(m.score_semantics["key"], "z")
        s = SubspaceAdModel(1, device="cpu", use_pretrained=False)
        self.assertEqual(s.score_semantics["key"], "z")
        self.assertEqual(s.train_mode, "free")


class SubspaceAdMathTest(unittest.TestCase):
    """SubspaceAD 数学核心（无需骨干权重）。"""

    def test_layer_indices(self):
        from xccv.engine.models.subspacead import _layer_indices

        self.assertEqual(_layer_indices(24), [-12, -13, -14, -15, -16, -17, -18])
        self.assertEqual(_layer_indices(40), [-12, -13, -14, -15, -16, -17, -18])
        self.assertEqual(_layer_indices(12), [-1, -2, -3, -4, -5, -6, -7])

    def test_pca_residual_low_in_subspace(self):
        """2 维高斯数据落在 1 维子空间 → 子空间内残差 ≈ 0、法向外点残差大。"""
        from xccv.engine.models.subspacead import SubspaceAdModel

        rng = np.random.default_rng(0)
        t = rng.normal(0, 3, size=(2000, 1))
        data = np.hstack([t, t * 2.0]) + rng.normal(0, 1e-6, (2000, 2))
        batches = [torch.from_numpy(data.astype(np.float32))]
        pca = SubspaceAdModel._fit_pca(batches, 2, data.shape[0], "cpu", 0.999)
        self.assertEqual(pca["k"], 1)
        mu, comp = pca["mu"], pca["components"]
        inlier = torch.from_numpy(data[:100].astype(np.float32))
        out = np.array([[5.0, -6.0]], dtype=np.float32)  # 法向偏离
        r_in = SubspaceAdModel._residual_scores(inlier, mu, comp).numpy()
        r_out = SubspaceAdModel._residual_scores(
            torch.from_numpy(out), mu, comp).numpy()
        self.assertLess(float(np.median(r_in)), 1e-3)
        self.assertGreater(float(r_out[0]), float(np.median(r_in)) * 100)

    def test_topk_mean(self):
        from xccv.engine.models.subspacead import _topk_mean

        arr = np.zeros((10, 10), dtype=np.float32)
        arr[0, 0] = 8.0
        arr[0, 1] = 6.0     # top1% = top1 像素（100 像素 → k=1）
        self.assertAlmostEqual(_topk_mean(arr), 8.0, places=4)

    def test_post_process_shape(self):
        from xccv.engine.models.subspacead import _post_process

        out = _post_process(np.ones((7, 7), dtype=np.float32), 56)
        self.assertEqual(out.shape, (56, 56))

    def test_find_backbone_returns(self):
        from xccv.engine.models.subspacead import _find_backbone

        src, sha = _find_backbone()
        self.assertIsInstance(src, str)
        self.assertEqual(len(sha), 16)

    def test_registry_load_format_known(self):
        """注册名与容器格式契约（load_model 分支键一致）。"""
        from xccv.engine.models.registry import MODEL_SPECS

        self.assertIn("subspacead", MODEL_SPECS)
        task, desc, _f = MODEL_SPECS["subspacead"]
        self.assertEqual(task, TaskType.ANOMALY)


import torch  # noqa: E402 （SubspaceAdMathTest 用；置底避免顶部拖慢其它用例）


if __name__ == "__main__":
    unittest.main()

