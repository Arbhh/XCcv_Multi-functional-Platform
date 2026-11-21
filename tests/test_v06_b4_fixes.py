# -*- coding: utf-8 -*-
"""回归：训练几何修复（Y2）、OK 负样本训练、P2 检测头、旧字样清零。

- YoloGeometryTest：训练取样与验证/推理同一几何（长边 imgsz 保持长宽比），
  框像素长宽比不再被非等比拉伸（Y1geo 对照实验：修复宽松 mAP +67%）；
- NegativeSampleTest：负样本文件夹（OK/良品/负样本/背景）内未标注图进入
  训练与验证，标注图划分不变，超量自动抽样，训练产物含误检统计键；
- P2HeadTest / P2CheckboxTest：P2 小目标头作为训练参数（仅 YOLO 家族），
  模板构建、检测头按步长对齐迁移、训练线程接线、复选框持久化与预检提示；
- LegacyWordingTest：平台代码中旧工具字样（注释/字符串）零残留。
"""

import os
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xccv.engine.dataset.base import TaskType
from xccv.engine.dataset.reader import (
    NEGATIVE_DIR_NAMES, is_negative_dir_name, scan_negative_images)
from xccv.engine.dataset.prepare import prepare
from xccv.engine.models.base import negative_fp_stats
from xccv.engine.models.registry import MODEL_SPECS, create_model
from xccv.training.preflight import param_findings

import numpy as np

from tests.synthetic import make_labelme_dataset


def _write_min_jpg(path: Path, w: int = 64, h: int = 48):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((h, w, 3), dtype=np.uint8))


class YoloGeometryTest(unittest.TestCase):
    """训练分支与验证分支同一几何（长边缩放，保持长宽比）。"""

    @classmethod
    def setUpClass(cls):
        from ultralytics.cfg import get_cfg

        from xccv.engine.models.yolo_data import InMemoryYoloDataset

        cls._ds_cls = InMemoryYoloDataset
        hyp = get_cfg()
        for k in ("mosaic", "mixup", "copy_paste", "cutmix", "degrees",
                  "translate", "scale", "shear", "perspective", "flipud",
                  "fliplr", "bgr", "hsv_h", "hsv_s", "hsv_v", "erasing"):
            if hasattr(hyp, k):
                setattr(hyp, k, 0.0)
        cls._hyp = hyp
        tmp = cls._tmp = __import__("tempfile").TemporaryDirectory()
        root, classes = make_labelme_dataset(
            Path(tmp.name) / "ds", n_images=2, size=(128, 64),
            shape_type="rectangle", classes=("cat",))
        samples, cls.classes, _ = __import__(
            "xccv.engine.dataset.reader", fromlist=["x"]).scan_labelme_dataset(root)
        cls.samples = samples

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _ds(self, rect: bool):
        return self._ds_cls(
            self.samples, self.classes, "detection", None,
            img_path="xccv://in-memory", imgsz=64, batch_size=2,
            augment=not rect, hyp=self._hyp, rect=rect, cache=None,
            single_cls=False, stride=32, pad=0.0 if not rect else 0.5,
            prefix="", data={"channels": 3}, fraction=1.0)

    def test_train_keeps_aspect_ratio(self):
        """训练（rect=False）：非方形图长边缩放到 imgsz、保持长宽比
        （此后 ultralytics 训练变换只做 letterbox 补边，不再变形）。"""
        ds = self._ds(rect=False)
        lab = ds.get_image_and_label(0)
        h, w = lab["img"].shape[:2]
        self.assertEqual(max(h, w), 64)
        self.assertAlmostEqual(w / h, 2.0, delta=0.1,
                               msg="128x64 图应缩成 64x32（不再拉伸成 64x64）")
        self.assertEqual(tuple(lab["resized_shape"]), (h, w))

    def test_boxes_pixel_aspect_preserved(self):
        """整条训练管线后框的像素长宽比与原图一致（旧实现拉伸 ×4）。"""
        ds = self._ds(rect=False)
        item = ds[0]
        H, W = item["img"].shape[-2:]
        got = np.asarray(item["bboxes"], dtype=np.float32).reshape(-1, 4)
        base = ds.labels[0]["bboxes"]
        ar_in = (got[:, 2] * W) / (got[:, 3] * H)
        ar_src = (base[:, 2] * 128.0) / (base[:, 3] * 64.0)
        self.assertTrue(np.allclose(ar_in, ar_src, rtol=0.1),
                        f"{ar_in} vs {ar_src}")

    def test_val_rect_same_geometry(self):
        """验证（rect=True）与训练取样几何一致。"""
        lab_v = self._ds(rect=True).get_image_and_label(0)
        lab_t = self._ds(rect=False).get_image_and_label(0)
        self.assertEqual(lab_v["img"].shape[:2], lab_t["img"].shape[:2])

    def test_train_letterbox_pad_114(self):
        """与 ultralytics 官方一致：训练图经 LetterBox 补 114 灰边成 imgsz
        方形（不拉伸）。128x64 → 内容 64x32，上下各补 16 行。"""
        item = self._ds(rect=False)[0]
        img = item["img"]          # (3, 64, 64) uint8 tensor
        self.assertEqual(tuple(img.shape[-2:]), (64, 64))
        top = img[:, :8, :].float()
        self.assertTrue(bool((top == 114).all()), "补边应为 114 灰")

    def test_resize_op_not_applied(self):
        """YOLO 训练忽略 resize 算子的非等比拉伸（只取长边作网络输入）。"""
        ds = self._ds_cls(
            self.samples, self.classes, "detection",
            {"resize": {"width": 32, "height": 320}},
            img_path="xccv://in-memory", imgsz=64, batch_size=2,
            augment=True, hyp=self._hyp, rect=False, cache=None,
            single_cls=False, stride=32, pad=0.0, prefix="",
            data={"channels": 3}, fraction=1.0)
        h, w = ds.get_image_and_label(0)["img"].shape[:2]
        self.assertAlmostEqual(w / h, 2.0, delta=0.1,
                               msg="不应被 resize 算子拉成 32x320")


class NegativeSampleTest(unittest.TestCase):
    """OK 图负样本：扫描、划分、倍增与误检统计。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = __import__("tempfile").TemporaryDirectory()
        root = Path(cls.tmp.name) / "ds"
        root.mkdir(parents=True)
        _, _ = make_labelme_dataset(
            root / "缺陷", n_images=6, size=(64, 48), classes=("cat", "dog"))
        for i in range(5):  # OK 负样本：5 张未标注图
            _write_min_jpg(root / "OK" / f"ok_{i}.jpg")
        _write_min_jpg(root / "OK" / "with_json.jpg")   # OK 目录但有标注 → 非负样本
        _write_min_jpg(root / "缺陷" / "unlabeled.jpg")  # 缺陷目录未标注 → 不收
        (root / "OK" / "with_json.json").write_text(
            '{"imagePath": "with_json.jpg", "imageWidth": 64, "imageHeight": 48,'
            ' "shapes": [{"label": "cat", "shape_type": "rectangle",'
            ' "points": [[1, 1], [10, 10]]}]}', encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_dir_names_and_scan(self):
        self.assertTrue(is_negative_dir_name("OK") and is_negative_dir_name("良品")
                        and is_negative_dir_name("负样本"))
        self.assertFalse(is_negative_dir_name("黑点"))
        negs = scan_negative_images(Path(self.tmp.name) / "ds")
        self.assertEqual(len(negs), 5, "只收 OK 目录内无 json 的 5 张")
        self.assertTrue(all(not s.shapes for s in negs))

    def test_prepare_splits_negatives_and_keeps_positives(self):
        root = Path(self.tmp.name) / "ds"
        off = prepare(TaskType.DETECTION, "yolo", root, val_ratio=0.2,
                      seed=42, train_ratio=0.8)
        on = prepare(TaskType.DETECTION, "yolo", root, val_ratio=0.2,
                     seed=42, train_ratio=0.8, use_negatives=True)
        self.assertEqual(off.neg_train_count, 0)
        self.assertGreater(on.neg_train_count, 0)
        self.assertGreater(on.neg_val_count, 0)
        pos_off = [s for s in off.train_samples if s.shapes]
        pos_on = [s for s in on.train_samples if s.shapes]
        self.assertEqual([str(s.image_path) for s in pos_off],
                         [str(s.image_path) for s in pos_on],
                         "标注图划分必须与关闭负样本时完全一致")
        self.assertTrue(all(not s.shapes for s in on.train_samples
                            if s not in pos_on), "负样本应为空标注")

    def test_negative_cap(self):
        """负样本多于标注图 ×1.0 时训练侧抽样（验证侧保留）。"""
        root = Path(self.tmp.name) / "ds"
        data = prepare(TaskType.DETECTION, "yolo", root, val_ratio=0.0,
                       seed=42, train_ratio=1.0, use_negatives=True)
        n_pos_train = len(data.train_samples)
        self.assertLessEqual(data.neg_train_count, n_pos_train)
        self.assertGreater(data.neg_train_count, 0)

    def test_negative_fp_stats(self):
        """误检统计：分数阈值过滤 + 空样本集返回空。"""
        class _P:
            def __init__(self, scores):
                self.scores = scores

        negs = [type("S", (), {"image_path": Path("x.jpg"), "shapes": []})()]
        calls = []

        def fake_predict(img):
            calls.append(1)
            return _P([0.9, 0.1])

        def fake_read(path):
            return np.zeros((8, 8, 3), dtype=np.uint8)

        from unittest import mock

        with mock.patch("xccv.engine.io_utils.imread_unicode", fake_read):
            out = negative_fp_stats(fake_predict, negs, conf=0.25)
        self.assertEqual(out["neg_fp_per_img"], 1.0)  # 0.1 低于阈值不计
        self.assertEqual(out["neg_fp_img_rate"], 1.0)
        self.assertEqual(negative_fp_stats(fake_predict, [], conf=0.25), {})


class P2HeadTest(unittest.TestCase):
    """P2 小目标头＝训练参数（与 slim 头/小目标锚框同一机制，仅 YOLO 家族）：
    enable_p2 切换/模板/预训练迁移/训练线程接线/预检提示。"""

    def test_registry_has_no_p2_names(self):
        """P2 不再是独立模型名：下拉只列基础模型，由复选框叠加。"""
        self.assertFalse([n for n in MODEL_SPECS if n.endswith("-p2")])

    def test_enable_p2_switches_yolo_variants(self):
        for name, task, base in (("yolo11n", TaskType.DETECTION, "yolo11n"),
                                 ("yolo11s", TaskType.DETECTION, "yolo11s"),
                                 ("yolo11n-seg", TaskType.SEGMENTATION, "yolo11n-seg"),
                                 ("yolo11s-seg", TaskType.SEGMENTATION, "yolo11s-seg")):
            m = create_model(name, task, 2, device="cpu", use_pretrained=True)
            self.assertFalse(m.p2)
            m.enable_p2()
            m.enable_p2()   # 幂等
            self.assertTrue(m.p2)
            self.assertEqual(m.p2_source, base)
            self.assertTrue(m.init_weight.endswith(f"{base}-p2.yaml"), m.init_weight)
            self.assertTrue(os.path.isfile(re.sub(
                r"(\d+)([nslmx])(.*)$", r"\1\3", m.init_weight)),
                f"统一模板必须存在: {m.init_weight}")

    def test_trainer_thread_wiring(self):
        """训练线程：项目勾选 P2 → YOLO 模型启用、R-CNN 忽略；快照只记实际生效。"""
        from unittest import mock

        from xccv.training import trainer_thread as T
        from xccv.training.project import Project

        captured = {}

        def fake_train(self, cfg, reporter=None):
            captured["cfg"], captured["model"] = cfg, self
            return {}

        for name, task, expect in (("yolo11n-seg", "segmentation", True),
                                   ("maskrcnn18", "segmentation", False)):
            p = Project(name="p2", dataset_dir=str(ROOT), model_dir=str(ROOT),
                        task=task, model_name=name, p2_head=True)
            th = T.TrainThread(p, ROOT / "_tmp" / "p2_wiring")
            stats = type("S", (), {"n_images": 1, "classes": ["a"]})()
            with mock.patch.object(T, "collect_stats", return_value=stats), \
                    mock.patch("xccv.engine.models.base.BaseModel.train", fake_train):
                th.run()
            self.assertEqual(captured["cfg"].p2_head, expect, name)
            self.assertEqual(bool(getattr(captured["model"], "p2", False)), expect)

    def test_templates_have_p2_level(self):
        import yaml

        base = ROOT / "xccv" / "engine" / "models" / "configs"
        for f in ("yolo11-p2.yaml", "yolo11-seg-p2.yaml"):
            data = yaml.safe_load((base / f).read_text(encoding="utf-8"))
            head = data["head"]
            detect = head[-1]
            self.assertEqual(len(detect[0]), 4, f"{f} 检测层应为 4 个（含 P2）")
            self.assertEqual(detect[2], "Segment" if "seg" in f else "Detect")
            # P2 分支必须引用骨干第 2 层（stride 4）
            cat_inputs = [h for h in head if h[2] == "Concat" and 2 in h[0]]
            self.assertTrue(cat_inputs, f"{f} 缺少 P2（backbone 层2）Concat")
            if "seg" in f:
                # proto 必须取 stride-8 特征（层16）：Proto 仅上采样 ×2，
                # 掩码须与 GT mask_ratio=4 同步（取 P2 会 matmul 形状崩）
                self.assertEqual(head[-1][0][0], 16, f"{f} proto 输入应为层16")

    def test_shift_head_levels(self):
        """检测头按步长对齐迁移：目标 stride 8/16/32 分支逐张量等于源
        P3/P4/P5 分支；新增 stride 4 分支以源最细分支（stride 8）初始化
        （随机初始化收敛极慢）——检测模板输入 [P2,P3,P4,P5] 与分割模板
        [P3,P2,P4,P5]（proto 取 P3）都必须对齐。"""
        torch = __import__("torch")
        from ultralytics import YOLO

        from xccv.engine.offline import WEIGHTS_DIR
        from xccv.engine.models.yolo import _shift_head_levels

        cfg_dir = ROOT / "xccv" / "engine" / "models" / "configs"
        for base_name in ("yolo11n", "yolo11n-seg"):
            base = Path(WEIGHTS_DIR) / "ultralytics" / f"{base_name}.pt"
            if not base.is_file():
                self.skipTest("随包权重缺失")
            net = YOLO(str(cfg_dir / f"{base_name}-p2.yaml")).model
            src = YOLO(str(base)).model
            head, shead = net.model[-1], src.model[-1]
            s_dst = [int(s) for s in head.stride]
            s_src = [int(s) for s in shead.stride]
            moved = _shift_head_levels(net, src)
            self.assertGreater(moved, 100, base_name)
            pairs = [(s_dst.index(s), i, s) for i, s in enumerate(s_src)]
            pairs.append((s_dst.index(4), s_src.index(8), 4))  # 新增 P2 ← 源 P3
            for j, i, s in pairs:
                for branch in ("cv2", "cv3"):
                    a = getattr(head, branch)[j].state_dict()
                    b = getattr(shead, branch)[i].state_dict()
                    same = [k for k in b if k in a and a[k].shape == b[k].shape]
                    self.assertTrue(same, (base_name, branch, s))
                    self.assertTrue(all(torch.equal(a[k], b[k].float()) for k in same),
                                    f"{base_name} {branch} 目标 stride{s} 分支来源不对")

    def test_preflight_p2_hint(self):
        from xccv.training.project import Project

        def codes(model_name, p2):
            p = Project(name="x", dataset_dir="d", model_dir="m",
                        task="detection", model_name=model_name, p2_head=p2)
            return {f.code for f in param_findings(p, None, cuda_available=False)}

        self.assertIn("p2_vram", codes("yolo11n", True))
        self.assertIn("p2_not_yolo", codes("fasterrcnn", True))
        self.assertFalse({"p2_vram", "p2_not_yolo"} & codes("yolo11n", False))


from tests.test_ui_upgrade import _TrainEnv  # noqa: E402


class P2CheckboxTest(_TrainEnv):
    """训练窗口「P2 小目标检测头（仅 YOLO 家族生效）」复选框：加载/保存/持久化。"""

    def test_checkbox_persists(self):
        from xccv.training.project import load_project

        pw = self._window()
        self.assertIn("YOLO", pw.par_p2.text())
        self.assertFalse(pw.par_p2.isChecked())
        pw.par_p2.setChecked(True)
        pw._save_params()
        self.assertTrue(load_project("ui").p2_head)
        pw2 = self._window()
        self.assertTrue(pw2.par_p2.isChecked())


class LegacyWordingTest(unittest.TestCase):
    """平台代码（xccv/ 与 tests/）中旧工具字样零残留（含注释）。"""

    def test_no_legacy_wording(self):
        pat = re.compile("DL" + "CV", re.IGNORECASE)   # 拼接书写：本文件自身不命中
        hits = []
        for start in ("xccv", "tests"):
            for dp, dn, fn in os.walk(ROOT / start):
                dn[:] = [d for d in dn if d not in ("__pycache__",)]
                for f in fn:
                    if not f.endswith((".py", ".html", ".js", ".qss")):
                        continue
                    text = (Path(dp) / f).read_text(encoding="utf-8",
                                                    errors="ignore")
                    for i, line in enumerate(text.splitlines(), 1):
                        if pat.search(line):
                            hits.append(f"{Path(dp, f)}:{i}")
        self.assertEqual(hits, [], f"平台代码仍含旧字样: {hits[:8]}")


if __name__ == "__main__":
    unittest.main()


