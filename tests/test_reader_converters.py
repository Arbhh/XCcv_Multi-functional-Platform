"""数据读取/划分/内存标签构造/准备的单元测试（训练数据零落盘）。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from xccv.engine.dataset.base import TaskType
from xccv.engine.dataset.converters import split_samples
from xccv.engine.dataset.prepare import prepare
from xccv.engine.dataset.reader import scan_classification_dataset, scan_labelme_dataset
from xccv.engine.models.yolo_data import build_label_dict
from tests.synthetic import make_classification_dataset, make_labelme_dataset


class ReaderConverterTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.det_dir, self.classes = make_labelme_dataset(self.root / "det", n_images=6)

    def tearDown(self):
        self._tmp.cleanup()

    def test_scan_labelme_with_classes_file(self):
        samples, classes, from_file = scan_labelme_dataset(self.det_dir)
        self.assertEqual(len(samples), 6)
        self.assertEqual(classes, ["cat", "dog"])
        self.assertTrue(from_file)
        self.assertEqual(samples[0].width, 96)
        self.assertEqual(len(samples[0].shapes), 1)

    def test_scan_labelme_without_classes_file(self):
        (self.det_dir / "classes.txt").unlink()
        _, classes, from_file = scan_labelme_dataset(self.det_dir)
        self.assertEqual(classes, ["cat", "dog"])
        self.assertFalse(from_file)

    def test_split_ratio(self):
        samples, _, _ = scan_labelme_dataset(self.det_dir)
        train, val = split_samples(samples, 0.3, seed=1)
        self.assertEqual(len(train) + len(val), 6)
        self.assertEqual(len(val), 2)

    def test_split_train_ratio_legacy_equivalent(self):
        """train_ratio=None 与 1-val_ratio 划分结果必须逐样本一致（旧口径兼容）。"""
        samples, _, _ = scan_labelme_dataset(self.det_dir)
        t1, v1 = split_samples(samples, 0.3, seed=7)
        t2, v2 = split_samples(samples, 0.3, seed=7, train_ratio=0.7)
        self.assertEqual([id(s) for s in t1], [id(s) for s in t2])
        self.assertEqual([id(s) for s in v1], [id(s) for s in v2])

    def test_split_train_ratio_overlap(self):
        """训练 100% + 验证 20%（稀缺类别场景）：训练集含全部样本，验证为其子集。"""
        samples, _, _ = scan_labelme_dataset(self.det_dir)
        train, val = split_samples(samples, 0.2, seed=3, train_ratio=1.0)
        self.assertEqual(len(train), 6)
        self.assertEqual(len(val), 1)  # round(6*0.2)
        train_keys = {s.image_path for s in train}
        for s in val:  # 验证集允许与训练集重叠（子集）
            self.assertIn(s.image_path, train_keys)

    def test_split_train_ratio_disjoint(self):
        """训练+验证 ≤ 100% 时互斥：验证只从训练之外的剩余抽取。"""
        samples, _, _ = scan_labelme_dataset(self.det_dir)
        train, val = split_samples(samples, 0.2, seed=3, train_ratio=0.7)
        self.assertEqual(len(train), 4)  # round(6*0.7)
        self.assertEqual(len(val), 1)    # 剩余 2 张里取 round(2*0.2/0.3)=1
        val_paths = {s.image_path for s in val}
        self.assertFalse(val_paths & {s.image_path for s in train})

    def test_split_train_ratio_stratify_overlap(self):
        """分层 + 重叠：每类独立按比例抽取，小类也能训满（分类数据集）。"""
        cls_dir, _ = make_classification_dataset(self.root / "cls", per_class=6)
        samples, classes = scan_classification_dataset(cls_dir)
        self.assertEqual(classes, ["cat", "dog"])
        train, val = split_samples(samples, 0.5, seed=5, stratify=True,
                                   train_ratio=1.0)
        self.assertEqual(len(train), 12)
        for cls in classes:
            self.assertEqual(sum(1 for s in train if s.label == cls), 6)
            self.assertEqual(sum(1 for s in val if s.label == cls), 3)
            train_keys = {s.image_path for s in train}
            for s in val:
                if s.label == cls:
                    self.assertIn(s.image_path, train_keys)

    def test_build_label_dict_detection(self):
        """检测标签：归一化 xywh，与类别索引一致。"""
        samples, classes, _ = scan_labelme_dataset(self.det_dir)
        lb = build_label_dict(samples[0], classes, "detection")
        self.assertEqual(lb["bbox_format"], "xywh")
        self.assertTrue(lb["normalized"])
        self.assertEqual(lb["shape"], (samples[0].height, samples[0].width))
        self.assertEqual(len(lb["cls"]), len(lb["bboxes"]))
        self.assertGreaterEqual(len(lb["cls"]), 1)
        cls = int(lb["cls"][0, 0])
        box = lb["bboxes"][0]
        self.assertEqual(classes[cls], samples[0].shapes[0].label)
        self.assertTrue(np.all(box >= 0) and np.all(box <= 1))
        self.assertEqual(lb["segments"], [])

    def test_build_label_dict_segmentation(self):
        """分割标签：多边形顶点归一化，矩形转四角点。"""
        seg_dir, _ = make_labelme_dataset(self.root / "seg", n_images=4, shape_type="polygon")
        samples, classes, _ = scan_labelme_dataset(seg_dir)
        lb = build_label_dict(samples[0], classes, "segmentation")
        seg = lb["segments"][0]
        self.assertGreaterEqual(seg.shape[0], 4)  # 至少 4 个顶点
        self.assertLessEqual(float(seg.max()), 1.0)
        # 框为轮廓外接框
        self.assertEqual(len(lb["bboxes"]), len(lb["cls"]))

    def test_build_label_dict_rectangle_in_seg_task(self):
        """分割任务中的矩形应按矩形掩码（四角点）处理。"""
        samples, classes, _ = scan_labelme_dataset(self.det_dir)  # rectangle 数据
        lb = build_label_dict(samples[0], classes, "segmentation")
        self.assertEqual(lb["segments"][0].shape[0], 4)  # 恰好四角点

    def test_prepare_yolo(self):
        data = prepare(TaskType.DETECTION, "yolo", self.det_dir, val_ratio=0.25)
        self.assertEqual(data.classes, ["cat", "dog"])
        self.assertEqual(data.train_count + data.val_count, 6)
        self.assertEqual(len(data.train_samples), data.train_count)
        self.assertEqual(len(data.val_samples), data.val_count)

    def test_prepare_zero_disk(self):
        """数据准备全程零落盘：不产生任何导出目录或文件。"""
        before = sorted(p.name for p in self.det_dir.rglob("*"))
        data = prepare(TaskType.SEGMENTATION, "yolo", self.det_dir, val_ratio=0.25,
                       multiply_factor=2, augment_ops={"random_brightness": {"max_delta": 20}})
        after = sorted(p.name for p in self.det_dir.rglob("*"))
        self.assertEqual(before, after)          # 数据集目录无新增
        self.assertEqual(data.train_count, (6 - round(6 * 0.25)) * 2)
        self.assertEqual(len(data.train_samples), data.train_count)

    def test_prepare_classification(self):
        cls_dir, _ = make_classification_dataset(self.root / "cls", per_class=4)
        data = prepare(TaskType.CLASSIFICATION, "folder", cls_dir, val_ratio=0.25)
        self.assertEqual(data.classes, ["cat", "dog"])
        self.assertEqual(data.train_count, 6)
        self.assertEqual(data.val_count, 2)

    def test_prepare_family_task_mismatch(self):
        with self.assertRaises(ValueError):
            prepare(TaskType.DETECTION, "folder", self.det_dir, val_ratio=0.2)


if __name__ == "__main__":
    unittest.main()

