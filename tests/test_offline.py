"""离线权重：随包基准权重优先、完全离线训练的配置断言。

基准权重不随仓库分发（体积与许可各异），本地可选放置；对应断言在权重缺失时跳过。
"""

import os
import unittest
from pathlib import Path

from xccv.engine.dataset.base import TaskType
from xccv.engine.models.yolo import YoloModel
from xccv.engine.offline import WEIGHTS_DIR, setup_offline_weights

_HAS_YOLO11N = (WEIGHTS_DIR / "ultralytics" / "yolo11n.pt").is_file()
_HAS_YOLO11S_SEG = (WEIGHTS_DIR / "ultralytics" / "yolo11s-seg.pt").is_file()


class OfflineWeightsTest(unittest.TestCase):
    def test_setup_redirects_torch_home(self):
        """随包 torch 权重目录存在时才重定向 TORCH_HOME（无权重仓库不指向空目录）。"""
        setup_offline_weights()
        expected = str((Path(__file__).resolve().parents[1]
                        / "xccv" / "engine" / "weights" / "torch"))
        if Path(expected).is_dir():
            self.assertEqual(os.environ.get("TORCH_HOME"), expected)
        else:
            self.assertNotEqual(os.environ.get("TORCH_HOME"), expected)

    def test_weights_dir_points_inside_package(self):
        self.assertTrue(str(WEIGHTS_DIR).endswith(str(Path("engine") / "weights")))

    def test_yolo_pretrained_path_points_at_bundled_dir(self):
        m = YoloModel("yolo11n", TaskType.DETECTION, 2, use_pretrained=True)
        self.assertEqual(Path(m.init_weight),
                         WEIGHTS_DIR / "ultralytics" / "yolo11n.pt")

    @unittest.skipUnless(_HAS_YOLO11N, "未放置基准权重 yolo11n.pt（可选）")
    def test_yolo_uses_bundled_pretrained(self):
        m = YoloModel("yolo11n", TaskType.DETECTION, 2, use_pretrained=True)
        self.assertTrue(Path(m.init_weight).is_file())

    @unittest.skipUnless(_HAS_YOLO11S_SEG, "未放置基准权重 yolo11s-seg.pt（可选）")
    def test_yolo_seg_uses_bundled_pretrained(self):
        m = YoloModel("yolo11s", TaskType.SEGMENTATION, 2, use_pretrained=True)
        self.assertTrue(Path(m.init_weight).is_file())
        self.assertIn("yolo11s-seg.pt", str(m.init_weight))

    def test_yolo_no_pretrained_still_yaml(self):
        m = YoloModel("yolo11n", TaskType.DETECTION, 2, use_pretrained=False)
        self.assertEqual(m.init_weight, "yolo11n.yaml")


if __name__ == "__main__":
    unittest.main()

