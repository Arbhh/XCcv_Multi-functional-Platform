"""数据契约测试：labelme JSON → 内部标注 → YOLO 行的往返一致性，
以及流程图 IR 的端口 schema 拒绝语义（评审建议的契约回归锁）。

这些测试锁住"标注格式"与"流程图结构"两条对外数据契约：
改动任何一侧的序列化/校验行为时，必须显式更新本文件并说明原因。
"""

import json
import tempfile
import unittest
from pathlib import Path

from xccv.engine.dataset.base import Sample, Shape
from xccv.engine.dataset.converters import _yolo_det_line, _yolo_seg_line, split_counts, split_samples
from xccv.engine.dataset.reader import is_negative_dir_name, read_labelme_json


def _rect(label, x1, y1, x2, y2):
    return Shape(label=label, shape_type="rectangle",
                 points=[[x1, y1], [x2, y2]])


class YoloLineContract(unittest.TestCase):
    """YOLO 行契约：归一化到 [0,1]、越界裁剪、画面外目标返回 None。"""

    def test_det_line_normalized(self):
        sh = _rect("缺陷", 100, 80, 300, 240)
        line = _yolo_det_line(sh, 640, 480, 2)
        parts = line.split(" ")
        self.assertEqual(len(parts), 5)
        self.assertEqual(parts[0], "2")
        cx, cy, w, h = (float(v) for v in parts[1:])
        self.assertTrue(all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)))
        self.assertAlmostEqual(w, 200.0 / 640.0, places=6)
        self.assertAlmostEqual(h, 160.0 / 480.0, places=6)
        self.assertAlmostEqual(cx, 200.0 / 640.0, places=6)
        self.assertAlmostEqual(cy, 160.0 / 480.0, places=6)

    def test_det_line_clipped_to_frame(self):
        sh = _rect("缺陷", -50, -40, 2000, 1600)
        line = _yolo_det_line(sh, 640, 480, 0)
        _c, cx, cy, w, h = (float(v) for v in line.split(" "))
        self.assertLessEqual(cx + w / 2, 1.0 + 1e-6)
        self.assertLessEqual(cy + h / 2, 1.0 + 1e-6)
        self.assertGreaterEqual(cx - w / 2, -1e-6)

    def test_det_line_none_when_out_of_frame(self):
        sh = _rect("缺陷", 800, 600, 900, 700)
        self.assertIsNone(_yolo_det_line(sh, 640, 480, 0))

    def test_seg_line_closed_ring(self):
        pts = [[10, 10], [110, 10], [110, 110], [10, 110]]
        sh = Shape(label="缺陷", shape_type="polygon", points=pts)
        line = _yolo_seg_line(sh, 640, 480, 1)
        vals = [float(v) for v in line.split(" ")]
        cls, coords = vals[0], vals[1:]
        self.assertEqual(cls, 1)
        self.assertEqual(len(coords) % 2, 0)
        self.assertGreaterEqual(len(coords), 6)   # 至少 3 点
        self.assertTrue(all(0.0 <= v <= 1.0 for v in coords))
        # 多边形坐标顺序保持（首点 = 原始首点归一化）
        self.assertAlmostEqual(coords[0], 10.0 / 640.0, places=6)
        self.assertAlmostEqual(coords[1], 10.0 / 480.0, places=6)


class SplitConsistencyContract(unittest.TestCase):
    """split_counts 与 split_samples 同口径（环状图推算的契约）。"""

    @staticmethod
    def _samples(n):
        return [Sample(image_path=Path(f"{i}.png"), width=640, height=480,
                       label=f"c{i % 3}") for i in range(n)]

    def test_counts_match_samples_mutually_exclusive(self):
        for n in (10, 67, 120):
            tr, va = split_samples(self._samples(n), 0.2, seed=7, train_ratio=0.8)
            ctr, cva = split_counts(n, 0.2, 0.8)
            self.assertEqual((len(tr), len(va)), (ctr, cva), f"n={n}")

    def test_counts_match_samples_overlap(self):
        for n in (50, 100):
            tr, va = split_samples(self._samples(n), 0.3, seed=7, train_ratio=0.9)
            ctr, cva = split_counts(n, 0.3, 0.9)
            self.assertEqual((len(tr), len(va)), (ctr, cva), f"n={n}")

    def test_counts_match_samples_val_zero_reuse(self):
        tr, va = split_samples(self._samples(30), 0.0, seed=7, train_ratio=1.0)
        ctr, cva = split_counts(30, 0.0, 1.0)
        self.assertEqual((len(tr), len(va)), (ctr, cva))
        self.assertEqual(len(tr), len(va))   # 验证集复用训练集


class LabelmeReaderContract(unittest.TestCase):
    """labelme JSON → Sample 的读取契约。"""

    def test_read_labelme_json_shapes(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            img = d / "a.png"
            img.write_bytes(b"\x89PNG fake")
            jp = d / "a.json"
            jp.write_text(json.dumps({
                "version": "5.8.1", "flags": {},
                "shapes": [
                    {"label": "黑点", "points": [[1, 2], [30, 40]],
                     "shape_type": "rectangle", "group_id": None, "flags": {}},
                    {"label": "划痕", "points": [[5, 5], [20, 8], [25, 30]],
                     "shape_type": "polygon", "group_id": None, "flags": {}},
                ],
                "imagePath": "a.png", "imageData": None,
                "imageHeight": 480, "imageWidth": 640,
            }), encoding="utf-8")
            s = read_labelme_json(jp)
            self.assertEqual(s.image_path, img)
            self.assertEqual([sh.label for sh in s.shapes], ["黑点", "划痕"])
            self.assertEqual(s.shapes[0].shape_type, "rectangle")
            self.assertEqual(s.shapes[1].points[2], [25.0, 30.0])

    def test_ok_dir_names(self):
        self.assertTrue(is_negative_dir_name("OK"))
        self.assertTrue(is_negative_dir_name("ok"))
        self.assertTrue(is_negative_dir_name("良品"))
        self.assertFalse(is_negative_dir_name("NG"))
        self.assertFalse(is_negative_dir_name("缺陷"))


class FlowSchemaContract(unittest.TestCase):
    """流程图 IR 结构契约：未知类型 / 槽越界 / 类型不匹配 必须被拒绝。

    图内 type 用注册名（无 xccv2/ 前缀——流程文件保存时带前缀，
    由 flow_session._normalize_types 剥除后进入 Executor，此处同语义）。
    """

    @staticmethod
    def _graph(nodes, links):
        return {"nodes": nodes, "links": links}

    @classmethod
    def setUpClass(cls):
        from xccv.engine.flow.ops import load_all

        load_all()

    def test_unknown_node_type_rejected(self):
        from xccv.engine.flow.executor import Executor, FlowError

        g = self._graph(
            [{"id": 1, "type": "no_such_node", "properties": {}}],
            [])
        with self.assertRaises(FlowError):
            Executor(g).validate()

    def test_slot_overflow_rejected(self):
        # 连线为 LiteGraph 5 元组 [link_id, src, sslot, dst, dslot]；
        # 输出槽越界在构造期即被拒绝（IndexError），validate 前不可能通过
        from xccv.engine.flow.executor import Executor

        g = self._graph(
            [{"id": 1, "type": "image_loader", "properties": {"path": "x"}},
             {"id": 2, "type": "result_output", "properties": {}}],
            [[1, 1, 99, 2, 0]])   # 输出槽 99 越界
        with self.assertRaises(Exception):
            Executor(g).validate()

    def test_port_type_mismatch_rejected(self):
        from xccv.engine.flow.executor import Executor, FlowError

        g = self._graph(
            [{"id": 1, "type": "image_loader", "properties": {"path": "x"}},
             {"id": 2, "type": "result_output", "properties": {}}],
            [[1, 1, 0, 2, 1]])    # image 口 → result 口，类型不匹配
        with self.assertRaises(FlowError):
            Executor(g).validate()


if __name__ == "__main__":
    unittest.main()
