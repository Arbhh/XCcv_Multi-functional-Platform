"""V0.6 P3 回归：T3-2 removed 端口 / T3-3 裁图族 / T3-4 relabel_map / T3-5 输出与逻辑算子。

纪律：中文路径一律 io_utils；判空不用真值；检出输入同时覆盖
list 与 ndarray 两种 boxes 形态。
"""

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from xccv.engine.flow import ops
from xccv.engine.flow.executor import Executor, FlowError
from xccv.engine.flow.ir import ImageData, MeasureResult, NumberVal
from xccv.engine.flow.node_base import RunContext
from xccv.engine.models.base import Prediction

ops.load_all()
import xccv.engine.flow.legacy_adapter as la  # noqa: E402  过滤节点注册
from xccv.engine.flow.ops import crop as cr  # noqa: E402
from xccv.engine.flow.ops import ai_bridge as ab  # noqa: E402
from xccv.engine.flow.ops import logic as lg  # noqa: E402
from xccv.engine.flow.ops import output as out  # noqa: E402


def _img(h=100, w=200, ch=3):
    rng = np.random.default_rng(7)
    shape = (h, w, ch) if ch else (h, w)
    return ImageData(mat=rng.integers(0, 255, size=shape, dtype=np.uint8))


def _pred(as_array=False, masks=False):
    boxes = [[10, 10, 30, 30], [50, 20, 110, 80], [150, 60, 190, 95]]
    labels, scores = [0, 1, 0], [0.9, 0.4, 0.7]
    m = None
    if masks:
        m = []
        for x1, y1, x2, y2 in boxes:
            mk = np.zeros((100, 200), bool)
            mk[y1:y2, x1:x2] = True
            m.append(mk)
    if as_array:
        return Prediction(labels=np.array(labels), scores=np.array(scores),
                          boxes=np.array(boxes, dtype=float), masks=m)
    return Prediction(labels=labels, scores=scores, boxes=boxes, masks=m)


# ---------------------------------------------------------------- T3-2
class RemovedPortTest(unittest.TestCase):
    def test_apply_flow_kept_plus_removed(self):
        from xccv.engine.detection_flow import apply_flow

        flow = {"nodes": [{"op": "confidence", "params": {"min": 0.5}}]}
        kept, info = apply_flow(_pred(masks=True), (200, 100), flow)
        rem = info["removed_prediction"]
        self.assertEqual(len(kept.boxes) + len(rem.boxes), 3)
        self.assertEqual(info["removed"], len(rem.boxes))
        self.assertEqual(rem.scores, [0.4])
        self.assertEqual(rem.boxes, [[50.0, 20.0, 110.0, 80.0]])
        self.assertEqual(len(rem.masks), 1)                  # 掩码随检出分配
        self.assertTrue(rem.masks[0][50, 80])

    def test_filter_nodes_declare_removed_last(self):
        for cls in (la.FilterConfidence, la.FilterClass, la.FilterBboxSize,
                    la.FilterBboxArea, la.FilterMaskArea, la.FilterCountCheck,
                    la.FilterRegionBlock, la.FilterRegionKeep):
            names = [p.name for p in cls.OUTPUTS]
            # 图像打头（定稿），removed 收尾
            self.assertEqual(names[0], "image", cls.NAME)
            self.assertIn("result", names, cls.NAME)
            self.assertEqual(names[-1], "removed", cls.NAME)
        self.assertEqual([p.name for p in la.FilterCountCheck.OUTPUTS],
                         ["image", "result", "count", "removed"])

    def test_graph_consumes_removed_port(self):
        """removed（槽 1）接下游计数：kept 与 removed 分配、计数一致。"""
        graph = {"nodes": [
            {"id": 1, "type": "detection_source", "properties": {}},
            {"id": 2, "type": "filter_confidence", "properties": {"min": 0.5}},
            {"id": 3, "type": "filter_count_check", "properties": {"min": 0, "max": 0}},
            {"id": 4, "type": "filter_count_check", "properties": {"min": 0, "max": 0}}],
            "links": [[1, 1, 0, 2, 1], [2, 2, 1, 3, 1], [3, 2, 2, 4, 1]]}
        ctx = RunContext()
        ctx.set("inject_detection", _pred())
        res = Executor(graph, ctx).run()
        self.assertTrue(all(r.ok for r in res.values()), {k: r.error for k, r in res.items()})
        self.assertEqual(len(res[3].outputs["result"].boxes), 2)
        self.assertEqual(len(res[4].outputs["result"].boxes), 1)

    def test_old_graph_slot0_only_unchanged(self):
        """旧快照（只连 detection 槽 0）经迁移层后与旧链逐位一致。"""
        from xccv.engine.detection_flow import apply_flow
        from xccv.engine.flow.port_migrate import migrate_ports

        graph = {"nodes": [
            {"id": 1, "type": "detection_source", "properties": {}},
            {"id": 2, "type": "filter_confidence", "properties": {"min": 0.5}}],
            "links": [[1, 1, 0, 2, 0]]}
        ctx = RunContext()
        ctx.set("inject_detection", _pred())
        got = Executor(migrate_ports(graph), ctx).run()[2].outputs["result"]
        ref, _ = apply_flow(_pred(), (200, 100),
                            {"nodes": [{"op": "confidence", "params": {"min": 0.5}}]})
        self.assertEqual(got.boxes, ref.boxes)
        self.assertEqual(got.scores, ref.scores)


# ---------------------------------------------------------------- T3-3
class CropByDetectionTest(unittest.TestCase):
    P = {"select": "largest", "index": 0, "margin_pct": 0.0, "square": False,
         "min_side": 0, "save_dir": "", "name_prefix": "crop"}

    def _run(self, det, ctx=None, img=None, **kw):
        params = dict(self.P, **kw)
        return cr.CropByDetection().run({"image": img or _img(), "result": det},
                                        params, ctx or RunContext())["image"]

    def test_largest_first_index(self):
        for as_array in (False, True):
            out_l = self._run(_pred(as_array))
            self.assertEqual(out_l.mat.shape[:2], (60, 60))      # 框 2 面积最大
            self.assertEqual(out_l.meta["roi_rect"], [50, 20, 110, 80])
            out_f = self._run(_pred(as_array), select="first")
            self.assertEqual(out_f.meta["roi_rect"], [10, 10, 30, 30])
            out_i = self._run(_pred(as_array), select="index", index=2)
            self.assertEqual(out_i.meta["roi_rect"], [150, 60, 190, 95])
            self.assertEqual(len(out_i.meta["crops"]), 3)

    def test_margin_square_min_side_and_clamp(self):
        o = self._run(_pred(), select="first", margin_pct=50.0)   # 20→40 宽
        self.assertEqual(o.meta["roi_rect"], [0, 0, 40, 40])
        sq = self._run(_pred(), select="index", index=2, square=True)  # 40×35→40×40
        x1, y1, x2, y2 = sq.meta["roi_rect"]
        self.assertEqual((x2 - x1, y2 - y1), (40, 40))             # 整数尺寸，不多一行
        self.assertEqual(sq.mat.shape[:2], (40, 40))
        ms = self._run(_pred(), select="first", min_side=50)
        self.assertEqual(ms.meta["roi_rect"], [0, 0, 50, 50])     # 贴边平移回图内，尺寸保持

    def test_border_shift_keeps_size(self):
        det = Prediction(labels=[0], scores=[0.9], boxes=[[180, 80, 200, 100]])
        o = self._run(det, min_side=40)
        self.assertEqual(o.meta["roi_rect"], [160, 60, 200, 100])
        tiny = ImageData(mat=np.zeros((30, 30, 3), np.uint8))   # 图比要求小：才截短
        o2 = self._run(Prediction(labels=[0], scores=[0.9], boxes=[[5, 5, 10, 10]]),
                       img=tiny, min_side=64)
        self.assertEqual(o2.meta["roi_rect"], [0, 0, 30, 30])

    def test_empty_and_out_of_range_warn(self):
        ctx = RunContext()
        empty = self._run(Prediction(labels=[], scores=[], boxes=[]), ctx=ctx)
        self.assertTrue(empty.meta["empty"])
        self.assertEqual(empty.mat.shape[:2], (1, 1))
        self.assertTrue(any("没有检出" in w for w in ctx.warnings))
        ctx2 = RunContext()
        oor = self._run(_pred(), ctx=ctx2, select="index", index=9)
        self.assertTrue(oor.meta["empty"])
        self.assertTrue(any("越界" in w for w in ctx2.warnings))

    def test_save_all_crops_chinese_dir(self):
        from xccv.engine.io_utils import imread_unicode

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "裁图输出"
            ctx = RunContext()
            ctx.set("source_name", "样本图.png")
            o = self._run(_pred(), ctx=ctx, save_dir=str(d), name_prefix="缺陷")
            files = sorted(p.name for p in d.iterdir())
            self.assertEqual(files, ["缺陷_样本图_00.png", "缺陷_样本图_01.png",
                                     "缺陷_样本图_02.png"])
            self.assertEqual(imread_unicode(d / files[1]).shape[:2], (60, 60))
            self.assertTrue(all(c["file"] for c in o.meta["crops"]))

    def test_grayscale_input(self):
        o = self._run(_pred(), img=_img(ch=None))
        self.assertEqual(o.mat.ndim, 2)


class GridTileTest(unittest.TestCase):
    P = {"rows": 2, "cols": 3, "overlap_px": 0, "index": 0, "save_dir": "",
         "name_prefix": "tile"}

    def _run(self, ctx=None, **kw):
        return cr.GridTile().run({"image": _img()}, dict(self.P, **kw),
                                 ctx or RunContext())["image"]

    def test_tiles_cover_image(self):
        o = self._run()
        rects = [t["rect"] for t in o.meta["tiles"]]
        self.assertEqual(len(rects), 6)
        cover = np.zeros((100, 200), int)
        for x1, y1, x2, y2 in rects:
            cover[y1:y2, x1:x2] += 1
        self.assertTrue((cover == 1).all())                       # 无重叠时恰好覆盖一次

    def test_overlap_and_index(self):
        o = self._run(overlap_px=5, index=4)                      # 第 2 行第 2 列
        x1, y1, x2, y2 = o.meta["roi_rect"]
        self.assertEqual((x1, y1), (62, 45))
        self.assertEqual(o.mat.shape[:2], (y2 - y1, x2 - x1))

    def test_out_of_range(self):
        ctx = RunContext()
        o = self._run(ctx=ctx, index=6)
        self.assertTrue(o.meta["empty"])
        self.assertTrue(any("越界" in w for w in ctx.warnings))

    def test_save_tiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "分块"
            self._run(save_dir=str(d))
            self.assertEqual(len(list(d.iterdir())), 6)

    def test_port_miswire_message(self):
        """端口错接（检出→图像）：执行器给出类型不匹配的中文报错。"""
        graph = {"nodes": [{"id": 1, "type": "detection_source", "properties": {}},
                           {"id": 2, "type": "grid_tile", "properties": {}}],
                 "links": [[1, 1, 0, 2, 0]]}
        with self.assertRaises(FlowError) as cm:
            Executor(graph, RunContext()).validate()
        self.assertIn("端口类型不匹配", str(cm.exception))


# ---------------------------------------------------------------- T3-4
class RelabelMapTest(unittest.TestCase):
    CLASSES = ["scratch", "dirt"]

    def _ctx(self):
        ctx = RunContext()
        ctx.set("model_classes", list(self.CLASSES))
        return ctx

    def _run(self, ctx, mapping, action="keep", det=None):
        return ab.RelabelMap().run({"result": det or _pred()},
                                   {"mapping": mapping, "default_action": action}, ctx)

    def test_rename_by_name_and_id_keep(self):
        ctx = self._ctx()
        d = self._run(ctx, "scratch:划痕,1:脏污")["result"]
        names = ctx.get("model_classes")
        self.assertEqual(names[:2], self.CLASSES)                 # 原序号名字不变
        self.assertEqual([names[i] for i in d.labels], ["划痕", "脏污", "划痕"])
        self.assertEqual(d.extra["original_label"], ["scratch", "dirt", "scratch"])

    def test_drop_unmapped(self):
        ctx = self._ctx()
        d = self._run(ctx, "dirt:脏污", action="drop", det=_pred(as_array=True, masks=True))["result"]
        self.assertEqual(len(d.boxes), 1)
        self.assertEqual(len(d.masks), 1)
        self.assertEqual(ctx.get("model_classes")[d.labels[0]], "脏污")
        self.assertTrue(any("删除" in w for w in ctx.warnings))

    def test_result_json_shows_new_labels(self):
        ctx = self._ctx()
        d = self._run(ctx, "scratch:划痕")["result"]
        with tempfile.TemporaryDirectory() as tmp:
            ctx.set("source_name", "a.png")
            out.ResultOutput().run({"result": d}, {"out_dir": tmp}, ctx)
            data = json.loads((Path(tmp) / "a.json").read_text(encoding="utf-8"))
        labels = [r["label"] for r in data["detections"]]
        self.assertEqual(labels, ["划痕", "dirt", "划痕"])

    def test_validate_bad_mapping(self):
        node = ab.RelabelMap()
        self.assertTrue(node.validate({"mapping": "scratch", "default_action": "keep"}))
        self.assertTrue(node.validate({"mapping": "a:", "default_action": "keep"}))
        self.assertEqual(node.validate({"mapping": "a：b，c:d", "default_action": "keep"}), [])


# ---------------------------------------------------------------- T3-5
class ImageSinkTest(unittest.TestCase):
    P = {"out_dir": "", "what": "image", "fmt": "png", "quality": 95, "name_suffix": ""}

    def test_save_png_chinese_dir_and_overlay_jpg(self):
        from xccv.engine.io_utils import imread_unicode

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "保存图像"
            ctx = RunContext()
            ctx.set("source_name", "原图.bmp")
            r = out.ImageSink().run({"image": _img()}, dict(self.P, out_dir=str(d)), ctx)
            self.assertEqual(r, {})   # 输出节点无输出端口
            self.assertEqual(imread_unicode(d / "原图.png").shape, (100, 200, 3))
            r2 = out.ImageSink().run({"image": _img(), "result": _pred()},
                                     dict(self.P, out_dir=str(d), what="overlay", fmt="jpg",
                                          quality=80, name_suffix="_叠加"), ctx)
            self.assertTrue((d / "原图_叠加.jpg").is_file())

    def test_default_dir_under_flow_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = RunContext()
            ctx.set("flow_dir", tmp)
            ctx.set("source_name", "x.png")
            out.ImageSink().run({"image": _img()}, dict(self.P), ctx)
            self.assertTrue((Path(tmp) / "流程结果" / "图像" / "x.png").is_file())

    def test_empty_image_skipped(self):
        ctx = RunContext()
        img = ImageData(mat=np.zeros((1, 1, 3), np.uint8), meta={"empty": True})
        r = out.ImageSink().run({"image": img}, dict(self.P), ctx)
        self.assertEqual(r, {})
        self.assertTrue(ctx.warnings)


class CsvSinkTest(unittest.TestCase):
    def _run(self, path, ctx, columns="source,label,score,x1", append=True, **inputs):
        return out.CsvSink().run(inputs, {"out_path": str(path), "columns": columns,
                                          "append": append}, ctx)

    def _read(self, path):
        with open(path, encoding="utf-8-sig", newline="") as f:
            return list(csv.reader(f))

    def test_rows_append_and_bom(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "结果表.csv"
            ctx = RunContext()
            ctx.set("model_classes", ["划痕", "脏污"])
            ctx.set("source_name", "图1.png")
            self._run(path, ctx, result=_pred(as_array=True))
            self._run(path, ctx, result=_pred())
            rows = self._read(path)
            self.assertEqual(rows[0], ["source", "label", "score", "x1"])
            self.assertEqual(len(rows), 7)                        # 表头 + 3 + 3
            self.assertEqual(rows[1][:2], ["图1.png", "划痕"])
            self.assertTrue(path.read_bytes().startswith(bytes([0xEF, 0xBB, 0xBF])))  # BOM

    def test_header_self_heal_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.csv"
            ctx = RunContext()
            self._run(path, ctx, result=_pred())
            self._run(path, ctx, columns="source,label,value", result=_pred())
            baks = [p for p in Path(tmp).iterdir() if ".bak_" in p.name]
            self.assertEqual(len(baks), 1)                        # 旧文件备份不丢
            self.assertEqual(self._read(path)[0], ["source", "label", "value"])
            self.assertTrue(any("表头" in w for w in ctx.warnings))

    def test_measure_only_row_and_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.csv"
            ctx = RunContext()
            m = MeasureResult(value=3.5, unit="mm", passed=True)
            self._run(path, ctx, columns="value,unit,passed", append=False, result=m)
            self._run(path, ctx, columns="value,unit,passed", append=False,
                      number=NumberVal(value=7.0, unit="px"))
            rows = self._read(path)
            self.assertEqual(rows, [["value", "unit", "passed"], ["7.0", "px", ""]])

    def test_validate_unknown_columns(self):
        node = out.CsvSink()
        errs = node.validate({"out_path": "", "columns": "source,foo", "append": True})
        self.assertTrue(any("未知列" in e for e in errs))

    def test_nothing_to_write_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = RunContext()
            r = self._run(Path(tmp) / "n.csv", ctx)
            self.assertEqual(r, {})
            self.assertTrue(ctx.warnings)


class BoolOpTest(unittest.TestCase):
    def _r(self, op, a, b=None, with_b=True):
        inputs = {"a": a}
        if with_b:
            inputs["b"] = b
        return lg.BoolOp().run(inputs, {"op": op}, RunContext())["result"]

    def test_truth_table(self):
        for a in (False, True):
            for b in (False, True):
                self.assertEqual(self._r("and", a, b), a and b)
                self.assertEqual(self._r("or", a, b), a or b)
                self.assertEqual(self._r("nor", a, b), not (a or b))
            self.assertEqual(self._r("not", a, with_b=False), not a)

    def test_missing_b_raises(self):
        with self.assertRaises(ValueError):
            self._r("and", True, with_b=False)


class RegistryCountP3Test(unittest.TestCase):
    def test_p3_ops_registered(self):
        from xccv.engine.flow.registry import list_nodes

        names = {n["name"] for n in list_nodes()}
        for n in ("crop_by_detection", "grid_tile", "relabel_map", "image_sink",
                  "csv_sink", "bool_op"):
            self.assertIn(n, names)


if __name__ == "__main__":
    unittest.main()
