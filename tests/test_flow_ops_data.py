"""平台测试：3 个历史测试模块的合并模块。

由以下模块按域合并而成，用例逐段原样迁移、断言未改：
- test_v06_flow_ops_p3
- test_v06_flow_regions
- test_v06_flow_unify
"""


# == 来源模块：test_v06_flow_ops_p3 ==
"""检出过滤回归：removed 端口 / 裁图族 / relabel_map / 输出与逻辑算子。

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


# ---------------------------------------------------------------- removed 端口过滤
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


# ---------------------------------------------------------------- 裁图族
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


# ---------------------------------------------------------------- relabel_map
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


# ---------------------------------------------------------------- 输出与逻辑算子
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

# == 来源模块：test_v06_flow_regions ==
"""检出过滤区域解析器（矩形 / 圆 / 多边形，旧串零改动兼容）。

- 旧串回归：随机矩形（归一化/像素/坐标颠倒/中文逗号/坏条目混杂）与 V0.5.6
  旧实现逐点判定一致；
- 新形状：circle（归一化半径按短边、像素）与 polygon 命中/未命中/边界；
- 混合区域串、apply_flow 的屏蔽/保留区域端到端；
- 打包严格校验：旧文案逐字不变 + 新形状错误文案；
- 过滤节点 validate（流程图打包口径）接受新形状。
"""

import random
import unittest

import numpy as np

from xccv.engine.flow import regions as rg


def _legacy_parse(value, w, h):
    """V0.5.6 detection_flow._parse_regions 原样（参考实现）。"""
    out = []
    if isinstance(value, str):
        entries = [s for s in value.split(";") if s.strip()]
    elif isinstance(value, (list, tuple)):
        entries = list(value)
    else:
        return out
    for e in entries:
        try:
            if isinstance(e, str):
                nums = [float(v) for v in e.replace("，", ",").split(",") if v.strip()]
            elif isinstance(e, (list, tuple)):
                nums = [float(v) for v in e]
            else:
                continue
            if len(nums) != 4:
                continue
            x1, y1, x2, y2 = nums
            if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.0 + 1e-9:
                x1, x2 = x1 / w, x2 / w
                y1, y2 = y1 / h, y2 / h
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            out.append([x1, y1, x2, y2])
        except (TypeError, ValueError):
            continue
    return out


def _legacy_in_any(cx, cy, regions, w, h):
    for x1, y1, x2, y2 in _legacy_parse(regions, w, h):
        if x1 * w <= cx <= x2 * w and y1 * h <= cy <= y2 * h:
            return True
    return False


class LegacyRectCompatTest(unittest.TestCase):
    def test_random_legacy_strings_match_old_impl(self):
        rnd = random.Random(20261102)
        w, h = 640, 480
        for _ in range(300):
            parts = []
            for _k in range(rnd.randint(1, 3)):
                if rnd.random() < 0.5:  # 归一化
                    vals = [round(rnd.random(), 3) for _ in range(4)]
                else:  # 像素（含颠倒坐标）
                    vals = [rnd.randint(0, w), rnd.randint(0, h),
                            rnd.randint(0, w), rnd.randint(0, h)]
                sep = "，" if rnd.random() < 0.2 else ","
                parts.append(sep.join(str(v) for v in vals))
            if rnd.random() < 0.2:
                parts.append("a,b,c,d")      # 坏条目：两边都应跳过
            if rnd.random() < 0.2:
                parts.append("1,2,3")        # 坏条目
            value = ";".join(parts)
            for _p in range(10):
                cx, cy = rnd.uniform(0, w), rnd.uniform(0, h)
                self.assertEqual(rg.any_contains(value, cx, cy, w, h),
                                 _legacy_in_any(cx, cy, value, w, h), value)

    def test_list_form_and_boundaries_match(self):
        w, h = 100, 50
        value = [[0.1, 0.2, 0.5, 0.6], [80, 10, 60, 40]]   # 第二条像素+颠倒
        for cx, cy in [(10, 10), (50, 30), (60, 10), (80, 40), (70, 25), (0, 0), (99, 49)]:
            self.assertEqual(rg.any_contains(value, cx, cy, w, h),
                             _legacy_in_any(cx, cy, value, w, h), (cx, cy))

    def test_rect_prefix_equals_bare(self):
        w, h = 200, 100
        for cx, cy in [(20, 20), (100, 50), (150, 90)]:
            self.assertEqual(rg.any_contains("rect:0.1,0.1,0.6,0.6", cx, cy, w, h),
                             rg.any_contains("0.1,0.1,0.6,0.6", cx, cy, w, h))


class NewShapesTest(unittest.TestCase):
    def test_circle_normalized_radius_by_short_side(self):
        w, h = 200, 100       # 短边 100 → r=0.2 即 20px
        region = "circle:0.5,0.5,0.2"
        self.assertTrue(rg.any_contains(region, 100, 50, w, h))      # 圆心
        self.assertTrue(rg.any_contains(region, 120, 50, w, h))      # 边界（距 20）
        self.assertFalse(rg.any_contains(region, 121, 50, w, h))     # 刚出界
        self.assertFalse(rg.any_contains(region, 100, 75, w, h))

    def test_circle_pixel(self):
        region = "circle:300,200,50"
        self.assertTrue(rg.any_contains(region, 330, 230, 640, 480))  # 距 42.4
        self.assertFalse(rg.any_contains(region, 340, 240, 640, 480))  # 距 56.6

    def test_polygon_triangle_hit_miss_boundary(self):
        region = "polygon:10,10,110,10,10,110"  # 直角三角形（像素）
        w, h = 200, 200
        self.assertTrue(rg.any_contains(region, 30, 30, w, h))
        self.assertTrue(rg.any_contains(region, 60, 10, w, h))       # 边上
        self.assertFalse(rg.any_contains(region, 90, 90, w, h))      # 斜边外侧
        self.assertFalse(rg.any_contains(region, 5, 5, w, h))

    def test_polygon_normalized(self):
        region = "polygon:0,0,1,0,1,0.5"
        self.assertTrue(rg.any_contains(region, 90, 10, 100, 100))
        self.assertFalse(rg.any_contains(region, 10, 90, 100, 100))

    def test_mixed_entries_union(self):
        value = "0,0,0.1,0.1;circle:0.9,0.9,0.05;polygon:40,40,60,40,50,60"
        w, h = 100, 100
        self.assertTrue(rg.any_contains(value, 5, 5, w, h))
        self.assertTrue(rg.any_contains(value, 90, 90, w, h))
        self.assertTrue(rg.any_contains(value, 50, 45, w, h))
        self.assertFalse(rg.any_contains(value, 30, 70, w, h))

    def test_bad_new_shape_entries_skipped_at_runtime(self):
        value = "circle:1,2;polygon:1,1,2,2;ellipse:1,2,3;0,0,0.5,0.5"
        self.assertEqual(len(rg.parse_regions(value)), 1)
        self.assertTrue(rg.any_contains(value, 10, 10, 100, 100))


class ApplyFlowRegionShapesTest(unittest.TestCase):
    def _pred(self):
        from xccv.engine.models.base import Prediction

        # 三个检出：中心 (20,20)、(100,50)、(180,80)
        return Prediction(labels=[0, 0, 0], scores=[0.9, 0.8, 0.7],
                          boxes=[[10, 10, 30, 30], [90, 40, 110, 60], [170, 70, 190, 90]])

    def test_region_block_circle(self):
        from xccv.engine.detection_flow import apply_flow

        flow = {"nodes": [{"op": "region_block", "params": {"regions": "circle:100,50,15"}}]}
        out, info = apply_flow(self._pred(), (200, 100), flow)
        self.assertEqual(len(out.boxes), 2)
        self.assertEqual(info["removed"], 1)

    def test_region_keep_polygon(self):
        from xccv.engine.detection_flow import apply_flow

        flow = {"nodes": [{"op": "region_keep",
                           "params": {"regions": "polygon:0,0,120,0,120,100,0,100"}}]}
        out, _info = apply_flow(self._pred(), (200, 100), flow)
        self.assertEqual(len(out.boxes), 2)   # 第三个中心 x=180 在多边形外

    def test_legacy_rect_chain_unchanged(self):
        from xccv.engine.detection_flow import apply_flow

        flow = {"nodes": [{"op": "region_block", "params": {"regions": "0,0,0.25,0.5"}}]}
        out, _info = apply_flow(self._pred(), (200, 100), flow)
        self.assertEqual([b[0] for b in out.boxes], [90.0, 170.0])


class PackValidationTest(unittest.TestCase):
    def test_legacy_messages_verbatim(self):
        from xccv.engine.detection_flow import region_param_problems

        self.assertEqual(region_param_problems("1,2,3"),
                         ["第 1 条区域不是 4 个数字（应为 x1,y1,x2,y2）"])
        self.assertEqual(region_param_problems("5,5,5,5"), ["第 1 条区域宽或高为零/为负"])
        self.assertEqual(region_param_problems("a,b,c,d"), ["第 1 条区域不是有效的数字"])
        self.assertEqual(region_param_problems(12, "节点X："), ["节点X：区域参数类型无效"])
        self.assertEqual(region_param_problems(None), [])
        self.assertEqual(region_param_problems(0), [])

    def test_new_shape_problems(self):
        probs = rg.region_problems(
            "circle:1,2;circle:0.5,0.5,0;polygon:0,0,1,1;polygon:0,0,1,1,2,2;ellipse:1,2,3")
        self.assertEqual(len(probs), 5)
        self.assertIn("圆应为 3 个数字", probs[0])
        self.assertIn("半径", probs[1])
        self.assertIn("至少 3 个顶点", probs[2])
        self.assertIn("面积为零", probs[3])
        self.assertIn("未知区域形状", probs[4])

    def test_valid_new_shapes_pass(self):
        self.assertEqual(rg.region_problems(
            "rect:0.1,0.1,0.5,0.5;circle:0.5,0.5,0.2;polygon:0,0,10,0,5,8"), [])

    def test_filter_node_validate_accepts_new_shapes(self):
        from xccv.engine.flow.legacy_adapter import FilterRegionBlock

        node = FilterRegionBlock()
        self.assertEqual(node.validate({"regions": "circle:0.5,0.5,0.1"}), [])
        self.assertTrue(node.validate({"regions": "circle:0.5,0.5"}))


class RegionGeometryTest(unittest.TestCase):
    def test_to_pixels_shapes(self):
        r = rg.parse_regions("circle:0.5,0.5,0.25")[0]
        self.assertEqual(r.to_pixels(400, 200), (200.0, 100.0, 50.0))
        poly = rg.parse_regions("polygon:0,0,0.5,0,0.5,0.5")[0].to_pixels(100, 100)
        np.testing.assert_allclose(poly, [[0, 0], [50, 0], [50, 50]])

# == 来源模块：test_v06_flow_unify ==
"""回归：流程图端口彻底统一。

- image_list 端口类型取消：PORT_TYPES 只余 image/result/number，全库节点
  无 image_list 端口（多张集合走 image 端口数据载体 ImageList）；
- 并行输入自动映射（executor._map_over）：多张集合过任意非 HANDLES_LIST
  节点逐张执行；标量输入广播；输出重组 ImageList/ResultList；
  图像项 meta 继承输入项（roi_rect 沿链传递）；结果集合解包 ResultItem.value；
  张数不一致报人话错误；
- 主结果端口标签统一「结果」（定位/测量/AI/过滤族），特殊项数值端口保留专名；
- 结果合并生成即裁剪（前端 onAdded，schema 侧断言 HANDLES_LIST 与迁移）；
- 旧图迁移 schema 3：list_pick 节点连同连线删除。
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from xccv.engine.flow.executor import Executor
from xccv.engine.flow.ir import (PORT_TYPES, Geometry, ImageData, ImageList,
                                 NumberVal, ResultItem, ResultList)
from xccv.engine.flow.node_base import RunContext, VisionNode
from xccv.engine.flow.port_migrate import SCHEMA, migrate_ports
from xccv.engine.flow.registry import get_node_class, list_nodes
from xccv.modeltools.flow_session import ensure_ops_registered

ensure_ops_registered()


def _lib():
    return {info["name"]: info for info in list_nodes()}


class PortUnifyTest(unittest.TestCase):
    """端口类型与标签统一（批注二.1/二.2/二.6）。"""

    def test_port_types_three(self):
        self.assertEqual(set(PORT_TYPES), {"image", "result", "number"})

    def test_no_image_list_ports(self):
        for info in list_nodes():
            for p in info["inputs"] + info["outputs"]:
                self.assertNotEqual(p["type"], "image_list",
                                    (info["name"], p["name"]))

    def test_primary_result_labels(self):
        """主结果输出统一「结果」；特殊项数值端口保留专名。"""
        lib = _lib()
        for name in ("find_line", "find_circle", "find_corner", "template_match",
                     "fit_rectangle", "find_line_pair", "blob_analysis",
                     "measure_distance", "measure_angle", "detection_to_measure",
                     "model_detection", "merge_results",
                     "model_on_list", "op_on_list", "mask_to_box",
                     "filter_overlap", "filter_topk", "relabel_map"):
            outs = lib[name]["outputs"]
            r = next(p for p in outs if p["name"] == "result")
            self.assertEqual(r["label"], "结果", name)
        # 特殊项（+特殊项口径）：数值端口保留专名
        lib2 = {info["name"]: info for info in list_nodes()}
        self.assertEqual(next(p for p in lib2["find_line"]["outputs"]
                              if p["name"] == "residual")["label"], "拟合残差")
        self.assertEqual(next(p for p in lib2["template_match"]["outputs"]
                              if p["name"] == "score")["label"], "匹配得分")

    def test_all_ports_chinese_labels(self):
        """全部端口有中文 label（画布不再出现 image/number 英文裸名，批注二.2/二.6）。"""
        import re

        for info in list_nodes():
            if info["name"].startswith("custom/"):
                continue   # 自定义算子端口名用户自定，豁免
            for p in info["inputs"] + info["outputs"]:
                self.assertTrue(re.match(r"^[\u4e00-\u9fff]", p["label"]),
                                (info["name"], p["name"], p["label"]))

    def test_native_list_nodes_flagged(self):
        for name in ("merge_results", "model_on_list", "op_on_list"):
            self.assertTrue(get_node_class(name).HANDLES_LIST, name)
        for name in ("gaussian_blur", "find_circle", "compare_value", "math_op",
                     "crop_by_detection_all"):
            self.assertFalse(get_node_class(name).HANDLES_LIST, name)

    def test_list_pick_unregistered(self):
        self.assertIsNone(get_node_class("list_pick"))


def _tiles():
    """三张 50x40 亮圆裁图（局部坐标同 (25,20)），roi 依次右移 100px。"""
    import cv2

    items = []
    for k in range(3):
        m = np.zeros((40, 50), np.uint8)
        cv2.circle(m, (25, 20), 12, 230, -1)
        items.append(ImageData(mat=cv2.cvtColor(m, cv2.COLOR_GRAY2BGR),
                               meta={"roi_rect": [k * 100, 0, k * 100 + 50, 40],
                                     "source_size": [300, 40]}))
    return ImageList(items=items, meta={"source_size": [300, 40]})


class AutoMapTest(unittest.TestCase):
    """并行输入自动映射（批注二.4 的执行器语义）。"""

    def _run_graph(self, nodes, links, external=None):
        res = Executor({"nodes": nodes, "links": links}, RunContext(),
                       external=external or {}).run()
        errs = {k: r.error for k, r in res.items() if not r.ok}
        self.assertFalse(errs, errs)
        return res

    def test_blur_over_tiles_meta_inherited(self):
        g = {"nodes": [{"id": 1, "type": "gaussian_blur", "properties": {}}],
             "links": []}
        res = self._run_graph(g["nodes"], g["links"], external={(1, 0): _tiles()})
        out = res[1].outputs["image"]
        self.assertIsInstance(out, ImageList)
        self.assertEqual(len(out), 3)
        self.assertEqual(out.items[1].meta["roi_rect"], [100, 0, 150, 40])
        self.assertEqual(out.items[1].meta["source_size"], [300, 40])

    def test_locate_over_tiles_then_merge_offsets(self):
        """多张过定位节点：结果集合带 roi，合并回原图坐标逐张偏移。"""
        g = {"nodes": [
            {"id": 1, "type": "gaussian_blur", "properties": {}},
            {"id": 2, "type": "find_circle", "properties": {"polarity": "bright"}},
            {"id": 3, "type": "merge_results", "properties": {"nms_iou": 0.0}}],
            "links": [[1, 1, 0, 2, 0], [2, 2, 1, 3, 1]]}
        res = self._run_graph(g["nodes"], g["links"], external={(1, 0): _tiles()})
        rl = res[2].outputs["result"]
        self.assertIsInstance(rl, ResultList)
        self.assertEqual([it.roi_rect for it in rl.items],
                         [[0, 0, 50, 40], [100, 0, 150, 40], [200, 0, 250, 40]])
        boxes = res[3].outputs["result"].boxes
        self.assertEqual(len(boxes), 3)
        for k, b in enumerate(boxes):
            self.assertAlmostEqual((b[0] + b[2]) / 2, 100 * k + 25, delta=1.5)
            self.assertAlmostEqual((b[1] + b[3]) / 2, 20, delta=1.5)

    def test_result_list_unwrapped_to_values(self):
        """结果集合逐张映射解包 ResultItem.value（节点拿原值而非壳）。"""
        rl = ResultList(items=[
            ResultItem("number", NumberVal(value=float(v)), meta={"index": k})
            for k, v in enumerate((2.0, 8.0))])
        g = {"nodes": [{"id": 1, "type": "compare_value",
                        "properties": {"op": "gt", "threshold": 5.0}}],
             "links": []}
        res = self._run_graph(g["nodes"], g["links"], external={(1, 0): rl})
        out = res[1].outputs["result"]
        self.assertIsInstance(out, ResultList)
        self.assertEqual([it.value for it in out.items], [False, True])
        self.assertEqual([it.kind for it in out.items], ["bool", "bool"])

    def test_math_over_result_list(self):
        rl = ResultList(items=[
            ResultItem("number", NumberVal(value=10.0), meta={"index": 0}),
            ResultItem("number", NumberVal(value=30.0), meta={"index": 1})])
        g = {"nodes": [{"id": 1, "type": "math_op", "properties": {"op": "div"}}],
             "links": []}
        res = Executor(g, RunContext(),
                       external={(1, 0): rl, (1, 1): NumberVal(value=2.0)}).run()
        out = res[1].outputs["number"]
        self.assertIsInstance(out, ResultList)
        self.assertEqual([it.value.value for it in out.items], [5.0, 15.0])

    def test_length_mismatch_raises(self):
        a = _tiles()
        b = ImageList(items=a.items[:2])
        g = {"nodes": [{"id": 1, "type": "condition_branch", "properties": {}}],
             "links": []}
        ex = Executor(g, RunContext(),
                      external={(1, 0): True, (1, 1): a, (1, 2): b})
        res = ex.run()
        self.assertFalse(res[1].ok)
        self.assertIn("张数不一致", res[1].error)

    def test_single_image_untouched(self):
        """单张（非集合）输入不走映射，行为与旧版一致。"""
        img = ImageData(mat=np.full((40, 50, 3), 90, np.uint8))
        g = {"nodes": [{"id": 1, "type": "gaussian_blur", "properties": {}}],
             "links": []}
        res = self._run_graph(g["nodes"], g["links"], external={(1, 0): img})
        out = res[1].outputs["image"]
        self.assertIsInstance(out, ImageData)
        self.assertEqual(out.mat.shape, (40, 50, 3))

    def test_model_on_list_accepts_single(self):
        """原生集合节点收到单张 ImageData 也兼容（包成 1 张集合）。"""
        from xccv.engine.flow.ops.listops import _as_list

        img = ImageData(mat=np.zeros((4, 4, 3), np.uint8))
        lst = _as_list(img)
        self.assertEqual(len(lst), 1)
        self.assertIs(lst.items[0], img)


class MigrateSchema3Test(unittest.TestCase):
    """旧图迁移 schema 3：list_pick 连点带线删除（批注二.4）。"""

    def test_list_pick_dropped(self):
        flow = {"schema": 2, "last_node_id": 3, "last_link_id": 2,
                "nodes": [
                    {"id": 1, "type": "xccv2/grid_tile_all", "properties": {}},
                    {"id": 2, "type": "xccv2/list_pick", "properties": {"index": 0}},
                    {"id": 3, "type": "xccv2/gaussian_blur", "properties": {}}],
                "links": [[1, 1, 0, 2, 0], [2, 2, 0, 3, 0]]}
        out = migrate_ports(flow)
        self.assertEqual(SCHEMA, 3)
        self.assertEqual(out["schema"], 3)
        self.assertEqual({n["type"] for n in out["nodes"]},
                         {"xccv2/grid_tile_all", "xccv2/gaussian_blur"})
        self.assertEqual(out["links"], [])
        # 原图不被修改（浅拷贝纪律）
        self.assertEqual(len(flow["nodes"]), 3)
        self.assertEqual(len(flow["links"]), 2)

    def test_schema3_untouched(self):
        flow = {"schema": 3, "nodes": [], "links": []}
        self.assertIs(migrate_ports(flow), flow)

if __name__ == '__main__':
    unittest.main()
