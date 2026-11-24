# -*- coding: utf-8 -*-
"""滑窗物理裁图导出：把大图按滑动窗切成 tile 图片并同步切分标注。

口径对齐切图训练（engine.dataset.tiling）：
- 窗口起点序列复用 tiling._starts（步长=tile×(1-overlap)，覆盖末边）；
- 归属：形状外接框中心点落在窗口内才算该 tile 的目标；同时落在多个
  重叠窗口时只归首个扫描到的窗口（重叠区去重，同一目标只出现一次）；
- 可见面积比（裁剪后面积 / 原面积）低于 min_visibility 的目标丢弃；
- 无目标 tile 按 seeded 随机保留（bg_tile_keep），同数据集多次执行结果一致；
- mix_full_view 对应"整图额外导出"选项（默认关）。

默认参数 = 训练现行口径：tile=640 / overlap=0.2 / bg_tile_keep=0.15 /
min_visibility=0.3。
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from ..io_utils import imread_unicode
from ..dataset.tiling import _starts
from .labelops import SamplePair, _load_json, scan_pairs

TILE_DEFAULT = 640
OVERLAP_DEFAULT = 0.2
BG_KEEP_DEFAULT = 0.15
MIN_VIS_DEFAULT = 0.3


@dataclass
class TileOutput:
    """一个 tile 导出条目（预览与执行共用）。"""

    source: Path            # 源图路径
    x: int
    y: int
    w: int
    h: int
    full: bool = False      # 整图条目（mix_full_view）
    shapes: list = field(default_factory=list)  # 裁剪后的形状字典（窗口坐标）
    bg_keep: bool = False   # 无目标 tile 被背景保留率抽中
    prefix: str = ""        # 子目录前缀（递归扫描时防同名源图输出互撞）

    @property
    def stem(self) -> str:
        base = f"{self.prefix}_{self.source.stem}" if self.prefix else self.source.stem
        if self.full:
            return f"{base}__full"
        return f"{base}__x{self.x}_y{self.y}"

    @property
    def has_objects(self) -> bool:
        return bool(self.shapes)


def _bbox_of(shape: dict):
    """形状外接框（labelme points 列表 → [x1,y1,x2,y2]）。"""
    pts = [(p[0], p[1]) for p in shape.get("points", []) if len(p) >= 2]
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def _poly_area(pts) -> float:
    if len(pts) < 3:
        return 0.0
    s = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _clip_polygon(points, w, h, x, y):
    """Sutherland–Hodgman：多边形对窗口矩形裁剪，返回窗口坐标点列。"""
    # 四条边依次裁剪；inside 判据按窗口平移后的坐标系
    px = [(p[0] - x, p[1] - y) for p in points]

    def clip_edge(pts, inside, intersect):
        if not pts:
            return []
        out = []
        prev = pts[-1]
        prev_in = inside(prev)
        for cur in pts:
            cur_in = inside(cur)
            if cur_in:
                if not prev_in:
                    out.append(intersect(prev, cur))
                out.append(cur)
            elif prev_in:
                out.append(intersect(prev, cur))
            prev, prev_in = cur, cur_in
        return out

    def ix(a, b, fixed, horizontal):
        # 与 fixed=x 边求交（水平边）：按 x 插值 y；horizontal=False 时按 y 插值 x
        if horizontal:
            t = (fixed - a[0]) / (b[0] - a[0]) if b[0] != a[0] else 0.0
            return (fixed, a[1] + t * (b[1] - a[1]))
        t = (fixed - a[1]) / (b[1] - a[1]) if b[1] != a[1] else 0.0
        return (a[0] + t * (b[0] - a[0]), fixed)

    px = clip_edge(px, lambda p: p[0] >= 0, lambda a, b: ix(a, b, 0, True))
    px = clip_edge(px, lambda p: p[0] <= w, lambda a, b: ix(a, b, w, True))
    px = clip_edge(px, lambda p: p[1] >= 0, lambda a, b: ix(a, b, 0, False))
    px = clip_edge(px, lambda p: p[1] <= h, lambda a, b: ix(a, b, h, False))
    return px


def _clip_shape(shape: dict, w, h, x, y, min_vis: float):
    """把一个形状裁剪进窗口；返回窗口坐标形状字典或 None（被可见率过滤）。"""
    bbox = _bbox_of(shape)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    # 可见面积比：裁剪外接框面积 / 原外接框面积（多边形按裁剪多边形面积）
    ix1, iy1 = max(x1, x), max(y1, y)
    ix2, iy2 = min(x2, x + w), min(y2, y + h)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    full_area = max((x2 - x1) * (y2 - y1), 1e-6)
    st = shape.get("shape_type", "rectangle")
    clipped = None
    if st == "polygon":
        poly = _clip_polygon(shape.get("points", []), w, h, x, y)
        if len(poly) < 3:
            return None
        vis = _poly_area(poly) / max(_poly_area([(p[0], p[1]) for p in shape.get("points", [])]), 1e-6)
        if vis < min_vis:
            return None
        clipped = [[round(p[0], 2), round(p[1], 2)] for p in poly]
    elif st == "circle":
        # 圆按窗口与圆外接框求交，退化为裁剪框矩形（标注语义近似，报告里注明）
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if not (x <= cx < x + w and y <= cy < y + h):
            return None
        clipped = [
            [max(x1, x) - x, max(y1, y) - y],
            [min(x2, x + w) - x, min(y2, y + h) - y],
        ]
    else:
        vis = ((ix2 - ix1) * (iy2 - iy1)) / full_area
        if vis < min_vis:
            return None
        if st == "rectangle":
            clipped = [[ix1 - x, iy1 - y], [ix2 - x, iy2 - y]]
        else:
            # 点/线/线束等：点在窗口内则平移坐标整段保留，否则丢弃
            pts = [(p[0], p[1]) for p in shape.get("points", []) if len(p) >= 2]
            inside_pts = [p for p in pts if x <= p[0] < x + w and y <= p[1] < y + h]
            if not inside_pts:
                return None
            clipped = [[round(p[0] - x, 2), round(p[1] - y, 2)] for p in pts]

    out = dict(shape)
    out["points"] = clipped
    out["flags"] = dict(shape.get("flags") or {})
    return out


def plan_tiles(pairs: list, tile: int = TILE_DEFAULT, overlap: float = OVERLAP_DEFAULT,
               bg_tile_keep: float = BG_KEEP_DEFAULT, min_visibility: float = MIN_VIS_DEFAULT,
               seed: int = 42, export_full: bool = False,
               root: Path | None = None) -> list:
    """生成滑窗裁图计划（只读；需要读图定尺寸的源图逐张读取）。

    pairs 可由 labelops.scan_pairs 生成（未标注图也参与裁图，产出纯背景 tile）。
    root 给出数据集根目录时，子目录内的源图输出名带目录前缀，避免同名互撞。
    """
    rng = random.Random(seed)
    outputs: list = []
    for pair in pairs:
        img_path = pair.image if pair.image.exists() else (
            pair.json.with_suffix(".png") if not pair.labeled else pair.image)
        source = Path(img_path)
        if not source.exists():
            continue
        prefix = ""
        if root is not None:
            try:
                rel_parent = source.parent.relative_to(Path(root))
            except ValueError:
                rel_parent = Path()
            if rel_parent.parts:
                prefix = "_".join(rel_parent.parts)
        img = imread_unicode(str(source))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        shapes = []
        if pair.labeled:
            try:
                data = _load_json(pair.json)
                shapes = data.get("shapes", [])
            except Exception:
                shapes = []
        xs = _starts(iw, tile, overlap)
        ys = _starts(ih, tile, overlap)
        claimed = set()  # (shape_idx, tile_key) 归属去重由 claimed 保证全局唯一
        for y in ys:
            for x in xs:
                th, tw = min(tile, ih - y), min(tile, iw - x)
                item = TileOutput(source=source, x=x, y=y, w=tw, h=th,
                                  prefix=prefix)
                for si, shape in enumerate(shapes):
                    if si in claimed:
                        continue
                    bbox = _bbox_of(shape)
                    if bbox is None:
                        continue
                    bx1, by1, bx2, by2 = bbox
                    cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                    if not (x <= cx < min(x + tile, iw) and y <= cy < min(y + tile, ih)):
                        continue
                    clipped = _clip_shape(shape, tw, th, x, y, min_visibility)
                    if clipped is None:
                        # 可见率过滤也算已归属（与训练语义一致：目标不重复出现）
                        claimed.add(si)
                        continue
                    claimed.add(si)
                    item.shapes.append(clipped)
                if not item.has_objects:
                    if rng.random() >= bg_tile_keep:
                        continue
                    item.bg_keep = True
                outputs.append(item)
        if export_full:
            full = TileOutput(source=source, x=0, y=0, w=iw, h=ih, full=True,
                              shapes=[dict(s) for s in shapes], prefix=prefix)
            outputs.append(full)
    return outputs


def export_tiles(plan: list, dst: Path, progress=None) -> dict:
    """执行裁图导出：写 tile 图片与同名标注 JSON（imageData 不内嵌）。"""
    import cv2

    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    n_img = n_json = 0
    total = len(plan)
    for i, item in enumerate(plan):
        img = imread_unicode(str(item.source))
        if img is None:
            if progress:
                progress(i + 1, total, item.stem)
            continue
        tile_img = img if item.full else img[item.y:item.y + item.h, item.x:item.x + item.w]
        name = item.stem + ".png"
        ok, buf = cv2.imencode(".png", tile_img)
        if ok:
            buf.tofile(str(dst / name))
            n_img += 1
        # 背景 tile 也写空 shapes 的 JSON（负样本训练需要成对文件）
        data = {
            "version": "5.2.1",
            "shapes": item.shapes,
            "imagePath": name,
            "imageData": None,
            "imageHeight": int(tile_img.shape[0]),
            "imageWidth": int(tile_img.shape[1]),
            "flags": {},
        }
        with open(dst / (item.stem + ".json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        n_json += 1
        if progress:
            progress(i + 1, total, item.stem)
    return {"images": n_img, "jsons": n_json, "tiles": total}


def scan_dataset_for_tiles(root: Path) -> list:
    """便捷入口：扫描目录生成可用于 plan_tiles 的样本对。"""
    return scan_pairs(root)
