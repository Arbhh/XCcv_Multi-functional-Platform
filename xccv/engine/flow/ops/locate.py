"""定位类算子（5 个，）。

- blob_analysis：连通域斑点分析，输出包装为 Prediction（复用旧过滤链/
  数量判定/测试平台可视化，全库耦合价值最高的一点）；
- find_line：卡尺组沿扫描方向取边缘/线中心（亚像素）→ cv2.fitLine 鲁棒拟合；
- find_circle：以粗中心为圆心的同心卡尺（射线）取边界点（亚像素）→
  cv2.fitEllipse 拟合圆；
- find_corner：harris 角点 或 caliper_cross 十字卡尺单角点；
- template_match：模板匹配（硬性性能前置：ROI 限界 + 0.25x/0.5x 金字塔
  粗匹配 → 原分辨率局部精匹配；angle_step 默认 5°；max_matches 默认 1；
  模板录制 record(image, rect) 落「流程模板/」目录）。

复杂度说明（template_match）：搜索代价 ∝ 搜索区面积 × 角度数 × 模板面积。
金字塔把粗匹配压到 1/16 面积，精匹配只在候选点局部窗口内进行；8K 全图
多角度仍会慢，产线上请务必用 roi 参数限定搜索区。
"""

import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ..ir import Geometry, NumberVal
from ..node_base import ParamSpec, Port, VisionNode
from ..registry import register_node
from . import to_gray

_GRAY = "彩色输入自动转灰度；灰度原样处理。"


def _parabolic_offset(ym1: float, y0: float, yp1: float) -> float:
    """三点抛物线极值偏移（亚像素）：峰在 y0 处，返回 [-1, 1] 内的偏移。"""
    denom = ym1 - 2.0 * y0 + yp1
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (ym1 - yp1) / denom, -1.0, 1.0))


def _parse_roi(text: str, w: int, h: int) -> tuple[int, int, int, int]:
    """解析 "x1,y1,x2,y2" 像素 ROI；空串=全图。越界自动钳位。"""
    text = str(text or "").strip()
    if not text:
        return 0, 0, w, h
    nums = [float(v) for v in text.replace("，", ",").split(",") if v.strip()]
    if len(nums) != 4:
        raise ValueError("ROI 应为 4 个数字（x1,y1,x2,y2，像素坐标），或留空表示全图")
    x1, y1, x2, y2 = nums
    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI 宽或高为零/为负")
    return (max(0, int(x1)), max(0, int(y1)), min(w, int(round(x2))), min(h, int(round(y2))))


@register_node
class BlobAnalysis(VisionNode):
    """斑点分析：阈值分割 → 连通域 → 面积/圆度滤波，输出 AI 检出格式。

    输出为 Prediction（labels=0、score=1.0、boxes 为 xyxy），旧过滤链/
    数量判定/可视化直接可用；centers 端口输出各斑点中心点几何。
    圆度 = 4πA/P²（圆为 1，细长条远小于 1）。
    """

    NAME = "blob_analysis"
    CATEGORY = "定位"
    INPUTS = (Port("image", "image", label="图像"),)
    OUTPUTS = (Port("image", "image", label="图像"),
               Port("result", "result", label="结果"))
    PARAMS = {
        "threshold_mode": ParamSpec("threshold_mode", "二值化", "enum", "otsu",
                                    options=["otsu", "fixed"]),
        "thresh": ParamSpec("thresh", "固定阈值", "float", 128.0, range=(0.0, 255.0),
                            tooltip="仅 fixed 模式生效"),
        "invert": ParamSpec("invert", "暗目标", "bool", False,
                            tooltip="勾选：暗底亮目标反转后分析（亮底暗目标时使用）"),
        "min_area": ParamSpec("min_area", "最小面积", "int", 10, range=(0, 10 ** 7),
                              tooltip="像素²，小于该面积的斑点丢弃"),
        "max_area": ParamSpec("max_area", "最大面积", "int", 0, range=(0, 10 ** 8),
                              tooltip="0 表示不限制"),
        "circularity": ParamSpec("circularity", "圆度下限", "float", 0.0,
                                 range=(0.0, 1.0), tooltip="0 表示不过滤圆度"),
        "label_name": ParamSpec("label_name", "检出名称", "text", "目标",
                                tooltip="结果 JSON 中 detections.label 的对外"
                                        "暴露名称（可自定义）"),
    }

    def run(self, inputs, params, ctx):
        gray = to_gray(inputs["image"].mat)
        if params.get("invert"):
            gray = 255 - gray
        if params["threshold_mode"] == "otsu":
            _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            _, bin_img = cv2.threshold(gray, float(params["thresh"]), 255,
                                       cv2.THRESH_BINARY)
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN,
                                   np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        min_area = float(params["min_area"])
        max_area = float(params["max_area"])
        circ_min = float(params["circularity"])
        boxes = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < max(min_area, 1.0):
                continue
            if max_area and area > max_area:
                continue
            if circ_min > 0:
                peri = cv2.arcLength(c, True)
                circ = (4 * math.pi * area / (peri * peri)) if peri > 0 else 0.0
                if circ < circ_min:
                    continue
            x, y, bw, bh = cv2.boundingRect(c)
            boxes.append([float(x), float(y), float(x + bw), float(y + bh)])
        from ...models.base import Prediction  # 惰性导入：flow 包保持轻导入

        # 检出名称：写入会话供结果收集器作为类别名表使用
        label_name = str(params.get("label_name") or "").strip()
        if label_name:
            ctx.set("flow_class_names", [label_name])
        return {"result": Prediction(labels=[0] * len(boxes),
                                     scores=[1.0] * len(boxes),
                                     boxes=boxes),
                "image": inputs["image"]}


@register_node
class FindLine(VisionNode):
    """直线定位：卡尺组沿扫描方向逐条找边缘/线中心（亚像素）→ 鲁棒拟合。

    - 目标是工件边界（明暗台阶）用 edge；目标是划线/激光线（线比背景
      亮或暗）用 line；
    - 目标线大致水平 → scan=vertical（逐列竖扫）；大致竖直 →
      scan=horizontal（逐行横扫）；
    - roi 为像素坐标 "x1,y1,x2,y2"，留空=全图（建议给出以排除干扰）。
    """

    NAME = "find_line"
    CATEGORY = "定位"
    INPUTS = (Port("image", "image", label="图像"),)
    OUTPUTS = (Port("image", "image", label="图像"),
               Port("result", "result", label="结果"),
               Port("residual", "number", label="拟合残差"))
    PARAMS = {
        "target": ParamSpec("target", "目标类型", "enum", "edge",
                            options=["edge", "line"],
                            tooltip="edge=明暗边界；line=细线中心"),
        "scan": ParamSpec("scan", "卡尺方向", "enum", "vertical",
                          options=["vertical", "horizontal"],
                          tooltip="vertical=逐列竖扫（找水平线）；"
                                  "horizontal=逐行横扫（找竖直线）"),
        "roi": ParamSpec("roi", "测量区", "text", "",
                         tooltip="x1,y1,x2,y2 像素坐标；留空=全图"),
        "num_calipers": ParamSpec("num_calipers", "卡尺数量", "int", 32,
                                  range=(4, 256)),
        "polarity": ParamSpec("polarity", "极性", "enum", "any",
                              options=["any", "bright", "dark"],
                              tooltip="edge：bright=暗→亮；line：bright=线比背景亮"),
        "min_contrast": ParamSpec("min_contrast", "最小对比度", "float", 20.0,
                                  range=(1.0, 255.0)),
    }

    def run(self, inputs, params, ctx):
        gray = to_gray(inputs["image"].mat)
        h, w = gray.shape[:2]
        x1, y1, x2, y2 = _parse_roi(params.get("roi"), w, h)
        n = max(4, int(params["num_calipers"]))
        vertical = params["scan"] == "vertical"
        span = np.linspace(x1, x2 - 1, n).astype(int) if vertical else \
            np.linspace(y1, y2 - 1, n).astype(int)
        points = []
        for s in span:
            col = (gray[y1:y2, s].astype(np.float32) if vertical
                   else gray[s, x1:x2].astype(np.float32))
            hit = self._locate(col, params)
            if hit is None:
                continue
            if vertical:
                points.append((float(s), y1 + float(hit)))
            else:
                points.append((x1 + float(hit), float(s)))
        if len(points) < 2:
            raise ValueError("有效边缘点不足 2 个：请检查最小对比度/极性/测量区")
        pts = np.asarray(points, dtype=np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        corners = np.array([(x1, y1), (x2 - 1, y1), (x2 - 1, y2 - 1), (x1, y2 - 1)],
                           dtype=np.float64)
        t = (corners[:, 0] - x0) * vx + (corners[:, 1] - y0) * vy
        pa = np.array([x0 + t.min() * vx, y0 + t.min() * vy])
        pb = np.array([x0 + t.max() * vx, y0 + t.max() * vy])
        line_dir = np.array([vx, vy], dtype=np.float64)
        resid = float(np.mean(np.abs((pts - [x0, y0]) - np.outer(
            (pts - [x0, y0]) @ line_dir, line_dir))))
        geo = Geometry(kind="line", data=np.array([pa, pb]), subpixel=True)
        return {"result": geo, "residual": NumberVal(value=round(resid, 3), unit="px"),
                "image": inputs["image"]}

    @staticmethod
    def _locate(profile: np.ndarray, params: dict):
        """单条卡尺：返回剖面内的亚像素位置（浮点）或 None。"""
        p = profile
        if params["target"] == "edge":
            if len(p) < 3:
                return None
            g = (p[2:] - p[:-2]) / 2.0            # 中心差分梯度
            pol = params["polarity"]
            score = (g if pol == "bright" else -g) if pol != "any" else np.abs(g)
            i = int(np.argmax(score))
            if score[i] < float(params["min_contrast"]):
                return None
            off = _parabolic_offset(score[i - 1], score[i], score[i + 1]) \
                if 0 < i < len(score) - 1 else 0.0
            return i + 1 + off
        pol = params["polarity"]
        score = (p if pol == "bright" else -p) if pol != "any" \
            else (p - p.mean())
        i = int(np.argmax(score))
        if abs(float(score[i])) < float(params["min_contrast"]):
            return None
        off = _parabolic_offset(score[i - 1], score[i], score[i + 1]) \
            if 0 < i < len(score) - 1 else 0.0
        return i + off


@register_node
class FindCircle(VisionNode):
    """圆定位：粗定位中心 → 同心卡尺（射线）取边界点（亚像素）→ 椭圆拟合。

    输出 circle 几何 [cx, cy, r]。极性描述**目标明暗**：bright=亮目标
    （暗底亮圆，射线由内向外为亮→暗）；dark=暗目标（孔/暗斑）；any=自动。
    roi 可排除邻近干扰。
    """

    NAME = "find_circle"
    CATEGORY = "定位"
    INPUTS = (Port("image", "image", label="图像"),)
    OUTPUTS = (Port("image", "image", label="图像"),
               Port("result", "result", label="结果"))
    PARAMS = {
        "roi": ParamSpec("roi", "测量区", "text", "",
                         tooltip="x1,y1,x2,y2 像素坐标；留空=全图"),
        "num_rays": ParamSpec("num_rays", "射线数", "int", 36, range=(12, 180),
                              tooltip="同心卡尺数量（越多越稳、越慢）"),
        "polarity": ParamSpec("polarity", "极性", "enum", "any",
                              options=["any", "bright", "dark"],
                              tooltip="目标明暗：bright=亮目标；dark=暗目标"),
        "min_contrast": ParamSpec("min_contrast", "最小对比度", "float", 20.0,
                                  range=(1.0, 255.0)),
    }

    def run(self, inputs, params, ctx):
        gray = to_gray(inputs["image"].mat)
        h, w = gray.shape[:2]
        x1, y1, x2, y2 = _parse_roi(params.get("roi"), w, h)
        cx, cy = self._rough_center(gray, (x1, y1, x2, y2))
        k = max(12, int(params["num_rays"]))
        r_max = min(cx - x1, x2 - 1 - cx, cy - y1, y2 - 1 - cy)
        if r_max < 4:
            raise ValueError("测量区太小：圆心到 ROI 边界不足 4 像素")
        steps = min(int(r_max) - 1, 200)
        rr = np.linspace(2.0, r_max - 1.0, max(steps, 8), dtype=np.float32)
        angles = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
        map_x = (cx + np.outer(np.ones(k), rr) * np.cos(angles[:, None])).astype(np.float32)
        map_y = (cy + np.outer(np.ones(k), rr) * np.sin(angles[:, None])).astype(np.float32)
        profile = cv2.remap(np.ascontiguousarray(gray, dtype=np.float32),
                            map_x, map_y, cv2.INTER_LINEAR)
        grad = (profile[:, 2:] - profile[:, :-2]) / 2.0
        # 射线由目标内部向外：亮目标在边界为亮→暗（梯度负），故取反
        pol = params["polarity"]
        score = (-grad if pol == "bright" else grad) if pol != "any" \
            else np.abs(grad)
        th = float(params["min_contrast"])
        pts = []
        for row in range(k):
            i = int(np.argmax(score[row]))
            if score[row, i] < th:
                continue
            off = _parabolic_offset(score[row, i - 1], score[row, i],
                                    score[row, i + 1]) if 0 < i < grad.shape[1] - 1 else 0.0
            r_sub = float(rr[i + 1] + off)
            pts.append((cx + r_sub * math.cos(angles[row]),
                        cy + r_sub * math.sin(angles[row])))
        if len(pts) < 6:
            raise ValueError("有效边界点不足 6 个：请检查对比度/极性/测量区")
        ellipse = cv2.fitEllipse(np.asarray(pts, dtype=np.float32))
        (ecx, ecy), (ma, mb), _ang = ellipse
        geo = Geometry(kind="circle",
                       data=np.array([ecx, ecy, (ma + mb) / 4.0]), subpixel=True)
        return {"result": geo, "image": inputs["image"]}

    @staticmethod
    def _rough_center(gray, roi):
        """粗定位：Otsu 最大外轮廓的质心；失败用 ROI 中心（射线仍可能找回）。"""
        x1, y1, x2, y2 = roi
        patch = gray[y1:y2, x1:x2]
        _, bin_img = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            m = cv2.moments(c)
            if m["m00"]:
                return x1 + m["m10"] / m["m00"], y1 + m["m01"] / m["m00"]
        return (x1 + x2 - 1) / 2.0, (y1 + y2 - 1) / 2.0


@register_node
class FindCorner(VisionNode):
    """角点定位：harris 多角点 或 caliper_cross 十字卡尺单角点。

    - harris：经典角点检测，返回最多 max_corners 个点（按响应强度）；
    - caliper_cross：找测量区内最强的水平边缘行与竖直边缘列，交点即
      角点（适合 L 形/矩形单一角点，亚像素）。
    """

    NAME = "find_corner"
    CATEGORY = "定位"
    INPUTS = (Port("image", "image", label="图像"),)
    OUTPUTS = (Port("image", "image", label="图像"),
               Port("result", "result", label="结果"))
    PARAMS = {
        "mode": ParamSpec("mode", "模式", "enum", "harris",
                          options=["harris", "caliper_cross"]),
        "roi": ParamSpec("roi", "测量区", "text", "",
                         tooltip="x1,y1,x2,y2 像素坐标；留空=全图"),
        "max_corners": ParamSpec("max_corners", "角点上限", "int", 10,
                                 range=(1, 100), tooltip="仅 harris 模式"),
        "quality": ParamSpec("quality", "质量系数", "float", 0.1,
                             range=(0.01, 1.0), tooltip="仅 harris 模式"),
        "min_dist": ParamSpec("min_dist", "最小间距", "int", 10, range=(1, 500),
                              tooltip="仅 harris 模式（像素）"),
    }

    def run(self, inputs, params, ctx):
        gray = to_gray(inputs["image"].mat)
        h, w = gray.shape[:2]
        x1, y1, x2, y2 = _parse_roi(params.get("roi"), w, h)
        patch = np.ascontiguousarray(gray[y1:y2, x1:x2])
        if params["mode"] == "harris":
            res = cv2.goodFeaturesToTrack(
                patch, maxCorners=int(params["max_corners"]),
                qualityLevel=float(params["quality"]),
                minDistance=int(params["min_dist"]), useHarrisDetector=True)
            if res is None:
                raise ValueError("未找到角点：可降低质量系数或检查测量区")
            pts = res.reshape(-1, 2) + [x1, y1]
            return {"result": Geometry(kind="point", data=pts.astype(float),
                                       subpixel=False),
                    "image": inputs["image"]}
        f = patch.astype(np.float32)
        gy = np.abs(np.gradient(f, axis=0)).mean(axis=1)   # 每行水平边缘强度
        gx = np.abs(np.gradient(f, axis=1)).mean(axis=0)   # 每列竖直边缘强度
        iy, ix = int(np.argmax(gy)), int(np.argmax(gx))
        if gy[iy] < 1.0 and gx[ix] < 1.0:
            raise ValueError("未找到边缘：测量区内无明暗变化")
        oy = _parabolic_offset(gy[iy - 1], gy[iy], gy[iy + 1]) \
            if 0 < iy < len(gy) - 1 else 0.0
        ox = _parabolic_offset(gx[ix - 1], gx[ix], gx[ix + 1]) \
            if 0 < ix < len(gx) - 1 else 0.0
        data = np.array([[x1 + ix + ox, y1 + iy + oy]])
        return {"result": Geometry(kind="point", data=data, subpixel=True),
                "image": inputs["image"]}


@register_node
class TemplateMatch(VisionNode):
    """模板匹配：金字塔粗匹配（0.25x → 0.5x）→ 原分辨率局部精匹配，
    支持旋转搜索（angle_range ± 范围、angle_step 步长，默认 ±10° 步 5°）。

    性能硬前置：务必用 roi 限定搜索区（留空=全图，大图多角度会慢）。
    输出 rotated_rect 几何 [cx, cy, w, h, angle_deg]（w/h 为模板原始
    宽高，角度为模板旋转角）与匹配得分（TM_CCOEFF_NORMED，0~1）。
    得分低于 threshold 视为未找到，节点报错（下游走错误隔离）。

    模板录制：record(image, rect, out_dir) 从图上截取矩形存为模板 PNG
    （落流程 JSON 同目录「流程模板/」），路径填入 template_path。
    """

    NAME = "template_match"
    CATEGORY = "定位"
    INPUTS = (Port("image", "image", label="图像"),)
    OUTPUTS = (Port("image", "image", label="图像"),
               Port("result", "result", label="结果"),
               Port("score", "number", label="匹配得分"))
    PARAMS = {
        "template_path": ParamSpec("template_path", "模板图片", "path", "",
                                   tooltip="模板 PNG 路径（录制或外部图片）"),
        "roi": ParamSpec("roi", "搜索区", "text", "",
                         tooltip="x1,y1,x2,y2 像素坐标；留空=全图（大图慢）"),
        "angle_range": ParamSpec("angle_range", "角度范围", "float", 10.0,
                                 range=(0.0, 180.0), tooltip="±搜索范围（度）"),
        "angle_step": ParamSpec("angle_step", "角度步长", "float", 5.0,
                                range=(0.0, 45.0), tooltip="0 = 只匹配 0°"),
        "max_matches": ParamSpec("max_matches", "匹配数上限", "int", 1,
                                 range=(1, 50)),
        "threshold": ParamSpec("threshold", "得分下限", "float", 0.8,
                               range=(0.0, 1.0)),
    }

    def record(self, image: np.ndarray, rect, out_dir) -> str:
        """模板录制：截取 image 的 rect=(x1,y1,x2,y2) 存为模板 PNG。"""
        from ...io_utils import imwrite_unicode

        x1, y1, x2, y2 = (int(v) for v in rect)
        if x2 - x1 < 2 or y2 - y1 < 2:
            raise ValueError("模板录制区域太小（宽高至少 2 像素）")
        crop = np.asarray(image)[max(0, y1):y2, max(0, x1):x2]
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"tmpl_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        imwrite_unicode(path, crop)
        return str(path)

    # ---- 金字塔匹配主体 ----
    def run(self, inputs, params, ctx):
        from ...io_utils import imread_unicode

        path = str(params.get("template_path") or "")
        if not path:
            raise ValueError("未设置模板图片路径（可先在算子工作台录制模板）")
        tmpl_bgr = imread_unicode(path)
        if tmpl_bgr is None:
            raise ValueError(f"无法读取模板图片：{path}")
        tmpl = to_gray(tmpl_bgr)
        scene = to_gray(inputs["image"].mat)
        h, w = scene.shape[:2]
        x1, y1, x2, y2 = _parse_roi(params.get("roi"), w, h)
        region = np.ascontiguousarray(scene[y1:y2, x1:x2])
        angles = self._angle_list(float(params["angle_range"]),
                                  float(params["angle_step"]))
        max_m = int(params["max_matches"])
        th = float(params["threshold"])
        found = self._match_pyramid(region, tmpl, angles, max_m)
        best = max(found, key=lambda f: f["score"])
        if best["score"] < th:
            raise ValueError(
                f"模板匹配未达阈值：最高得分 {best['score']:.2f} < {th}"
                + ("" if not params.get("roi") else "（检查模板/搜索区/角度范围）"))
        found.sort(key=lambda f: -f["score"])
        f0 = found[0]
        rect = Geometry(kind="rotated_rect",
                        data=np.array([f0["cx"] + x1, f0["cy"] + y1,
                                       float(f0["tw"]), float(f0["th"]),
                                       float(f0["angle"])]),
                        subpixel=True)
        return {"result": rect, "score": NumberVal(value=round(float(f0["score"]), 4)),
                "image": inputs["image"]}

    @staticmethod
    def _angle_list(angle_range: float, angle_step: float) -> list:
        if angle_range <= 0 or angle_step <= 0:
            return [0.0]
        angles = {0.0}
        a = angle_step
        while a <= angle_range + 1e-9:
            angles.update((-a, a))
            a += angle_step
        return sorted(angles)

    def _match_pyramid(self, region: np.ndarray, tmpl: np.ndarray,
                       angles: list, max_m: int) -> list:
        """0.25x 全区粗匹配（取候选峰）→ 0.5x / 1.0x 局部精匹配。

        层级纪律：每个层级内场景与模板**同尺度**配对（都是同一插值来源），
        禁止跨层级直接相关（全分辨率窗口 × 降采样模板得分失真）。
        """
        rh, rw = region.shape
        rots = {a: self._rotate_template(tmpl, a) for a in angles}
        # 粗匹配层级：模板短边 ≥8px 的最小尺度
        scale = 0.25
        if min(tmpl.shape) * scale < 8:
            scale = 0.5
        levels = {1.0: region}
        for s in (0.5, scale):
            if s < 1.0 and s not in levels:
                levels[s] = cv2.resize(
                    region, (max(1, int(rw * s)), max(1, int(rh * s))),
                    interpolation=cv2.INTER_AREA)

        def tmpl_at(rot_full, s):
            if s == 1.0:
                return rot_full
            return cv2.resize(rot_full, (max(1, int(rot_full.shape[1] * s)),
                                         max(1, int(rot_full.shape[0] * s))),
                              interpolation=cv2.INTER_AREA)

        # 粗匹配：每个角度取前几个候选峰（NMS 分离）
        small = levels[scale]
        cands = []
        per_angle = 2 if max_m == 1 else max(3, max_m)
        for a in angles:
            rot_full, offx, offy, tw, th_ = rots[a]
            t_small = tmpl_at(rot_full, scale)
            if t_small.shape[0] >= small.shape[0] or t_small.shape[1] >= small.shape[1]:
                continue
            res = cv2.matchTemplate(small, t_small, cv2.TM_CCOEFF_NORMED)
            peaks = self._top_peaks(res, per_angle, 0.5 * max(t_small.shape))
            for (px, py), _s in peaks:
                cands.append({"score": float(_s), "cx": px + t_small.shape[1] / 2.0,
                              "cy": py + t_small.shape[0] / 2.0, "angle": a,
                              "tw": tw, "th": th_})
        if not cands:
            raise ValueError("模板大于搜索区域，无法匹配")
        cands.sort(key=lambda c: -c["score"])
        cands = cands[:max(12, max_m * 4)]
        # 精匹配：0.5 → 1.0，得分以原分辨率精匹配为准
        out = []
        for c in cands:
            rot_full, offx, offy, _tw, _th = rots[c["angle"]]
            cx, cy = c["cx"], c["cy"]
            score = c["score"]
            if 0.5 in levels:
                cx, cy, score = self._refine(levels, rot_full, cx, cy,
                                             scale, 0.5, 8)
            cx, cy, score = self._refine(levels, rot_full, cx, cy, 0.5, 1.0, 6)
            out.append({"cx": cx - offx + rot_full.shape[1] / 2.0,
                        "cy": cy - offy + rot_full.shape[0] / 2.0,
                        "score": score, "angle": c["angle"],
                        "tw": c["tw"], "th": c["th"]})
        out.sort(key=lambda f: -f["score"])
        picked: list = []
        sep = 0.7 * max(tmpl.shape)
        for f in out:   # 原分辨率去重：多目标保留分离的最佳者
            if all(math.hypot(f["cx"] - p["cx"], f["cy"] - p["cy"]) > sep
                   for p in picked):
                picked.append(f)
            if len(picked) >= max(1, max_m):
                break
        return picked

    @staticmethod
    def _top_peaks(res: np.ndarray, k: int, sep: float) -> list:
        """得分图前 k 个 NMS 峰（粗候选生成）。"""
        work = res.copy()
        peaks = []
        for _ in range(max(1, k)):
            _, smax, _, loc = cv2.minMaxLoc(work)
            if smax <= -1.0:
                break
            peaks.append((loc, float(smax)))
            x0 = max(0, loc[0] - int(sep))
            x1 = min(work.shape[1], loc[0] + int(sep) + 1)
            y0 = max(0, loc[1] - int(sep))
            y1 = min(work.shape[0], loc[1] + int(sep) + 1)
            work[y0:y1, x0:x1] = -1.0
        return peaks

    @staticmethod
    def _rotate_template(tmpl: np.ndarray, angle_deg: float):
        """模板绕中心旋转到扩张画布；返回 (图, 原模板中心在扩张画布的 x, y, w, h)。"""
        th, tw = tmpl.shape
        if abs(angle_deg) < 1e-9:
            return tmpl, tw / 2.0, th / 2.0, tw, th
        diag = math.ceil(math.hypot(th, tw))
        m = cv2.getRotationMatrix2D((tw / 2.0, th / 2.0), angle_deg, 1.0)
        m[0, 2] += diag / 2.0 - tw / 2.0
        m[1, 2] += diag / 2.0 - th / 2.0
        out = cv2.warpAffine(tmpl, m, (diag, diag))
        return out, diag / 2.0, diag / 2.0, tw, th

    @staticmethod
    def _refine(levels: dict, rot_full: np.ndarray,
                cx: float, cy: float, scale_in: float, scale_out: float,
                radius: int):
        """从 scale_in 层坐标精化到 scale_out 层：同尺度局部窗口内匹配。"""
        th, tw = rot_full.shape
        if scale_out == 1.0:
            t = rot_full
        else:
            t = cv2.resize(rot_full, (max(1, int(tw * scale_out)),
                                      max(1, int(th * scale_out))),
                           interpolation=cv2.INTER_AREA)
        thh, tww = t.shape
        region = levels[scale_out]
        ax = cx / scale_in * scale_out - tww / 2.0
        ay = cy / scale_in * scale_out - thh / 2.0
        wx = max(0, min(region.shape[1] - tww, int(round(ax)) - radius))
        wy = max(0, min(region.shape[0] - thh, int(round(ay)) - radius))
        sub = region[wy:wy + thh + 2 * radius, wx:wx + tww + 2 * radius]
        if sub.shape[0] <= thh or sub.shape[1] <= tww:
            return wx + tww / 2.0, wy + thh / 2.0, 0.0
        res = cv2.matchTemplate(sub, t, cv2.TM_CCOEFF_NORMED)
        _, smax, _, loc = cv2.minMaxLoc(res)
        return wx + loc[0] + tww / 2.0, wy + loc[1] + thh / 2.0, float(smax)


# ---------------------------------------------------------------- V0.6 T4-2


@register_node
class FitRectangle(VisionNode):
    """拟合矩形：二值化后取面积最大的外轮廓 → 最小外接旋转矩形。

    输出 rotated_rect 几何 [cx, cy, 长边, 短边, 角度°]（长边方向角归一到
    (-90, 90]）与角度数值；适合矩形工件的位置/尺寸/摆正角检测。
    threshold_mode=none 表示输入已是二值图（>0 为目标）。
    """

    NAME = "fit_rectangle"
    TITLE = "拟合矩形"
    CATEGORY = "定位"
    INPUTS = (Port("image", "image", label="图像"),)
    OUTPUTS = (Port("image", "image", label="图像"),
               Port("result", "result", label="结果"),
               Port("angle", "number", label="角度"))
    PARAMS = {
        "threshold_mode": ParamSpec("threshold_mode", "二值化", "enum", "otsu",
                                    options=["otsu", "fixed", "none"]),
        "thresh": ParamSpec("thresh", "固定阈值", "float", 128.0, range=(0.0, 255.0)),
        "invert": ParamSpec("invert", "暗目标", "bool", False),
        "min_area": ParamSpec("min_area", "最小面积", "int", 50, range=(0, 10 ** 8)),
        "roi": ParamSpec("roi", "搜索区", "text", "", tooltip="x1,y1,x2,y2 像素坐标；留空=全图"),
    }

    def run(self, inputs, params, ctx):
        gray = to_gray(inputs["image"].mat)
        h, w = gray.shape[:2]
        x1, y1, x2, y2 = _parse_roi(params.get("roi"), w, h)
        sub = gray[y1:y2, x1:x2]
        mode = params["threshold_mode"]
        if mode == "none":
            binary = (sub > 0).astype(np.uint8) * 255
            if params["invert"]:
                binary = 255 - binary
        else:
            flag = cv2.THRESH_BINARY_INV if params["invert"] else cv2.THRESH_BINARY
            if mode == "otsu":
                flag |= cv2.THRESH_OTSU
            _t, binary = cv2.threshold(sub, float(params["thresh"]), 255, flag)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= int(params["min_area"])]
        if not contours:
            raise ValueError("拟合矩形：未找到满足最小面积的目标轮廓（检查二值化/暗目标/最小面积）")
        c = max(contours, key=cv2.contourArea)
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
        if rw < rh:
            rw, rh, ang = rh, rw, ang + 90.0
        while ang <= -90.0:
            ang += 180.0
        while ang > 90.0:
            ang -= 180.0
        data = np.array([cx + x1, cy + y1, rw, rh, ang], dtype=np.float64)
        return {"result": Geometry(kind="rotated_rect", data=data, subpixel=True),
                "angle": NumberVal(value=round(float(ang), 3), unit="deg"),
                "image": inputs["image"]}


@register_node
class FindLinePair(VisionNode):
    """线对定位：两个测量区各用卡尺找一条边/线，输出两线、间距与夹角。

    典型用于宽度测量（板宽、缝宽）：测量区 1/2 分别框住两条边；两条边
    明暗方向常相反（如亮条带的上边暗→亮、下边亮→暗），故极性分别设置。
    卡尺找边算法与「直线定位」完全相同（复用同一实现）。
    """

    NAME = "find_line_pair"
    TITLE = "线对定位"
    CATEGORY = "定位"
    INPUTS = (Port("image", "image", label="图像"),)
    OUTPUTS = (Port("image", "image", label="图像"),
               Port("result", "result", label="结果"),
               Port("distance", "number", label="间距"))
    PARAMS = {
        "target": ParamSpec("target", "目标类型", "enum", "edge", options=["edge", "line"]),
        "scan": ParamSpec("scan", "卡尺方向", "enum", "vertical",
                          options=["vertical", "horizontal"],
                          tooltip="vertical=找水平线；horizontal=找竖直线"),
        "roi1": ParamSpec("roi1", "测量区1", "text", "", tooltip="x1,y1,x2,y2 像素坐标（必填）"),
        "roi2": ParamSpec("roi2", "测量区2", "text", "", tooltip="x1,y1,x2,y2 像素坐标（必填）"),
        "polarity1": ParamSpec("polarity1", "极性1", "enum", "any",
                               options=["any", "bright", "dark"]),
        "polarity2": ParamSpec("polarity2", "极性2", "enum", "any",
                               options=["any", "bright", "dark"]),
        "num_calipers": ParamSpec("num_calipers", "卡尺数量", "int", 32, range=(4, 256)),
        "min_contrast": ParamSpec("min_contrast", "最小对比度", "float", 20.0,
                                  range=(1.0, 255.0)),
    }

    def validate(self, params):
        errs = super().validate(params)
        for k in ("roi1", "roi2"):
            if not str(params.get(k) or "").strip():
                errs.append(f"线对定位：{self.PARAMS[k].label}未设置")
        return errs

    def run(self, inputs, params, ctx):
        from ..ir import MeasureResult
        from .measure import angle_between, line_line_distance

        finder = FindLine()
        common = {"target": params["target"], "scan": params["scan"],
                  "num_calipers": params["num_calipers"],
                  "min_contrast": params["min_contrast"]}
        lines = []
        for i in (1, 2):
            try:
                out = finder.run({"image": inputs["image"]},
                                 dict(common, roi=params[f"roi{i}"],
                                      polarity=params[f"polarity{i}"]), ctx)
            except ValueError as e:
                raise ValueError(f"线对定位：测量区{i}——{e}") from None
            lines.append(out["result"])
        l1 = np.asarray(lines[0].data, dtype=np.float64).reshape(2, 2)
        l2 = np.asarray(lines[1].data, dtype=np.float64).reshape(2, 2)
        dist = line_line_distance(l1, l2)
        ang = angle_between(l1, l2)
        if ang > 5.0:
            ctx.warn(f"线对定位：两线夹角 {ang:.1f}° 偏大，间距按中点垂距平均计算")
        return {"result": MeasureResult(value=round(dist, 4), unit="px"),
                "distance": NumberVal(value=round(dist, 4), unit="px"),
                "image": inputs["image"]}
