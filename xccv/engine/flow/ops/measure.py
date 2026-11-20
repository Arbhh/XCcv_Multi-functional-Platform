"""测量类算子（端口统一）：距离 / 角度（带容差判定）。

端口约定：
输入 = 图像（原图直通）+ 结果（待测几何：点/线/圆/旋转矩形/多边形）+
可选第二结果（双输入口径）；输出 = 图像 + 结果（MeasureResult，带容差判定）+
数值（长度/角度，供判定与逻辑节点使用）。

- measure_distance「距离测量」：单结果：直线=长度、圆=直径、旋转矩形=长边、
  点集=前两点距离；双结果：点-点=欧氏距离、点-线=垂距、线-线=两线间距
  （对称平均垂距，适合近平行线宽度测量）；圆/旋转矩形参与双输入时取其中心点。
- measure_angle「角度测量」：单线=相对水平方向角（-90°~90°］；
  旋转矩形=其角度；两线=夹角（acute 0~90° / signed -180°~180°）。

比例尺 scale(mm/px)>0 时距离换算为毫米。容差下限/上限均为 0 = 不判定
（passed=None）；与 detection_to_measure 同口径。
"""

import math

import numpy as np

from ..ir import Geometry, MeasureResult, NumberVal
from ..node_base import ParamSpec, Port, VisionNode
from ..registry import register_node

_TOL_PARAMS = {
    "tol_min": ParamSpec("tol_min", "容差下限", "float", 0.0, range=(-1e9, 1e9),
                         tooltip="下限与上限都为 0 = 不判定"),
    "tol_max": ParamSpec("tol_max", "容差上限", "float", 0.0, range=(-1e9, 1e9)),
}


def _judge(value: float, params) -> tuple:
    lo, hi = float(params["tol_min"]), float(params["tol_max"])
    if lo == 0.0 and hi == 0.0:
        return None, None
    passed = (value >= lo if lo else True) and (value <= hi if hi else True)
    return (lo, hi), bool(passed)


def _as_point(geo: Geometry) -> np.ndarray:
    d = np.asarray(geo.data, dtype=np.float64)
    if geo.kind == "point":
        return d.reshape(-1, 2)[0]
    if geo.kind in ("circle", "rotated_rect"):
        return d.reshape(-1)[:2]
    if geo.kind == "line":
        return d.reshape(2, 2).mean(axis=0)
    return d.reshape(-1, 2).mean(axis=0)   # polygon：质心近似


def _as_line(geo: Geometry) -> np.ndarray:
    return np.asarray(geo.data, dtype=np.float64).reshape(2, 2)


def point_line_distance(p, line) -> float:
    a, b = line
    ab = b - a
    n = float(np.hypot(*ab))
    if n < 1e-12:
        return float(np.hypot(*(p - a)))
    return abs(float(ab[0] * (p[1] - a[1]) - ab[1] * (p[0] - a[0]))) / n


def line_line_distance(l1, l2) -> float:
    """两线间距：各自中点到另一条线的垂距取平均（近平行线即"宽度"）。"""
    return 0.5 * (point_line_distance(l2.mean(axis=0), l1)
                  + point_line_distance(l1.mean(axis=0), l2))


def line_angle_deg(line) -> float:
    """直线相对水平方向的角度，归一化到 (-90, 90]（图像坐标 y 向下）。"""
    (x1, y1), (x2, y2) = line
    ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
    while ang <= -90.0:
        ang += 180.0
    while ang > 90.0:
        ang -= 180.0
    return ang


def angle_between(l1, l2, mode: str = "acute") -> float:
    d1, d2 = l1[1] - l1[0], l2[1] - l2[0]
    a = math.degrees(math.atan2(d2[1], d2[0]) - math.atan2(d1[1], d1[0]))
    while a <= -180.0:
        a += 360.0
    while a > 180.0:
        a -= 360.0
    if mode == "signed":
        return a
    a = abs(a)
    return 180.0 - a if a > 90.0 else a


def geometry_distance(a: Geometry, b: Geometry | None) -> float:
    """按几何种类计算距离（像素），口径见模块说明。"""
    if b is None:
        if a.kind == "line":
            l = _as_line(a)
            return float(np.hypot(*(l[1] - l[0])))
        if a.kind == "circle":
            return 2.0 * abs(float(np.asarray(a.data).reshape(-1)[2]))
        if a.kind == "rotated_rect":
            d = np.asarray(a.data, dtype=np.float64).reshape(-1)
            return float(max(abs(d[2]), abs(d[3])))
        pts = np.asarray(a.data, dtype=np.float64).reshape(-1, 2)
        if len(pts) < 2:
            raise ValueError("距离测量：单个点无法测距，请连接第二个几何输入")
        return float(np.hypot(*(pts[1] - pts[0])))
    if a.kind == "line" and b.kind == "line":
        return line_line_distance(_as_line(a), _as_line(b))
    if a.kind == "line":
        return point_line_distance(_as_point(b), _as_line(a))
    if b.kind == "line":
        return point_line_distance(_as_point(a), _as_line(b))
    return float(np.hypot(*(_as_point(a) - _as_point(b))))


def _result_geometry(v) -> Geometry | None:
    """统一结果端口 → 几何（非几何结果返回 None，由调用方报人话错误）。"""
    return v if isinstance(v, Geometry) else None


@register_node
class MeasureDistance(VisionNode):
    """距离测量：点-点/点-线/线-线距离、直线长度、圆直径（自动按几何种类）。

    结果输入须携带几何（定位类节点的 result）；比例尺>0 时输出毫米；
    容差判定写入测量结果（供结果聚合/保存结果读取）。
    """

    NAME = "measure_distance"
    TITLE = "距离测量"
    CATEGORY = "测量"
    INPUTS = (Port("image", "image", required=False, label="图像"),
              Port("result", "result", label="结果"),
              Port("result2", "result", required=False, label="结果2"))
    OUTPUTS = (Port("image", "image", label="图像"),
               Port("result", "result", label="结果"),
               Port("distance", "number", label="距离"))
    PARAMS = {
        "scale": ParamSpec("scale", "比例尺 mm/px", "float", 0.0, range=(0.0, 1e6),
                           tooltip="0 = 输出像素；>0 = 像素×比例尺得毫米"),
        **_TOL_PARAMS,
    }

    def run(self, inputs, params, ctx):
        a = _result_geometry(inputs.get("result"))
        if a is None:
            raise ValueError("距离测量：结果输入须为几何（请连接定位类节点的结果输出）")
        b = _result_geometry(inputs.get("result2"))
        d = geometry_distance(a, b)
        scale = float(params["scale"])
        unit = "mm" if scale > 0 else "px"
        value = round(d * scale if scale > 0 else d, 4)
        tol, passed = _judge(value, params)
        img = inputs.get("image")
        out = {"result": MeasureResult(value=value, unit=unit, tolerance=tol, passed=passed),
               "distance": NumberVal(value=value, unit=unit)}
        if img is not None:
            out["image"] = img
        return out


@register_node
class MeasureAngle(VisionNode):
    """角度测量：单线=相对水平角、旋转矩形=其角度、两线=夹角（度）。"""

    NAME = "measure_angle"
    TITLE = "角度测量"
    CATEGORY = "测量"
    INPUTS = (Port("image", "image", required=False, label="图像"),
              Port("result", "result", label="结果"),
              Port("result2", "result", required=False, label="结果2"))
    OUTPUTS = (Port("image", "image", label="图像"),
               Port("result", "result", label="结果"),
               Port("angle", "number", label="角度"))
    PARAMS = {
        "mode": ParamSpec("mode", "夹角口径", "enum", "acute", options=["acute", "signed"],
                          tooltip="acute=0~90°；signed=从 a 转到 b 的有向角 -180~180°"),
        **_TOL_PARAMS,
    }

    def run(self, inputs, params, ctx):
        a = _result_geometry(inputs.get("result"))
        if a is None:
            raise ValueError("角度测量：结果输入须为几何（请连接定位类节点的结果输出）")
        b = _result_geometry(inputs.get("result2"))
        if b is None:
            if a.kind == "line":
                ang = line_angle_deg(_as_line(a))
            elif a.kind == "rotated_rect":
                ang = float(np.asarray(a.data, dtype=np.float64).reshape(-1)[4])
            else:
                raise ValueError("角度测量：单输入需为直线或旋转矩形")
        else:
            if a.kind != "line" or b.kind != "line":
                raise ValueError("角度测量：两输入都需为直线")
            ang = angle_between(_as_line(a), _as_line(b), params["mode"])
        value = round(ang, 4)
        tol, passed = _judge(value, params)
        out = {"result": MeasureResult(value=value, unit="deg", tolerance=tol, passed=passed),
               "angle": NumberVal(value=value, unit="deg")}
        img = inputs.get("image")
        if img is not None:
            out["image"] = img
        return out
