"""预处理类算子（11 个，）。

彩色/灰度行为约定（每个算子 docstring 均已写明）：
- gaussian_blur / median_blur / bilateral_filter / morphology：灰度或彩色
  均可（彩色逐通道处理，cv2 原生支持）；
- clahe / threshold / canny_edge / sobel_edge / fft_filter / roi_extract
  的 annulus 模式：彩色输入自动转灰度；
- roi_extract 的 rect 模式：直接裁剪（彩色保持彩色）。
"""

import cv2
import numpy as np

from ..ir import ImageData
from ..node_base import ParamSpec, Port, VisionNode
from ..registry import register_node
from . import ensure_odd, to_gray

_GRAY = "彩色输入自动转灰度；灰度原样处理。"


def _as_image(mat, meta=None) -> ImageData:
    return ImageData(mat=mat, meta=dict(meta or {}))


@register_node
class GaussianBlur(VisionNode):
    """高斯模糊：抑制高频噪声（核数越大越平滑）。

    灰度/彩色均可，彩色逐通道处理；核数自动补奇。
    """

    NAME = "gaussian_blur"
    CATEGORY = "预处理"
    INPUTS = (Port("image", "image", label="图像"),)
    OUTPUTS = (Port("image", "image", label="图像"),)
    PARAMS = {
        "ksize": ParamSpec("ksize", "核数", "int", 3, range=(1, 31),
                           tooltip="高斯核边长（奇数，越大越模糊）"),
        "sigma": ParamSpec("sigma", "Sigma", "float", 0.0, range=(0.0, 10.0),
                           tooltip="标准差；0 = 按核数自动计算"),
    }

    def run(self, inputs, params, ctx):
        img = inputs["image"]
        k = ensure_odd(params["ksize"], ctx, "核数")
        out = cv2.GaussianBlur(img.mat, (k, k), params["sigma"])
        return {"image": _as_image(out, img.meta)}


@register_node
class MedianBlur(VisionNode):
    """中值模糊：去椒盐噪声、保边优于高斯。灰度/彩色均可。"""

    NAME = "median_blur"
    CATEGORY = "预处理"
    INPUTS = (Port("image", "image", label="图像"),)
    OUTPUTS = (Port("image", "image", label="图像"),)
    PARAMS = {
        "ksize": ParamSpec("ksize", "核数", "int", 3, range=(3, 31),
                           tooltip="中值窗口边长（奇数）"),
    }

    def run(self, inputs, params, ctx):
        img = inputs["image"]
        k = ensure_odd(params["ksize"], ctx, "核数")
        return {"image": _as_image(cv2.medianBlur(img.mat, k), img.meta)}


@register_node
class BilateralFilter(VisionNode):
    """双边滤波：磨皮去噪同时保留边缘。灰度/彩色均可（彩色逐通道）。"""

    NAME = "bilateral_filter"
    CATEGORY = "预处理"
    INPUTS = (Port("image", "image", label="图像"),)
    OUTPUTS = (Port("image", "image", label="图像"),)
    PARAMS = {
        "d": ParamSpec("d", "邻域直径", "int", 9, range=(1, 15)),
        "sigma_color": ParamSpec("sigma_color", "颜色σ", "float", 75.0,
                                 range=(1.0, 200.0), tooltip="越大越混色"),
        "sigma_space": ParamSpec("sigma_space", "空间σ", "float", 75.0,
                                 range=(1.0, 200.0), tooltip="越大越平滑"),
    }

    def run(self, inputs, params, ctx):
        img = inputs["image"]
        out = cv2.bilateralFilter(img.mat, int(params["d"]),
                                  float(params["sigma_color"]),
                                  float(params["sigma_space"]))
        return {"image": _as_image(out, img.meta)}


@register_node
class Clahe(VisionNode):
    """CLAHE 自适应直方图均衡：增强低对比度缺陷。

    彩色输入自动转灰度；灰度原样处理。
    """

    NAME = "clahe"
    CATEGORY = "预处理"
    INPUTS = (Port("image", "image", label="图像"),)
    OUTPUTS = (Port("image", "image", label="图像"),)
    PARAMS = {
        "clip_limit": ParamSpec("clip_limit", "对比度上限", "float", 2.0,
                                range=(0.1, 10.0)),
        "tile_size": ParamSpec("tile_size", "分块", "int", 8, range=(1, 32),
                               tooltip="tileGridSize 边长"),
    }

    def run(self, inputs, params, ctx):
        img = inputs["image"]
        gray = to_gray(img.mat)
        clahe = cv2.createCLAHE(clipLimit=float(params["clip_limit"]),
                                tileGridSize=(int(params["tile_size"]),) * 2)
        return {"image": _as_image(clahe.apply(gray), img.meta)}


@register_node
class Threshold(VisionNode):
    """二值化：global 固定阈值 / otsu 自动阈值（忽略 thresh）/
    adaptive 局部自适应。

    彩色输入自动转灰度；灰度原样处理。
    """

    NAME = "threshold"
    CATEGORY = "预处理"
    INPUTS = (Port("image", "image", label="图像"),)
    OUTPUTS = (Port("image", "image", label="图像"),)
    PARAMS = {
        "mode": ParamSpec("mode", "模式", "enum", "global",
                          options=["global", "otsu", "adaptive"]),
        "thresh": ParamSpec("thresh", "阈值", "float", 127.0, range=(0.0, 255.0),
                            tooltip="仅 global 模式生效"),
        "adaptive_method": ParamSpec("adaptive_method", "自适应方法", "enum",
                                     "mean", options=["mean", "gaussian"],
                                     tooltip="仅 adaptive 模式生效"),
        "block_size": ParamSpec("block_size", "邻域", "int", 11, range=(3, 51),
                                tooltip="adaptive 邻域边长（自动补奇）"),
        "invert": ParamSpec("invert", "反相", "bool", False,
                            tooltip="勾选后亮底暗目标 → 暗底亮目标"),
    }

    def run(self, inputs, params, ctx):
        img = inputs["image"]
        gray = to_gray(img.mat)
        mode = params["mode"]
        inv = cv2.THRESH_BINARY_INV if params["invert"] else cv2.THRESH_BINARY
        if mode == "otsu":
            _, out = cv2.threshold(gray, 0, 255, inv | cv2.THRESH_OTSU)
        elif mode == "adaptive":
            method = (cv2.ADAPTIVE_THRESH_MEAN_C if params["adaptive_method"] == "mean"
                      else cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
            k = ensure_odd(params["block_size"], ctx, "邻域")
            out = cv2.adaptiveThreshold(gray, 255, method,
                                        cv2.THRESH_BINARY, k, -2)
            if params["invert"]:
                out = 255 - out
        else:
            _, out = cv2.threshold(gray, float(params["thresh"]), 255, inv)
        return {"image": _as_image(out, img.meta)}


@register_node
class Morphology(VisionNode):
    """形态学：腐蚀/膨胀/开/闭/梯度/顶帽/黑帽。灰度/彩色均可。"""

    NAME = "morphology"
    CATEGORY = "预处理"
    INPUTS = (Port("image", "image", label="图像"),)
    OUTPUTS = (Port("image", "image", label="图像"),)
    PARAMS = {
        "op": ParamSpec("op", "运算", "enum", "open",
                        options=["erode", "dilate", "open", "close",
                                 "gradient", "tophat", "blackhat"]),
        "kernel_w": ParamSpec("kernel_w", "核宽", "int", 3, range=(1, 31)),
        "kernel_h": ParamSpec("kernel_h", "核高", "int", 3, range=(1, 31)),
        "iterations": ParamSpec("iterations", "迭代次数", "int", 1, range=(1, 10)),
    }

    _OPS = {"erode": cv2.MORPH_ERODE, "dilate": cv2.MORPH_DILATE,
            "open": cv2.MORPH_OPEN, "close": cv2.MORPH_CLOSE,
            "gradient": cv2.MORPH_GRADIENT, "tophat": cv2.MORPH_TOPHAT,
            "blackhat": cv2.MORPH_BLACKHAT}

    def run(self, inputs, params, ctx):
        img = inputs["image"]
        kw = ensure_odd(params["kernel_w"], ctx, "核宽")
        kh = ensure_odd(params["kernel_h"], ctx, "核高")
        kernel = np.ones((kw, kh), np.uint8)
        out = cv2.morphologyEx(img.mat, self._OPS[params["op"]], kernel,
                               iterations=int(params["iterations"]))
        return {"image": _as_image(out, img.meta)}


@register_node
class CannyEdge(VisionNode):
    """Canny 边缘检测。

    彩色输入自动转灰度；灰度原样处理。
    """

    NAME = "canny_edge"
    CATEGORY = "预处理"
    INPUTS = (Port("image", "image", label="图像"),)
    OUTPUTS = (Port("image", "image", label="图像"),)
    PARAMS = {
        "low": ParamSpec("low", "低阈值", "float", 50.0, range=(0.0, 255.0)),
        "high": ParamSpec("high", "高阈值", "float", 150.0, range=(0.0, 255.0)),
        "aperture": ParamSpec("aperture", "Sobel 核", "enum", 3,
                              options=[3, 5, 7]),
    }

    def run(self, inputs, params, ctx):
        img = inputs["image"]
        low, high = float(params["low"]), float(params["high"])
        if low > high:  # 边界约定：自动交换并警告
            low, high = high, low
            ctx.warn("Canny 低阈值大于高阈值，已自动交换")
        out = cv2.Canny(to_gray(img.mat), low, high,
                        apertureSize=int(params["aperture"]))
        return {"image": _as_image(out, img.meta)}


@register_node
class SobelEdge(VisionNode):
    """Sobel 梯度：dx/dy 至少一个为 1；同时为 1 时取两方向幅值。

    彩色输入自动转灰度；灰度原样处理。
    """

    NAME = "sobel_edge"
    CATEGORY = "预处理"
    INPUTS = (Port("image", "image", label="图像"),)
    OUTPUTS = (Port("image", "image", label="图像"),)
    PARAMS = {
        "dx": ParamSpec("dx", "X 方向", "enum", 1, options=[0, 1]),
        "dy": ParamSpec("dy", "Y 方向", "enum", 0, options=[0, 1]),
        "ksize": ParamSpec("ksize", "核数", "enum", 3, options=[1, 3, 5, 7],
                           tooltip="1 = Scharr（小核高精度）"),
    }

    def run(self, inputs, params, ctx):
        img = inputs["image"]
        gray = to_gray(img.mat)
        dx, dy, k = int(params["dx"]), int(params["dy"]), int(params["ksize"])
        if dx == 0 and dy == 0:
            dx = 1
            ctx.warn("dx/dy 不能同时为 0，已按 X 方向计算")
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=k)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=k)
        if dx and dy:
            mag = cv2.magnitude(gx, gy)
            out = cv2.convertScaleAbs(mag)
        else:
            out = cv2.convertScaleAbs(gx if dx else gy)
        return {"image": _as_image(out, img.meta)}


@register_node
class FftFilter(VisionNode):
    """FFT 频域滤波：去除周期性纹理/摩尔纹（低通/高通/带阻）。

    彩色输入自动转灰度；灰度原样处理。
    """

    NAME = "fft_filter"
    CATEGORY = "预处理"
    INPUTS = (Port("image", "image", label="图像"),)
    OUTPUTS = (Port("image", "image", label="图像"),)
    PARAMS = {
        "mode": ParamSpec("mode", "模式", "enum", "lowpass",
                          options=["lowpass", "highpass", "bandstop"]),
        "cutoff_ratio": ParamSpec("cutoff_ratio", "截止比", "float", 0.1,
                                  range=(0.01, 0.5),
                                  tooltip="截止半径 = 比例 × min(宽,高)/2"),
        "bandwidth": ParamSpec("bandwidth", "阻带宽度比", "float", 0.05,
                               range=(0.01, 0.3), tooltip="仅 bandstop 生效"),
    }

    def run(self, inputs, params, ctx):
        img = inputs["image"]
        gray = to_gray(img.mat).astype(np.float32)
        h, w = gray.shape
        crow, ccol = h // 2, w // 2
        base = min(h, w) / 2.0
        r = float(params["cutoff_ratio"]) * base
        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt((yy - crow) ** 2 + (xx - ccol) ** 2)
        mode = params["mode"]
        if mode == "lowpass":
            mask = (dist <= r).astype(np.float32)
        elif mode == "highpass":
            mask = (dist > r).astype(np.float32)
        else:  # bandstop：阻断 r ~ r+带宽 的环形频段
            r2 = r + float(params["bandwidth"]) * base
            mask = ((dist < r) | (dist > r2)).astype(np.float32)
        f = np.fft.fftshift(np.fft.fft2(gray))
        back = np.fft.ifft2(np.fft.ifftshift(f * mask))
        out = np.clip(np.abs(back), 0, 255).astype(np.uint8)
        return {"image": _as_image(out, img.meta)}


@register_node
class AffineTransform(VisionNode):
    """仿射变换：6 元矩阵 "a,b,tx,c,d,ty"（行优先），边界复制防黑边。

    矩阵手工填写属高级用法；常规"定位归正"接「几何转归正矩阵」的结果输出
    （matrix_in 端口，值=6 元矩阵），端口优先于手填文本。
    """

    NAME = "affine_transform"
    TITLE = "仿射变换"
    CATEGORY = "预处理"
    INPUTS = (Port("image", "image", label="图像"),
              Port("matrix_in", "result", required=False, label="归正矩阵"))
    OUTPUTS = (Port("image", "image", label="图像"),)
    PARAMS = {
        "matrix": ParamSpec("matrix", "矩阵(6元)", "text", "1,0,0,0,1,0",
                            tooltip="行优先 a,b,tx,c,d,ty"),
        "interp": ParamSpec("interp", "插值", "enum", "linear",
                            options=["nearest", "linear", "cubic"]),
    }

    _INTERP = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR,
               "cubic": cv2.INTER_CUBIC}

    def validate(self, params):
        errs = super().validate(params)
        try:
            nums = [float(v) for v in str(params.get("matrix", "")).split(",")
                    if v.strip()]
            if len(nums) != 6:
                errs.append("「矩阵(6元)」应为 6 个数字（a,b,tx,c,d,ty）")
        except (TypeError, ValueError):
            errs.append("「矩阵(6元)」应为 6 个数字（a,b,tx,c,d,ty）")
        return errs

    def run(self, inputs, params, ctx):
        img = inputs["image"]
        m_in = inputs.get("matrix_in")
        if m_in is not None and getattr(m_in, "values", None):
            # geometry_to_affine 等上游的矩阵端口优先于手填文本
            nums = [float(v) for v in m_in.values]
            if len(nums) != 6:
                raise ValueError("矩阵端口应含 6 个数值（a,b,tx,c,d,ty）")
        else:
            nums = [float(v) for v in str(params["matrix"]).split(",") if v.strip()]
        m = np.float32([[nums[0], nums[1], nums[2]],
                        [nums[3], nums[4], nums[5]]])
        h, w = img.mat.shape[:2]
        out = cv2.warpAffine(img.mat, m, (w, h),
                             flags=self._INTERP[params["interp"]],
                             borderMode=cv2.BORDER_REPLICATE)
        return {"image": _as_image(out, img.meta)}


@register_node
class RoiExtract(VisionNode):
    """ROI 提取：rect 裁剪 / annulus 环形掩膜 / polygon 几何掩膜。

    - rect：x/y/w/h 为归一化比例（0~1），输出裁剪子图；
    - annulus：以图中心为圆心的环形区域（内外半径比 0~1，基于
      max(宽,高)/2），环外置黑，整图尺寸不变；
    - polygon：「几何结果」输入端口（值=polygon 顶点，来自定位类节点的
      结果输出）构建掩膜，多边形外置黑，整图尺寸不变。
    """

    NAME = "roi_extract"
    TITLE = "ROI 提取"
    CATEGORY = "预处理"
    INPUTS = (Port("image", "image", label="图像"),
              Port("polygon", "result", required=False, label="几何结果"))
    OUTPUTS = (Port("image", "image", label="图像"),)
    PARAMS = {
        "mode": ParamSpec("mode", "模式", "enum", "rect",
                          options=["rect", "annulus", "polygon"]),
        "x": ParamSpec("x", "X 比例", "float", 0.0, range=(0.0, 1.0),
                       tooltip="仅 rect 模式"),
        "y": ParamSpec("y", "Y 比例", "float", 0.0, range=(0.0, 1.0)),
        "w": ParamSpec("w", "宽比例", "float", 1.0, range=(0.0, 1.0)),
        "h": ParamSpec("h", "高比例", "float", 1.0, range=(0.0, 1.0)),
        "rin": ParamSpec("rin", "内半径比", "float", 0.3, range=(0.0, 1.0),
                         tooltip="仅 annulus 模式（基于 max(宽,高)/2）"),
        "rout": ParamSpec("rout", "外半径比", "float", 0.9, range=(0.0, 1.0)),
    }

    def run(self, inputs, params, ctx):
        img = inputs["image"]
        mat = img.mat
        h, w = mat.shape[:2]
        mode = params["mode"]
        if mode == "rect":
            x1 = int(round(params["x"] * w))
            y1 = int(round(params["y"] * h))
            x2 = min(w, int(round((params["x"] + params["w"]) * w)))
            y2 = min(h, int(round((params["y"] + params["h"]) * h)))
            if x2 - x1 < 1 or y2 - y1 < 1:
                raise ValueError("rect 区域宽/高不足 1 像素，请检查比例参数")
            meta = dict(img.meta)
            meta["roi_rect"] = [x1, y1, x2, y2]
            return {"image": ImageData(mat=mat[y1:y2, x1:x2].copy(), meta=meta)}
        if mode == "annulus":
            rin = min(float(params["rin"]), float(params["rout"]))
            rout = max(float(params["rin"]), float(params["rout"]))
            base = max(w, h) / 2.0
            yy, xx = np.ogrid[:h, :w]
            dist = np.sqrt((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2)
            mask = ((dist >= rin * base) & (dist <= rout * base))
            out = np.where(mask[..., None] if mat.ndim == 3 else mask, mat, 0)
            out = out.astype(mat.dtype)
            meta = dict(img.meta)
            meta["roi"] = {"kind": "annulus", "rin": rin, "rout": rout}
            return {"image": ImageData(mat=out, meta=meta)}
        # polygon：几何结果输入端口（值=polygon 顶点）
        geo = inputs.get("polygon")
        if geo is None:
            raise ValueError("polygon 模式需要「几何结果」输入端口（定位类节点的结果）")
        pts = np.asarray(geo.data, dtype=np.int32).reshape(-1, 1, 2)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        keep = mask > 0
        out = np.where(keep[..., None] if mat.ndim == 3 else keep, mat, 0)
        out = out.astype(mat.dtype)
        meta = dict(img.meta)
        meta["roi_polygon_points"] = int(len(geo.data))
        return {"image": ImageData(mat=out, meta=meta)}


# ---------------------------------------------------------------- 预处理算子


def _lut_apply(mat, lut):
    """uint8 查表（彩色逐通道同一张表）。"""
    return cv2.LUT(mat, lut.astype(np.uint8))


@register_node
class SharpenUnsharp(VisionNode):
    """反锐化掩模锐化：原图 + 强度×(原图 − 高斯模糊)，增强边缘细节。

    灰度/彩色均可（逐通道）。阈值>0 时只锐化差值超过阈值的像素（抑制噪声放大）。
    """

    NAME = "sharpen_unsharp"
    TITLE = "反锐化锐化"
    CATEGORY = "预处理"
    INPUTS = (Port("image", "image", label="图像"),)
    OUTPUTS = (Port("image", "image", label="图像"),)
    PARAMS = {
        "amount": ParamSpec("amount", "强度", "float", 1.0, range=(0.0, 5.0),
                            tooltip="0=不锐化；1~2 常用"),
        "ksize": ParamSpec("ksize", "半径核数", "int", 5, range=(1, 51),
                           tooltip="高斯核边长（奇数，越大强调越粗的边缘）"),
        "threshold": ParamSpec("threshold", "阈值", "int", 0, range=(0, 255),
                               tooltip="差值不超过阈值的像素不锐化（抑噪）"),
    }

    def run(self, inputs, params, ctx):
        img = inputs["image"]
        k = ensure_odd(params["ksize"], ctx, "半径核数")
        src = img.mat.astype(np.float32)
        blur = cv2.GaussianBlur(src, (k, k), 0)
        diff = src - blur
        if int(params["threshold"]) > 0:
            diff = np.where(np.abs(diff) > float(params["threshold"]), diff, 0.0)
        out = np.clip(src + float(params["amount"]) * diff, 0, 255).astype(np.uint8)
        return {"image": _as_image(out, img.meta)}


@register_node
class GammaCorr(VisionNode):
    """Gamma 校正：out = 255×(in/255)^γ（γ<1 提亮暗部，γ>1 压暗亮部）。

    灰度/彩色均可（查表，毫秒级）。
    """

    NAME = "gamma_corr"
    TITLE = "Gamma 校正"
    CATEGORY = "预处理"
    INPUTS = (Port("image", "image", label="图像"),)
    OUTPUTS = (Port("image", "image", label="图像"),)
    PARAMS = {
        "gamma": ParamSpec("gamma", "Gamma", "float", 1.0, range=(0.1, 5.0),
                           tooltip="<1 提亮暗部；>1 压暗；1=不变"),
    }

    def run(self, inputs, params, ctx):
        img = inputs["image"]
        g = float(params["gamma"])
        lut = np.clip(255.0 * (np.arange(256) / 255.0) ** g + 0.5, 0, 255)
        return {"image": _as_image(_lut_apply(img.mat, lut), img.meta)}


@register_node
class Invert(VisionNode):
    """反色：out = 255 − in（暗目标转亮目标，便于后续亮目标算子处理）。"""

    NAME = "invert"
    TITLE = "反色"
    CATEGORY = "预处理"
    INPUTS = (Port("image", "image", label="图像"),)
    OUTPUTS = (Port("image", "image", label="图像"),)
    PARAMS = {}

    def run(self, inputs, params, ctx):
        img = inputs["image"]
        return {"image": _as_image(cv2.bitwise_not(img.mat), img.meta)}


@register_node
class BrightnessContrast(VisionNode):
    """亮度对比度：out = 对比度×in + 亮度（饱和截断到 0~255）。

    灰度/彩色均可。对比度 1、亮度 0 = 不变。
    """

    NAME = "brightness_contrast"
    TITLE = "亮度对比度"
    CATEGORY = "预处理"
    INPUTS = (Port("image", "image", label="图像"),)
    OUTPUTS = (Port("image", "image", label="图像"),)
    PARAMS = {
        "contrast": ParamSpec("contrast", "对比度", "float", 1.0, range=(0.0, 5.0),
                              tooltip="增益 α：>1 拉开明暗差"),
        "brightness": ParamSpec("brightness", "亮度", "float", 0.0, range=(-255.0, 255.0),
                                tooltip="偏置 β：正值提亮"),
    }

    def run(self, inputs, params, ctx):
        img = inputs["image"]
        lut = np.clip(np.arange(256) * float(params["contrast"])
                      + float(params["brightness"]) + 0.5, 0, 255)
        return {"image": _as_image(_lut_apply(img.mat, lut), img.meta)}


def _smooth_background(src_f32, k: int):
    """大核背景估计：先缩小再高斯再放大（核 >31 时把耗时从 O(k) 降到常数级）。"""
    h, w = src_f32.shape[:2]
    s = max(1, k // 31)
    if s > 1 and min(h, w) // s >= 8:
        small = cv2.resize(src_f32, (max(1, w // s), max(1, h // s)), interpolation=cv2.INTER_AREA)
        kk = max(3, (k // s) | 1)
        bg = cv2.GaussianBlur(small, (kk, kk), 0)
        return cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    k = max(3, k | 1)
    return cv2.GaussianBlur(src_f32, (k, k), 0)


@register_node
class FlatFieldCorr(VisionNode):
    """平场校正（背景均匀化）：消除光照不均/暗角，让同一阈值全图适用。

    - blur_divide：大核模糊估计背景，out = in / 背景 × 背景均值（推荐）；
    - blur_subtract：out = in − 背景 + 背景均值（适合加性阴影）；
    - reference：用参考平场图（同相机拍的空白/均匀板）做除法校正。
    灰度/彩色均可（逐通道）；核数取目标尺寸的 3~5 倍。
    """

    NAME = "flat_field_corr"
    TITLE = "平场校正"
    CATEGORY = "预处理"
    INPUTS = (Port("image", "image", label="图像"),)
    OUTPUTS = (Port("image", "image", label="图像"),)
    PARAMS = {
        "mode": ParamSpec("mode", "方式", "enum", "blur_divide",
                          options=["blur_divide", "blur_subtract", "reference"]),
        "ksize": ParamSpec("ksize", "背景核数", "int", 51, range=(3, 1001),
                           tooltip="模糊估背景的核边长（奇数）；取目标尺寸 3~5 倍"),
        "flat_path": ParamSpec("flat_path", "参考平场图", "path", "",
                               tooltip="仅 reference 方式：均匀亮板/空场图"),
    }

    def run(self, inputs, params, ctx):
        img = inputs["image"]
        src = img.mat.astype(np.float32)
        mode = params["mode"]
        if mode == "reference":
            from ...io_utils import imread_unicode

            path = str(params.get("flat_path") or "")
            flat = imread_unicode(path) if path else None
            if flat is None:
                raise ValueError("reference 方式需要可读取的参考平场图")
            if img.mat.ndim == 2 and flat.ndim == 3:
                flat = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
            elif img.mat.ndim == 3 and flat.ndim == 2:
                flat = cv2.cvtColor(flat, cv2.COLOR_GRAY2BGR)
            if flat.shape[:2] != img.mat.shape[:2]:
                ctx.warn("平场校正：参考图尺寸与输入不同，已缩放对齐")
                flat = cv2.resize(flat, (img.mat.shape[1], img.mat.shape[0]))
            bg = flat.astype(np.float32)
        else:
            k = ensure_odd(params["ksize"], ctx, "背景核数")
            bg = _smooth_background(src, k)
        axes = (0, 1)
        mean = bg.mean(axis=axes, keepdims=True) if bg.ndim == 3 else float(bg.mean())
        if mode == "blur_subtract":
            out = src - bg + mean
        else:
            out = src / np.maximum(bg, 1.0) * mean
        return {"image": _as_image(np.clip(out, 0, 255).astype(np.uint8), img.meta)}


@register_node
class BitPlane(VisionNode):
    """位平面：提取灰度图第 N 位（0=最低位，7=最高位）为黑白图；
    或「保留高位」把低于第 N 位的比特清零（灰度量化，突出大尺度结构）。

    彩色输入先转灰度。
    """

    NAME = "bit_plane"
    TITLE = "位平面"
    CATEGORY = "预处理"
    INPUTS = (Port("image", "image", label="图像"),)
    OUTPUTS = (Port("image", "image", label="图像"),)
    PARAMS = {
        "plane": ParamSpec("plane", "位序号", "int", 7, range=(0, 7),
                           tooltip="0=最低位（噪声多），7=最高位（主体轮廓）"),
        "keep_above": ParamSpec("keep_above", "保留高位", "bool", False,
                                tooltip="开：保留第 N 位及以上（量化图）；关：只取第 N 位（黑白图）"),
    }

    def run(self, inputs, params, ctx):
        img = inputs["image"]
        gray = to_gray(img.mat)
        p = int(params["plane"])
        if params["keep_above"]:
            mask = np.uint8((0xFF << p) & 0xFF)
            out = np.bitwise_and(gray, mask)   # cv2.bitwise_and 不收 numpy 标量
        else:
            out = (((gray >> p) & 1) * 255).astype(np.uint8)
        return {"image": _as_image(out, img.meta)}


