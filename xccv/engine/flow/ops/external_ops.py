"""外部算子节点：在流程里调用「外部视觉引擎后端」的算子 / 过程 / 方案。

后端以适配器注册（`engine/flow/external/`，ExternalEngine 接口）：仓库内置
OpenCV 展示后端（mock，常见算子语义可直接运行）；真实厂商 SDK 后端按同一
接口注册后即可在节点中选择，节点逻辑零改动。后端不可用/算子失败时节点报
可读原因（运行失败标红、不崩溃），其余节点照常运行。输出对接平台统一 IR：
- image：后端返回的图像（未返回时透传输入图，方便后续节点继续）；
- value：第一个数值（NumberVal，附加数值端口）；
- result：全部数值（MeasureResult，统一结果端口）。
"""

import json

import numpy as np

from ..ir import ImageData, MeasureResult, NumberVal
from ..node_base import ParamSpec, Port, VisionNode
from ..registry import register_node


def _names(text) -> list:
    return [s.strip() for s in str(text or "").replace("，", ",").split(",") if s.strip()]


@register_node
class ExternalOp(VisionNode):
    """外部算子：调用外部引擎后端的算子（如 mean_image）或过程/方案文件。

    内置 OpenCV 展示后端：算子名填 OpenCV 风格算子（mean_image / threshold /
    edges_sub_pix 等），控制参数填 JSON 数组；过程/方案文件返回演示结果。
    真实厂商 SDK 后端（HALCON、VisionMaster 等）按 ExternalEngine 接口注册
    后即可在本节点的「外部引擎」中选择。
    """

    NAME = "external_op"
    TITLE = "外部算子（插件后端）"
    CATEGORY = "自定义"
    INPUTS = (Port("image", "image", label="图像"),)
    OUTPUTS = (Port("image", "image", label="图像"),
               Port("value", "number", label="数值"),
               Port("result", "result", label="全部数值"))
    PARAMS = {
        "engine": ParamSpec("engine", "外部引擎", "enum", "mock",
                            options=["mock"]),
        "op": ParamSpec("op", "算子/文件", "text", "",
                        tooltip="算子名（如 mean_image / threshold）或过程/方案文件路径"),
        "args": ParamSpec("args", "控制参数(JSON)", "text", "[]",
                          tooltip="算子：JSON 数组按顺序；过程/方案：JSON 对象按参数名"),
        "procedure": ParamSpec("procedure", "过程/流程名", "text", "",
                               tooltip="过程/流程名（留空=文件名）"),
        "image_param": ParamSpec("image_param", "图像输入名", "text", "Image",
                                 tooltip="过程/方案接收图像的输入参数名"),
        "outputs": ParamSpec("outputs", "数值输出名", "text", "",
                             tooltip="过程/流程的数值输出名，逗号分隔"),
        "image_output": ParamSpec("image_output", "图像输出名", "text", "",
                                  tooltip="过程的图像输出参数名；留空=透传输入图"),
    }

    def run(self, inputs, params, ctx):
        from ..external import ExternalEngineError, get_engine

        name = str(params["engine"])
        engine = get_engine(name)
        if engine is None:
            raise ValueError(f"外部算子：未注册的外部引擎「{name}」")
        ok, msg = engine.available()
        if not ok:
            raise ValueError(f"外部算子不可用：{msg}")
        try:
            raw = json.loads(str(params.get("args") or "[]") or "[]")
        except ValueError as e:
            raise ValueError(f"外部算子：控制参数 JSON 格式错误（{e}）") from None
        call = {"procedure": params.get("procedure"),
                "image_param": params.get("image_param"),
                "outputs": _names(params.get("outputs")),
                "image_output": params.get("image_output")}
        if isinstance(raw, dict):
            call["control"] = raw
        elif isinstance(raw, list):
            call["args"] = raw
        else:
            raise ValueError("外部算子：控制参数须为 JSON 数组（算子）或对象（过程）")
        img = inputs["image"]
        try:
            res = engine.run(str(params.get("op") or ""), img.mat, call)
        except ExternalEngineError as e:
            raise ValueError(str(e)) from None
        out_img = img
        if res.image is not None and np.asarray(res.image).size:
            out_img = ImageData(mat=np.ascontiguousarray(res.image),
                                meta={**img.meta, "external": engine.name})
        values = [float(v) for v in res.values]
        if res.extra.get("others"):
            ctx.warn(f"外部算子：以下输出未转换（非图像/数值）：{res.extra['others'][:5]}")
        return {"image": out_img,
                "value": NumberVal(value=values[0] if values else 0.0),
                "result": MeasureResult(values=values, unit="")}
