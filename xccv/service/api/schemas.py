"""请求模型与统一错误结构。

响应体不建 pydantic 模型——直接返回 result_json 统一 JSON（schema_version=1，
契约单一真相源在 result_json.py 与 docs/result_schema.md，避免双份定义漂移）。
"""

from pydantic import BaseModel, Field


class LoadModelRequest(BaseModel):
    """POST /v1/models/load 请求体。"""

    path: str = Field(..., description="模型文件路径（.pt/.engine），服务端白名单校验")
    device: str = Field("auto", description="auto / cpu / gpu")
    alias: str = Field("", description="业务别名（默认取文件名）；仅作标识不参与寻址")
    warmup: bool = Field(False, description="加载后用空图预热一次")


class UnloadModelRequest(BaseModel):
    model_id: str


class AdminShutdownRequest(BaseModel):
    confirm: bool = Field(False, description="必须显式 true 才执行")


def error_envelope(code: str, message: str, detail: str = "") -> dict:
    """统一错误结构（SDK 映射为 XccvException 子类）。"""
    return {"code": code, "message": message, "detail": detail}
