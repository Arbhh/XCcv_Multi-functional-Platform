"""外部视觉引擎接口：为「外部算子」节点提供可插拔的厂商后端。

- ExternalEngine：适配器基类——name / title / available() / list_ops() / run()；
- 注册表：register_engine / get_engine / list_engines；内置 OpenCV 展示后端
  （mock，常见算子语义可直接运行）。真实厂商 SDK（HALCON、VisionMaster 等）
  按同一接口实现适配器并注册即可接入，节点与流程零改动；
- 结果统一：run() 返回 ExternalResult(image, values, extra)，由节点转成平台 IR
  （ImageData / NumberVal / MeasureResult），与统一结果格式衔接。
"""

from dataclasses import dataclass, field


class ExternalEngineError(RuntimeError):
    """外部引擎调用失败（可读原因；节点据此报错，不弹堆栈）。"""


@dataclass
class ExternalResult:
    image: object = None                          # ndarray（BGR/GRAY uint8）或 None
    values: list = field(default_factory=list)    # 数值输出（按输出顺序展平）
    extra: dict = field(default_factory=dict)     # 其它输出（字符串、原始对象摘要等）


class ExternalEngine:
    """外部引擎适配器基类。子类实现 available / run（list_ops 可选）。"""

    name = ""
    title = ""

    def available(self) -> tuple:
        """(是否可用, 说明)——不可用时说明即安装/配置指引。不得抛异常。"""
        return False, f"{self.title or self.name}：适配器未实现"

    def list_ops(self) -> list:
        return []

    def run(self, op: str, image, params: dict) -> ExternalResult:
        raise ExternalEngineError(f"{self.title or self.name}：适配器未实现 run()")


_ENGINES: dict = {}
_BUILTIN_LOADED = False


def register_engine(engine: ExternalEngine) -> ExternalEngine:
    """注册（或替换）一个外部引擎适配器；返回该适配器。"""
    if not engine.name:
        raise ValueError("外部引擎适配器必须有 name")
    _ENGINES[engine.name] = engine
    return engine


def _ensure_builtin() -> None:
    global _BUILTIN_LOADED
    if not _BUILTIN_LOADED:
        _BUILTIN_LOADED = True
        from . import mock_adapter  # noqa: F401  导入即注册


def get_engine(name: str):
    _ensure_builtin()
    return _ENGINES.get(str(name or ""))


def list_engines() -> list:
    """[{name, title, available, message}]——设置/诊断界面与测试用。"""
    _ensure_builtin()
    out = []
    for e in _ENGINES.values():
        ok, msg = e.available()
        out.append({"name": e.name, "title": e.title, "available": bool(ok), "message": msg})
    return out
