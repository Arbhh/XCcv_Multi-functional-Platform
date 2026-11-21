"""推理服务配置：platform_data/service/config.json（首次启动自动生成）。

预留更新接口：新增配置字段给默认值即可向后兼容（旧配置文件缺字段自动回退）。
"""

import json
import secrets
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..app_paths import app_root

SERVICE_DATA_DIR = app_root() / "platform_data" / "service"
CONFIG_PATH = SERVICE_DATA_DIR / "config.json"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8600
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


@dataclass
class ServiceConfig:
    """服务运行配置（JSON 持久化；改 Key/端口后需重启服务生效）。"""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    api_key: str = ""                       # 空 = 首次启动自动生成
    model_dirs: list[str] = field(default_factory=list)  # 路径白名单（跨机部署必填）
    allow_any_path: bool = True             # 仅回环绑定语义下放行任意本地路径
    max_loaded_models: int = 3              # LRU 驻留上限（超出自动卸载最久未用）
    queue_size: int = 8                     # 推理并发准入上限（超出返回 429）
    infer_conf: float = 0.25                # 默认置信度（调用方可按请求覆盖）
    infer_iou: float = 0.20                 # 默认 IOU（V0.7.1 全平台统一 0.2）
    timing_enabled: bool = True             # 响应头 X-XCcv-Timing 分段耗时
    max_infer_side: int = 0                 # 推理降采样边长上限（0=按模型 meta 自适应）
    min_loaded_models: list[str] = field(default_factory=list)  # 启动即预载的模型路径
    access_log: bool = False                # uvicorn 访问日志（高频调用建议关）
    log_level: str = "info"                 # uvicorn 日志级别
    max_batch: int = 32                     # 批量推理单请求张数上限（超出 400）
    max_upload_mb: float = 100.0            # 单请求体体积上限 MB（超出 413）
    max_pixels: int = 50_000_000            # 单图像素上限 w*h（超出 413）

    def __post_init__(self):
        # 非回环绑定强制收口：必须白名单 + 关闭任意路径放行（安全红线）
        if self.host not in _LOOPBACK_HOSTS:
            self.allow_any_path = False


def load_config() -> ServiceConfig:
    """读取配置（缺文件生成含随机 Key 的默认配置；坏字段逐项回退默认）。"""
    cfg = ServiceConfig()
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    for f in ("host", "port", "api_key", "model_dirs", "allow_any_path",
              "max_loaded_models", "queue_size", "infer_conf", "infer_iou",
              "timing_enabled", "max_infer_side", "min_loaded_models",
              "access_log", "log_level", "max_batch", "max_upload_mb",
              "max_pixels"):
        if f in data:
            try:
                setattr(cfg, f, data[f])
            except (TypeError, ValueError):
                pass  # 坏值回退默认（留痕）
    cfg.__post_init__()
    if not cfg.api_key:
        cfg.api_key = generate_api_key()
        save_config(cfg)
    return cfg


def save_config(cfg: ServiceConfig) -> None:
    SERVICE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=1),
                           encoding="utf-8")


def generate_api_key() -> str:
    return secrets.token_hex(16)


