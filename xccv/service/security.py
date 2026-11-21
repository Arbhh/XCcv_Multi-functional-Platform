"""API Key 校验与模型路径白名单（服务端硬闸，全部在此收口）。"""

import hmac
from pathlib import Path


class AuthError(Exception):
    """API Key 校验失败（→ 401）。"""


class PathNotAllowed(Exception):
    """模型/流程路径不在白名单（→ 403）。"""


def verify_key(provided: str | None, expected: str) -> bool:
    """常量时间比对；expected 为空表示服务未启用鉴权（不建议的调试形态）。"""
    if not expected:
        return True
    return hmac.compare_digest((provided or "").encode("utf-8"),
                               str(expected).encode("utf-8"))


def ensure_path_allowed(path: str | Path, cfg) -> Path:
    """模型/流程文件路径白名单校验（C# 端永不直接读模型路径的屏障）。

    - allow_any_path=True（默认，仅回环部署）：放行本地任意路径；
    - 跨机部署（host 非 0.0.0.0 收口后）：只允许 model_dirs 下的路径。
    """
    p = Path(path)
    if cfg.allow_any_path:
        return p
    resolved = p.resolve()
    for d in cfg.model_dirs or []:
        try:
            resolved.relative_to(Path(d).resolve())
            return p
        except ValueError:
            continue
    raise PathNotAllowed(
        f"路径不在白名单内：{p}（允许目录：{cfg.model_dirs or '（未配置）'}）")

