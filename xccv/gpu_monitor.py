"""GPU 状态观测（nvidia-smi 低频轮询，无常驻高负载纪律）。

部署平台服务运行页常驻一行 GPU 状态（P-state / 温度 / 功耗 / 利用率 /
SM 频率）；P6/P8 即掉频（判读口径）。nvidia-smi 不在 PATH 或无
GPU 时返回 None，UI 自动隐藏该行（零噪音）。
"""

from __future__ import annotations

import csv
import io
import logging
import subprocess

log = logging.getLogger(__name__)

_QUERY_FIELDS = ("pstate", "temperature.gpu", "power.draw",
                 "utilization.gpu", "clocks.sm")
_QUERY = ("nvidia-smi", "--query-gpu=" + ",".join(_QUERY_FIELDS),
          "--format=csv,noheader,nounits")


def query_gpu(timeout: float = 5.0) -> dict | None:
    """单次查询 → {"pstate": "P0", "temp_c": 45, "power_w": 62.4,
    "util_pct": 12, "sm_mhz": 1500}；不可用/失败 → None。"""
    try:
        proc = subprocess.run(_QUERY, capture_output=True, text=True,
                              timeout=timeout, creationflags=getattr(
                                  subprocess, "CREATE_NO_WINDOW", 0))
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        row = next(csv.reader(io.StringIO(proc.stdout.strip())))
        if len(row) < len(_QUERY_FIELDS):
            return None
        return {"pstate": row[0].strip(),
                "temp_c": _f(row[1]),
                "power_w": _f(row[2]),
                "util_pct": _f(row[3]),
                "sm_mhz": _f(row[4])}
    except Exception:
        log.debug("nvidia-smi 查询失败", exc_info=True)
        return None


def _f(v: str) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def format_gpu_line(info: dict) -> str:
    """dict → 一行 UI 文案（P6/P8 掉频提示）。"""
    warn = ""
    p = str(info.get("pstate") or "").upper()
    if p in ("P6", "P8", "P10", "P12"):
        warn = "　⚠ 疑似掉频（性能状态偏低，查电源/驱动设置）"
    return (f"GPU：性能状态 {info.get('pstate')}　温度 {info.get('temp_c'):.0f}℃"
            f"　功耗 {info.get('power_w'):.0f}W　利用率 {info.get('util_pct'):.0f}%"
            f"　SM 频率 {info.get('sm_mhz'):.0f}MHz{warn}")

