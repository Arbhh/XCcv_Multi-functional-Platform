"""T10-1 分段打点：推理请求分段计时（§四.5.1）。

分段口径：decode（图片字节→np）/ pre（缩放与入模前处理）/ infer（模型推理）/
post（结果行构建）/ serialize（响应编码，含渲染 imencode）/ total（全程）。

用法（sync 端点内）：
    timer = StageTimer()
    with timer.stage("decode"):
        ...
    resp.headers["X-XCcv-Timing"] = timer.header()

cfg.timing_enabled=False 时端点直接不挂 header（计时对象本身开销 <1µs/段，
可忽略）。CPU 基线与【G】GPU 基线共用同一分段口径（runs/perf_baseline_v06.py）。
"""

import time
from contextlib import contextmanager


class StageTimer:
    """分段计时器：total 从构造起算，stage 作用域累计同名段。"""

    def __init__(self):
        self._t0 = time.perf_counter()
        self.stages: dict = {}

    @contextmanager
    def stage(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t
            self.stages[name] = round(self.stages.get(name, 0.0) + dt, 4)

    def total(self) -> float:
        return round(time.perf_counter() - self._t0, 4)

    def header(self) -> str:
        """X-XCcv-Timing 头值：`decode=1.2ms,infer=88.0ms,total=91.5ms`。"""
        parts = [f"{k}={round(v * 1000, 1)}ms"
                 for k, v in self.stages.items()]
        parts.append(f"total={round(self.total() * 1000, 1)}ms")
        return ",".join(parts)
