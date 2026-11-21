"""OpenCV 展示后端（外部算子的内置演示适配器）。

用途：在没有厂商视觉 SDK 的机器上，「外部算子」节点也能端到端跑通——
常见算子名直接映射到 OpenCV 等价实现（同名语义），过程/方案文件走演示
输出。真实厂商后端（HALCON、VisionMaster 等）实现 ExternalEngine 接口
并经 register_engine 注册即可替换本后端，节点与流程零改动。
"""

import cv2
import numpy as np

from . import ExternalEngine, ExternalEngineError, ExternalResult, register_engine


def _odd(v: int) -> int:
    """核数自动补奇（模糊类算子要求奇数核）。"""
    v = max(1, int(v))
    return v if v % 2 == 1 else v + 1


def _odd_pair(args, default: int) -> tuple:
    """[w, h] 或 [k] → 奇数核尺寸对。"""
    if len(args) >= 2:
        return _odd(args[0]), _odd(args[1])
    k = _odd(args[0]) if len(args) else _odd(default)
    return k, k


def _gray(mat):
    if mat.ndim == 3:
        return cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
    return mat


class MockEngine(ExternalEngine):
    """OpenCV 等价实现的展示后端：同名算子语义映射 + 过程文件演示输出。"""

    name = "mock"
    title = "OpenCV 展示后端（内置演示）"

    # 展示后端直接支持的算子（名称 → 一句话说明；其余名称走演示回退）
    _OPS = {
        "mean_image": "均值滤波（ksize [w, h]）",
        "gauss_filter": "高斯模糊（ksize [w, h]，奇数）",
        "median_blur": "中值模糊（ksize 标量）",
        "binomial_filter": "3x3 平滑",
        "rgb1_to_gray": "转灰度（BGR→GRAY）",
        "threshold": "阈值分割（[thr]，全局二值）",
        "edges_sub_pix": "Canny 边缘（[t1, t2]）",
        "invert_image": "反色",
        "count_regions": "连通域计数（返回区域数）",
    }

    def available(self) -> tuple:
        return True, ("内置演示后端：常见算子以 OpenCV 等价实现直接运行；"
                      "真实厂商 SDK 请按 ExternalEngine 接口注册自定义适配器")

    def list_ops(self) -> list:
        return sorted(self._OPS)

    def run(self, op: str, image, params: dict) -> ExternalResult:
        if image is None:
            raise ExternalEngineError("mock：输入图像为空")
        mat = np.asarray(image)
        name = str(op or "").strip()
        lower = name.lower().rsplit(".", 1)[-1]  # 过程/方案文件取词干名
        args = params.get("args") if isinstance(params.get("args"), (list, tuple)) else []
        values: list = []
        try:
            if lower == "mean_image":
                mat = cv2.blur(mat, _odd_pair(args, 9))
            elif lower == "gauss_filter":
                mat = cv2.GaussianBlur(mat, _odd_pair(args, 9), 0)
            elif lower == "median_blur":
                mat = cv2.medianBlur(mat, _odd(args[0]) if len(args) else 5)
            elif lower == "binomial_filter":
                mat = cv2.blur(mat, (3, 3))
            elif lower in ("rgb1_to_gray", "rgb_to_gray"):
                mat = _gray(mat)
            elif lower == "threshold":
                thr = int(args[0]) if len(args) else 128
                mat = cv2.threshold(_gray(mat), thr, 255, cv2.THRESH_BINARY)[1]
                values.append(float(thr))
            elif lower == "edges_sub_pix":
                t1 = int(args[0]) if len(args) else 80
                t2 = int(args[1]) if len(args) > 1 else 180
                mat = cv2.Canny(_gray(mat), t1, t2)
            elif lower == "invert_image":
                mat = cv2.bitwise_not(mat)
            elif lower == "count_regions":
                n, _ = cv2.connectedComponents(_gray(mat))
                values.append(float(n - 1))
            else:
                # 未识别的算子 / 过程 / 方案文件：演示语义 = 轻度平滑 + 均值输出
                mat = cv2.blur(mat, (5, 5))
                values.append(round(float(mat.mean()), 3))
        except ExternalEngineError:
            raise
        except Exception as e:  # noqa: BLE001  转可读原因，节点标红不崩
            raise ExternalEngineError(f"mock：算子「{name}」执行失败（{e}）") from None
        return ExternalResult(image=mat, values=values, extra={"backend": self.name})


register_engine(MockEngine())

