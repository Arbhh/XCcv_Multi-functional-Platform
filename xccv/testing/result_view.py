"""复盘单图检测结果渲染窗（V0.7.4 bug-优化项二.2）。

双击缩略图后按**复盘时保存的结果**即时叠加渲染（零重推理）：
- 带检出框的任务（检测/分割）：过阈检出画绿框、被阈值过滤的检出画红框
  （绿/红与右侧分组配色一致，阈值取当前维度面板的启用集）；
- 无框任务（异常检测/分类）：在图上叠加图像分数/预测类别横幅，不留空白。

渲染一次性完成（QPainter 画在缩放后的 pixmap 上），窗口为非模态——可同时
开多张对比；「用系统看图打开」保留原看图器路径。
"""

from __future__ import annotations

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (QDialog, QHBoxLayout, QLabel, QPushButton,
                             QScrollArea, QVBoxLayout)

from ..engine.io_utils import imread_unicode
from ..i18n import tr
from .review_logic import det_passes

_GREEN = "#2E9E5B"   # 过阈（与「正确」组同色）
_RED = "#D64545"     # 被过滤（与「过检」组同色）
_MAX_W = 1100        # 渲染画布最大宽（保持纵横比）


def _det_tuple(d: dict) -> list:
    """results 的检出 dict → 阈值过滤用的列序元组（batch_match 列序）。"""
    dims = d.get("_dims") or {}
    return [float(d.get("score") or 0.0), None,
            dims.get("area"), dims.get("short"), dims.get("long"),
            dims.get("fg_ratio"), dims.get("gray_diff")]


def render_result_pixmap(entry: dict, thresholds: dict, task: str = "detection"
                         ) -> QPixmap | None:
    """单图结果 → 叠加渲染的 pixmap（无图返回 None）。"""
    img = imread_unicode(entry.get("path") or "")
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = min(1.0, _MAX_W / max(1, w))
    disp_w, disp_h = max(1, int(w * scale)), max(1, int(h * scale))
    rgb = np.ascontiguousarray(img[:, :, ::-1]) if img.ndim == 3 else \
        np.ascontiguousarray(img)
    qimg = QImage(rgb.tobytes(), w, h, (3 * w) if img.ndim == 3 else w,
                  QImage.Format_RGB888 if img.ndim == 3
                  else QImage.Format_Grayscale8)
    pm = QPixmap.fromImage(qimg).scaled(
        disp_w, disp_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    font = QFont()
    font.setPixelSize(max(14, int(disp_h / 40)))
    p.setFont(font)

    task_v = str(task or "detection")
    if task_v in ("anomaly", "classification"):
        # 无框任务：横幅显示图像分数/预测类别（不留空白）
        if task_v == "anomaly":
            score = entry.get("image_score")
            sem = "z"
            text = f"{tr('图像分数')} {sem} = {score}" if score is not None \
                else tr("无图像分数")
        else:
            cls = (entry.get("classifications") or [{}])[0]
            text = f"{tr('预测')}：{cls.get('label', '—')}  " \
                   f"{cls.get('score', 0.0):.2f}"
        _draw_banner(p, text, disp_w, disp_h)
    else:
        for d in entry.get("detections") or []:
            passed = det_passes(_det_tuple(d), thresholds or {})
            x1, y1, x2, y2 = (max(0, int(v * scale)) for v in d.get("bbox") or
                              (0, 0, 0, 0))
            color = QColor(_GREEN if passed else _RED)
            pen = QPen(color, max(2, int(disp_w / 500) + 1))
            p.setPen(pen)
            p.drawRect(x1, y1, max(1, x2 - x1), max(1, y2 - y1))
            tag = f"{d.get('label', '')} {d.get('score', 0):.2f}"
            _draw_tag(p, tag, x1, max(0, y1 - int(disp_h * 0.03)), color)
    p.end()
    return pm


def _draw_banner(p: QPainter, text: str, w: int, h: int) -> None:
    p.setPen(QPen(QColor(_GREEN), 2))
    pad = max(8, int(h * 0.03))
    rect = p.boundingRect(0, 0, w, h, Qt.AlignLeft | Qt.AlignTop, text)
    p.fillRect(rect.adjusted(-pad, -4, pad, -2), QColor(30, 34, 40, 200))
    p.setPen(QColor("#ffffff"))
    p.drawText(rect, Qt.AlignLeft | Qt.AlignTop, text)


def _draw_tag(p: QPainter, text: str, x: int, y: int, color: QColor) -> None:
    rect = p.boundingRect(x, y, 400, 40, Qt.AlignLeft | Qt.AlignTop, text)
    p.fillRect(rect.adjusted(-2, -2, 2, 0), QColor(30, 34, 40, 190))
    p.setPen(QColor("#ffffff"))
    p.drawText(rect, Qt.AlignLeft | Qt.AlignTop, text)
    p.setPen(QPen(color, 2))


class ResultViewDialog(QDialog):
    """单图检测结果窗：叠加渲染 + 系统看图入口（非模态，可多开对比）。"""

    def __init__(self, parent, entry: dict, thresholds: dict,
                 task: str = "detection"):
        super().__init__(parent)
        self.entry = entry
        name = entry.get("file") or ""
        passed = sum(1 for d in entry.get("detections") or []
                     if det_passes(_det_tuple(d), thresholds or {}))
        total = len(entry.get("detections") or [])
        self.setWindowTitle(
            f"{name}  -  {tr('检测结果')}（{tr('过阈')} {passed}/{total}）")
        self.setMinimumSize(680, 520)
        pm = render_result_pixmap(entry, thresholds, task)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 8)
        lab = QLabel()
        if pm is not None:
            lab.setPixmap(pm)
            lab.setAlignment(Qt.AlignCenter)
        else:
            lab.setText(tr("图片读取失败"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignCenter)
        scroll.setWidget(lab)
        lay.addWidget(scroll, stretch=1)
        tip = QLabel(tr(
            "绿=过阈检出，红=被阈值过滤的检出；渲染取自复盘保存的结果，"
            "不重新推理"))
        tip.setObjectName("hint")
        tip.setWordWrap(True)
        lay.addWidget(tip)
        row = QHBoxLayout()
        btn_sys = QPushButton(tr("用系统看图打开"))
        btn_sys.clicked.connect(
            lambda: self._open_in_viewer(entry.get("path") or ""))
        row.addWidget(btn_sys)
        row.addStretch(1)
        btn_close = QPushButton(tr("关闭"))
        btn_close.clicked.connect(self.close)
        row.addWidget(btn_close)
        lay.addLayout(row)
        from ..i18n import localize_widget_tree

        localize_widget_tree(self)

    @staticmethod
    def _open_in_viewer(path: str) -> None:
        from PyQt5.QtGui import QDesktopServices
        from PyQt5.QtCore import QUrl

        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

