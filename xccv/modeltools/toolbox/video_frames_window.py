# -*- coding: utf-8 -*-
"""视频切分图片：OpenCV 抽帧（间隔帧数 / 目标帧率 / 时长上限）。

输出命名 <视频名>_f<序号>.png；可选感知哈希去重（dHash，跳过与上一张
导出帧几乎相同的画面）。抽帧为后台任务带进度条。
"""

import logging

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QFormLayout, QGroupBox, QLabel, QMessageBox, QProgressBar,
    QPushButton, QVBoxLayout,
)

from ...engine.dataops.video_frames import FramePlan, export_frames, plan_estimate
from ...i18n import tr
from .base import DataToolWindow
from .common import (
    DirRow, FileRow, NoWheelDoubleSpin, NoWheelSpin, confirm_execute,
)
from .registry import ToolSpec, register

log = logging.getLogger(__name__)


class VideoFramesWindow(DataToolWindow):
    """视频抽帧工具窗口。"""

    tool_key = "video_frames"

    def __init__(self, on_closed=None):
        super().__init__(on_closed=on_closed)

    @property
    def tool_title(self):
        return tr("视频切分图片")

    def _build_settings(self, lay: QVBoxLayout):
        opts = QGroupBox(tr("抽帧参数"))
        form = QFormLayout(opts)
        self.video_row = FileRow(tr("视频文件"), tr("视频 (*.mp4 *.avi *.mov *.mkv *.wmv *.flv)"))
        form.addRow(self.video_row)
        self.dst_row = DirRow(tr("输出目录"))
        form.addRow(self.dst_row)

        self.interval_spin = NoWheelSpin()
        self.interval_spin.setRange(1, 10000)
        self.interval_spin.setValue(30)
        self.interval_spin
        form.addRow(tr("间隔帧数（与目标帧率二选一）"), self.interval_spin)

        self.fps_spin = NoWheelDoubleSpin()
        self.fps_spin.setRange(0.0, 240.0)
        self.fps_spin.setSingleStep(0.5)
        self.fps_spin.setValue(0.0)
        self.fps_spin
        form.addRow(tr("目标帧率（>0 优先生效）"), self.fps_spin)

        self.dur_spin = NoWheelDoubleSpin()
        self.dur_spin.setRange(0.0, 100000.0)
        self.dur_spin.setValue(0.0)
        self.dur_spin
        form.addRow(tr("时长上限（秒，0=不限）"), self.dur_spin)

        self.max_spin = NoWheelSpin()
        self.max_spin.setRange(0, 1000000)
        self.max_spin.setValue(0)
        self.max_spin
        form.addRow(tr("导出帧数上限（0=不限）"), self.max_spin)

        self.dedup_box = QCheckBox(tr("感知哈希去重（跳过连续近似帧）"))
        form.addRow(self.dedup_box)
        lay.addWidget(opts)

        self.estimate_btn = QPushButton(tr("预估帧数"))
        self.estimate_btn.clicked.connect(self._estimate)
        lay.addWidget(self.estimate_btn)

        self.hint = QLabel(tr("先预估，确认后开始抽帧"))
        self.hint.setObjectName("hint")
        self.hint.setWordWrap(True)
        lay.addWidget(self.hint)

        self.bar = QProgressBar()
        self.bar.setAlignment(Qt.AlignCenter)
        self.bar.setVisible(False)
        lay.addWidget(self.bar)

        self.exec_btn = QPushButton(tr("开始抽帧"))
        self.exec_btn.clicked.connect(self._export)
        lay.addWidget(self.exec_btn)

    def _validate(self):
        video = self.video_row.path()
        dst = self.dst_row.path()
        if video is None or not video.exists():
            QMessageBox.information(self, tr("请检查"), tr("请先选择视频文件"))
            return None, None
        if dst is None:
            QMessageBox.information(self, tr("请检查"), tr("请填写输出目录"))
            return None, None
        return video, dst

    def _estimate(self):
        video, dst = self._validate()
        if video is None or self._busy():
            return

        def _work():
            plan = FramePlan(video=video, dst=dst,
                             interval=self.interval_spin.value(),
                             target_fps=self.fps_spin.value(),
                             max_duration_s=self.dur_spin.value(),
                             max_frames=self.max_spin.value(),
                             dedup=self.dedup_box.isChecked())
            return plan_estimate(plan)

        self._set_running(True)
        self._start_worker(_work, self._on_estimate_done)

    def _on_estimate_done(self, result: dict):
        self._set_running(False)
        if not result.get("ok"):
            QMessageBox.information(self, tr("请检查"), result.get("error", tr("无法打开视频")))
            return
        self.hint.setText(tr("源 {fps} fps / 共 {n} 帧；步长 {s} → 预计导出 {e} 帧").format(
            fps=result["fps"], n=result["total_frames"], s=result["step"],
            e=result["estimate"]))

    def _export(self):
        video, dst = self._validate()
        if video is None or self._busy():
            return
        if not confirm_execute(self, tr("开始抽帧并导出到输出目录？")):
            return
        self.bar.setVisible(True)
        self.bar.setRange(0, 0)  # 帧总数未知 → 忙碌滚动

        def _progress(done, total, name):
            self.bar.setMaximum(total)
            self.bar.setValue(done)

        def _work():
            plan = FramePlan(video=video, dst=dst,
                             interval=self.interval_spin.value(),
                             target_fps=self.fps_spin.value(),
                             max_duration_s=self.dur_spin.value(),
                             max_frames=self.max_spin.value(),
                             dedup=self.dedup_box.isChecked(),
                             progress=_progress)
            return export_frames(plan)

        self._set_running(True)
        self._start_worker(_work, self._on_done)

    def _on_done(self, result):
        self._set_running(False)
        self.bar.setVisible(False)
        self.hint.setText(tr("抽帧完成：导出 {n} 帧，去重跳过 {d} 帧（共扫描 {s} 帧）").format(
            n=result.frames, d=result.skipped_dup, s=result.total_scanned))

    def _set_running(self, running: bool):
        self.estimate_btn.setEnabled(not running)
        self.exec_btn.setEnabled(not running)


register(ToolSpec(
    key="video_frames",
    title="视频切分图片",
    desc="按间隔帧数或目标帧率抽取视频画面",
    color="#22c55e", icon="toolbox_tool.png",
    keywords="video frame 视频 抽帧 切分 图片 去重",
    tooltip="输出文件名为 视频名_f序号.png，可选感知哈希去重跳过连续近似帧，支持时长与帧数上限。",
    factory=VideoFramesWindow,
))
