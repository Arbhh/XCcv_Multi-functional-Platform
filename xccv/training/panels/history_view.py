"""训练历史视图（HistoryViewMixin）：历史记录列表条目/重扫/选中，
进度与时长刷新、损失/指标曲线绘制、逐类指标表、指标摘要、
记录对比与删除。

0.7.6 P-1 页级拆分：方法体与拆分前 training_window.py 完全一致
（逐行搬运）；组合根 ProjectWindow 经 mixin 继承装配。
"""

import logging
log = logging.getLogger(__name__)

import math
import shutil
from datetime import datetime
from pathlib import Path
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor, QIcon
from PyQt5.QtWidgets import QListWidgetItem, QMessageBox, QTableWidgetItem
from ...i18n import tr
from ..runs import RunRecord, format_duration, scan_runs, STATUS_COLOR
from ...ui_widgets import Badge, HoverCard, load_prefs, relative_time, save_prefs, toast
from .common import (
    _CURVE_COLORS, _status_dot_icon,
)


# ============================================================
# 页面方法（自 training_window.py 逐行搬运，语义不变）
# ============================================================

class HistoryViewMixin:
    # ================= 训练历史 =================
    def _make_run_item(self, rec: RunRecord) -> QListWidgetItem:
        """历史列表条目：状态点图标 + 时间 + 状态（按状态着色）+ 相对时间。"""
        model = rec.model_name or (self.project.model_name if rec.status == "running" else "")
        line2 = f"{rec.status_text}　{model}".strip()
        if rec.start_time:
            when = relative_time(rec.start_time)
            if when != rec.start_time:   # 解析失败时原样返回，不重复展示
                line2 = f"{rec.status_text} · {when}　{model}".strip()
        item = QListWidgetItem(QIcon(_status_dot_icon(
            STATUS_COLOR.get(rec.status, "#64748b"))),
            f"{rec.start_time or rec.run_dir.name}\n{line2}")
        item.setData(Qt.UserRole, str(rec.run_dir))
        item.setData(Qt.UserRole + 1, rec)
        item.setForeground(QBrush(QColor(STATUS_COLOR.get(rec.status, "#64748b"))))
        return item
    def _reload_history(self, select_dir: Path | None = None):
        """重扫模型储存目录刷新左侧列表；select_dir 给定时优先选中该记录。"""
        target = select_dir or self._selected_run_dir
        self.run_list.blockSignals(True)
        self.run_list.clear()
        self._prev_selected = []  # 列表已清空，缓存的多选状态随之失效
        n = 0
        if self.thread is not None and self._run_dir is not None:
            done = self._live_history[-1]["epoch"] if self._live_history else 0
            live = RunRecord(run_dir=self._run_dir, status="running",
                             start_time=self._live_start,
                             epochs_done=done, epochs_planned=self.project.epochs)
            self.run_list.addItem(self._make_run_item(live))
            n += 1
        for rec in scan_runs(self.project.model_dir):
            if self.thread is not None and self._run_dir is not None \
                    and rec.run_dir == self._run_dir:
                continue  # 仅训练进行中时跳过（进行中的目录已在顶部展示）
            self.run_list.addItem(self._make_run_item(rec))
            n += 1
        self.run_hint.setText("")
        empty = n == 0
        self.run_list.setVisible(not empty)
        self.run_empty.setVisible(empty)
        selected = None
        if target is not None:
            found = self.run_list.findItems(str(target), Qt.MatchExactly)
            if found:
                selected = found[0]
        if selected is None and self.run_list.count():
            selected = self.run_list.item(0)
        if selected is not None:
            self.run_list.setCurrentItem(selected)
        self.run_list.blockSignals(False)
        if selected is not None:
            self._on_run_selected(selected, None)
        else:
            self.train_status.setText(tr("尚未开始训练"))
            self._history = []
            self._reset_progress()
            self._update_plots()
            self.btn_delete_run.setEnabled(False)
    def _on_run_selected(self, cur, _prev=None):
        if cur is None:
            return
        rec: RunRecord = cur.data(Qt.UserRole + 1)
        self._selected_run_dir = rec.run_dir
        self.btn_open_dir.setEnabled(True)
        self._show_run(rec)
    def _show_run(self, rec: RunRecord):
        """渲染所选记录：实时训练显示进行中的数据，其余读 metrics.json。"""
        m = rec.metrics or {}
        live = rec.status == "running" and self.thread is not None
        if live:
            self._history = self._live_history
            self._build_echo_form(run_dir=rec.run_dir)
            self._update_overall_progress()
            self.run_start.setText(self._live_start or "—")
            self.run_end.setText("—")
            self._tick_duration()
        else:
            self._history = list(m.get("history") or [])
            self._build_echo_form(config=m.get("config"), run_dir=rec.run_dir)
            if getattr(self, "_train_mode", "epoch") == "free":
                # §四：免训练范式历史记录不走轮次口径
                self.progress_bar.setValue(100)
                self.progress_bar.setFormat(tr("免训练范式（单遍完成）"))
            else:
                self._update_progress(rec.epochs_done, rec.epochs_planned)
            self.run_start.setText(rec.start_time or "—")
            self.run_end.setText(rec.end_time or "—")
            self.run_duration.setText(format_duration(rec.duration_sec))
        self._update_plots()
        if live:
            self.train_status.setText(
                f"训练中… 第 {self._live_history[-1]['epoch']} 轮" if self._live_history else "训练中…")
            self.final_metrics.setText("")
            self.cm_canvas.ax.clear()
            self.cm_canvas.draw()
            self._update_per_class_table(None)  # 进行中无逐类指标
            self.btn_delete_run.setEnabled(False)  # 进行中的记录不可删除
        else:
            self.train_status.setText(f"{tr('历史记录：')}{rec.status_text}")
            err = m.get("error")
            self.final_metrics.setText(
                f"{tr('失败原因：')}{err}" if err else self._summarize_metrics(m))
            matrix = m.get("confusion_matrix")
            classes = m.get("classes") or []
            self.cm_canvas.ax.clear()
            if matrix:
                self.cm_canvas.update_matrix(matrix, classes)
            else:
                self.cm_canvas.draw()
            self._update_per_class_table(m)  # 历史详情逐类表
            self.btn_delete_run.setEnabled(True)
    def _update_progress(self, done: int, planned: int):
        pct = int(round(100 * min(done, planned) / planned)) if planned else 0
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(
            f"%p%　{tr('第')} {done} / {planned} {tr('轮')}" if planned else "%p%")
    def _reset_progress(self):
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.run_start.setText("—")
        self.run_end.setText("—")
        self.run_duration.setText("—")
    def _tick_duration(self):
        if self._live_start_dt is None:
            return
        self.run_duration.setText(format_duration(
            (datetime.now() - self._live_start_dt).total_seconds()))
    def _tick_live(self):
        """训练中每秒刷新：训练时长 + 总体进度（含轮内批次比例）+ GPU 显存。"""
        if self.thread is None:
            self._live_timer.stop()
            self.gpu_mem.setText("")
            return
        if self._selected_run_dir == self._run_dir:
            self._tick_duration()
            self._update_overall_progress()
            self._update_gpu_mem()
    # ================= 曲线绘制 =================
    def _loss_series(self):
        """损失曲线数据：训练损失与验证损失（逐轮）。

        §四：免训练范式无损失语义（历史里的 0.0 占位不画），Loss 区
        由 loss_free_hint 文案接管。"""
        if getattr(self, "_train_mode", "epoch") == "free":
            return []
        h = self._history
        xs = [x["epoch"] for x in h]
        out = []
        for k, label in (("train_loss", tr("训练损失")), ("val_loss", tr("验证损失"))):
            if any(k in x for x in h):
                out.append((label, xs, [x.get(k) for x in h]))
        return out
    def _metric_series(self):
        """评估指标曲线：固定两条 —— mAP 与 mAR。

        R-CNN 族逐轮产出宽松口径 map_loose/mar_loose（IoU {0.10,0.25,0.50}
        三档平均），直接绘制；YOLO 族无该口径，用 map50（AP@50）与 recall
        （IoU@0.5 召回）充当同位曲线。map50_95 / mask_* 等其余键不再进曲线
        （数值仍保留在 metrics.json 与最终指标摘要中）。分类只有 val_top1；
        §四：免训练范式（ANOMALY）为 auroc。
        """
        xs = [h["epoch"] for h in self._history]
        h = self._history
        if any("val_top1" in x for x in h):
            pairs = (("val_top1", "Top-1"),)
        elif any("auroc" in x for x in h):
            pairs = (("auroc", "AUROC"),)
        elif any("map_loose" in x for x in h):
            pairs = (("map_loose", "mAP"), ("mar_loose", "mAR"))
        elif any("map50" in x for x in h):
            pairs = (("map50", "mAP"), ("recall", "mAR"))
        else:
            return []
        return [(label, xs, [x.get(k) for x in h])
                for k, label in pairs if any(k in x for x in h)]
    def _on_batch_progress(self, epoch: int, batch_index: int, batch_total: int):
        """轮内进度上报：只缓存，由 1s 定时器统一刷新进度条（避免频繁重绘）。"""
        self._latest_progress = (epoch, batch_index, batch_total)
    def _update_overall_progress(self):
        """总体进度：轮次式 =（已完成轮次 + 当前轮内批次比例）/ 总轮次；
        §四 免训练范式 = 阶段化（1 特征抽取与建库 / 2 标定 / 3 验证评估）。"""
        if getattr(self, "_train_mode", "epoch") == "free":
            e, bi, nb = (self._latest_progress or (1, 0, 1))
            frac = ((max(int(e), 1) - 1) + (bi / nb if nb else 1.0)) / 3.0
            self.progress_bar.setValue(int(round(100 * min(frac, 1.0))))
            stage_names = {1: "特征抽取与建库", 2: "标定", 3: "验证评估"}
            sname = stage_names.get(int(e), "")
            eta = ""
            if 0.02 <= frac < 1.0 and self._live_start_dt is not None:
                elapsed = (datetime.now() - self._live_start_dt).total_seconds()
                remain_s = elapsed / frac * (1.0 - frac)
                eta = f"　{tr('剩余约')} {format_duration(remain_s)}"
            self.progress_bar.setFormat(
                f"%p%　{tr('阶段')} {e} / 3 {tr(sname)}　{bi}/{nb}{eta}")
            return
        planned = self.project.epochs
        if not planned:
            return
        done = self._live_history[-1]["epoch"] if self._live_history else 0
        frac = done / planned
        if self.thread is not None and self._latest_progress:
            e, bi, nb = self._latest_progress
            frac = max(frac, ((e - 1) + (bi / nb if nb else 1.0)) / planned)
            done = max(done, e)
        self.progress_bar.setValue(int(round(100 * min(frac, 1.0))))
        eta = ""
        if 0.02 <= frac < 1.0 and self._live_start_dt is not None:
            elapsed = (datetime.now() - self._live_start_dt).total_seconds()
            remain_s = elapsed / frac * (1.0 - frac)
            eta = f"　{tr('剩余约')} {format_duration(remain_s)}"
        self.progress_bar.setFormat(f"%p%　{tr('第')} {done} / {planned} {tr('轮')}{eta}")
    def _update_gpu_mem(self):
        """训练页顶部实时显示 GPU 显存占用。"""
        try:
            import torch

            if not torch.cuda.is_available():
                self.gpu_mem.setText("")
                return
            free_b, total_b = torch.cuda.mem_get_info()
            used = (total_b - free_b) / 2**30
            self.gpu_mem.setText(f"{tr('GPU 显存')} {used:.1f} / {total_b / 2**30:.1f} GB")
        except Exception:
            log.debug("GPU 显存查询失败（隐藏显存标签）")
            self.gpu_mem.setText("")
    def _on_epoch(self, epoch: int, logs: dict):
        self._live_history.append({"epoch": epoch, **logs})
        if self._selected_run_dir == self._run_dir:  # 正在查看本次训练 → 实时刷新
            self._history = self._live_history
            self._update_plots()
            self._update_overall_progress()
            self._tick_duration()
            if getattr(self, "_train_mode", "epoch") == "free":
                # §四：免训练范式的 on_epoch = 阶段级回调（1/2/3）
                self.train_status.setText(
                    f"{tr('训练中…')} {tr('阶段')} {epoch} / 3")
            else:
                self.train_status.setText(
                    f"{tr('训练中…')} {tr('第')} {epoch} {tr('轮')}")
    def _update_plots(self):
        xs = [h["epoch"] for h in self._history]
        width = 3  # 曲线加粗，长时观看更醒目
        # §四：免训练范式 Loss 区提示可见、轮次轴改"阶段"
        if hasattr(self, "loss_free_hint"):
            self.loss_free_hint.setVisible(
                getattr(self, "_train_mode", "epoch") == "free")
        # 损失曲线：训练损失 + 验证损失；每个轮次叠加空心点标注数据位置
        self.loss_plot.clear()
        for i, (key, sx, ys) in enumerate(self._loss_series()):
            color = _CURVE_COLORS[i % len(_CURVE_COLORS)]
            self.loss_plot.plot(
                sx, ys, pen=pg.mkPen(color, width=width), name=key,
                symbol="o", symbolSize=7,
                symbolPen=pg.mkPen(color, width=1.5),  # symbolBrush 默认空 → 空心点
            )
        # 评估指标固定两条：mAP 与 mAR（YOLO 用 AP@50/召回充当）
        self.metric_plot.clear()
        for i, (key, _, ys) in enumerate(self._metric_series()):
            color = _CURVE_COLORS[i % len(_CURVE_COLORS)]
            self.metric_plot.plot(
                xs, ys, pen=pg.mkPen(color, width=width), name=key,
                symbol="o", symbolSize=7, symbolPen=pg.mkPen(color, width=1.5),
            )
        # X 轴随轮次自适应（整数范围）；损失 Y 轴固定从 0 起、上限随实际损失
        # 自适应（pyqtgraph 默认自适应会把下限抬到最小值附近，损失零点参照消失）；
        # 指标图 Y 轴锁回 0~1
        x_max = max(2.5, len(self._history) + 0.5)
        loss_vals = [v for _, _, ys in self._loss_series() for v in ys
                     if isinstance(v, (int, float)) and math.isfinite(v)]
        loss_top = max(loss_vals) * 1.08 if loss_vals else 1.0
        self.loss_plot.setXRange(0.5, x_max, padding=0)
        self.loss_plot.setYRange(0, loss_top, padding=0)
        self.metric_plot.setXRange(0.5, x_max, padding=0)
        self.metric_plot.setYRange(0, 1, padding=0)
        self._loss_hover.install()   # clear() 会移除图元，重绘后重挂悬停提示
        self._metric_hover.install()
    def _on_train_finished(self, ok: bool, message: str, metrics: dict):
        self._live_timer.stop()
        self.thread = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if ok:
            self.train_status.setText("训练完成 ✔")
            self.final_metrics.setText(self._summarize_metrics(metrics))
            matrix = metrics.get("confusion_matrix")
            classes = metrics.get("classes") or []
            if matrix:
                self.cm_canvas.update_matrix(matrix, classes)
            self._update_per_class_table(metrics)  # 训练完成即出逐类表
        else:
            self.train_status.setText(tr("训练失败 ✘"))
            self.final_metrics.setText(f"{tr('失败原因：')}{message}")
            self._update_per_class_table(None)
            QMessageBox.critical(self, "训练失败", message)
        # 重扫历史（本次训练目录已有产出），保持选中该记录展示最终详情
        self._reload_history(select_dir=self._run_dir)
    def _update_per_class_table(self, metrics: dict | None):
        """逐类精确率/召回率表刷新。

        metrics 为 None 或无混淆矩阵（分类任务/失败/训练中）时清空——
        表只对检测/分割的历史详情与训练完成态有意义。
        """
        from ...engine.metrics.per_class import per_class_rows

        t = self.per_class_table
        t.setRowCount(0)
        if not metrics:
            return
        cm = metrics.get("confusion_matrix")
        classes = metrics.get("classes") or []
        if not cm or not classes:
            return
        final = metrics.get("final") or {}
        try:
            rows = per_class_rows(cm, classes, final.get("per_class_ap50"))
        except ValueError:  # 尺寸不符（异常档位）不阻塞详情展示
            return

        def _s(v):
            if v is None:
                return "-"
            return f"{v:.3f}" if isinstance(v, float) else str(v)

        t.setRowCount(len(rows))
        for r, row in enumerate(rows):
            vals = [row["label"], _s(row["gt"]), _s(row["pred"]), _s(row["tp"]),
                    _s(row["fp"]), _s(row["fn"]), _s(row["precision"]),
                    _s(row["recall"]), _s(row["ap50"])]
            for c, text in enumerate(vals):
                t.setItem(r, c, QTableWidgetItem(text))
    @staticmethod
    def _summarize_metrics(metrics: dict) -> str:
        parts = []
        if metrics.get("final_val_top1") is not None:
            parts.append(f"{tr('最终验证 Top-1')}：{metrics['final_val_top1']:.1%}")
        if metrics.get("best_val_top1") is not None:
            parts.append(f"{tr('最优验证 Top-1')}：{metrics['best_val_top1']:.1%}")
        final = metrics.get("final") or {}
        for key, label in (("map_loose", "mAP"), ("map50", "mAP@50"),
                           ("map50_95", "mAP@50-95"), ("mar_loose", "mAR"),
                           ("precision", tr("精确率")), ("recall", tr("召回率"))):
            if key in final:
                parts.append(f"{label}：{final[key]:.3f}")
        # OK 负样本贡献常驻可见（能力已在，让用户看见）
        neg_train = metrics.get("neg_train_images")
        if neg_train:
            neg_val = final.get("neg_val_images")
            fp = final.get("neg_fp_per_img")
            ok_txt = f"{tr('OK 负样本')}：{tr('训练')} {neg_train} {tr('张')}"
            if neg_val is not None:
                ok_txt += f" / {tr('验证')} {neg_val} {tr('张')}"
            if fp is not None:
                ok_txt += f"，{tr('误检')} {fp:.2f} {tr('框/张')}"
            parts.append(ok_txt)
        # Loss 曲线自动判读（欠拟合/过拟合/NaN 一句话）
        # §四：免训练范式无损失语义，跳过判读（"轮次不足 6 轮"无意义；
        # 本方法为 staticmethod，按 metrics.config.model_name 查 META）
        free = False
        try:
            from ...engine.models.registry import MODEL_META

            free = (MODEL_META.get(
                (metrics.get("config") or {}).get("model_name") or "",
                {}).get("train_mode") == "free")
        except Exception:  # noqa: BLE001  元数据不可得按轮次式
            free = False
        if not free:
            from ..loss_diagnosis import diagnose

            d = diagnose(metrics.get("history") or [])
            if d.get("text"):
                parts.append(d["text"])
        return "　|　".join(parts)
    def _current_run_record(self) -> RunRecord | None:
        item = self.run_list.currentItem()
        return item.data(Qt.UserRole + 1) if item is not None else None
    # ---- 实验对比（新窗口，2~3 条，含数据集指纹提示）----
    def _on_run_selection_changed(self):
        """多选上限 3 条：第 4 个 Ctrl+点击被拒绝（恢复上一状态）并提示。"""
        items = self.run_list.selectedItems()
        if len(items) > 3:
            keep = {it.data(Qt.UserRole) for it in self._prev_selected}
            self.run_list.blockSignals(True)
            for it in items:
                if it.data(Qt.UserRole) not in keep:
                    it.setSelected(False)
            self.run_list.blockSignals(False)
            self.statusBar().showMessage(tr("最多选择 3 条同时对比"), 4000)
            items = self.run_list.selectedItems()
        self._prev_selected = list(items)
    def _compare_candidates(self) -> list:
        """当前多选中的可对比记录（进行中的训练无 metrics，不参与）。"""
        out = []
        for item in self.run_list.selectedItems():
            rec: RunRecord = item.data(Qt.UserRole + 1)
            if rec is not None and rec.status != "running" and rec.metrics:
                out.append(rec)
        return out
    def _open_compare(self):
        recs = self._compare_candidates()
        if not 2 <= len(recs) <= 3:
            self.statusBar().showMessage(tr("请先选择 2~3 条已完成的训练记录"), 5000)
            return
        from ..compare_window import CompareWindow

        self._compare_window = CompareWindow(self.project, recs)
        self._compare_window.show()
        self._compare_window.raise_()
    def _delete_current_run(self):
        """删除当前选中的历史训练记录（连带输出文件夹，无法恢复）。"""
        rec = self._current_run_record()
        if rec is None:
            return
        if rec.status == "running" and self.thread is not None:
            QMessageBox.warning(self, tr("无法删除"), tr("该记录对应的训练正在进行中，无法删除"))
            return
        if QMessageBox.question(
                self, tr("删除训练记录"),
                f"{tr('确定删除该条训练记录吗？')}\n{rec.run_dir}\n\n"
                + tr("记录文件夹（含权重、指标、导出数据）将一并删除，且无法恢复。"),
                ) != QMessageBox.Yes:
            return
        shutil.rmtree(rec.run_dir, ignore_errors=True)
        self._selected_run_dir = None
        self._reload_history()
        self.statusBar().showMessage(tr("训练记录已删除"), 3000)


# ---- Cython 编译态槽包装：mixin 方法经继承进入窗口类，编译态同样需要
# 参数裁剪包装（见 xccv/qt_slots.py；幂等）----
from xccv.qt_slots import apply_auto_slot_class  # noqa: E402

apply_auto_slot_class(HistoryViewMixin)
