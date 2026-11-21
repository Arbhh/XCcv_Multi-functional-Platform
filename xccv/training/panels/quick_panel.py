"""Quick/专家双轨参数面板（QuickPanelMixin）：快速档 6 参数行、
预设卡换装（轮次 ⇄ 免训练两态）、智能推荐联动、参数双向同步。

0.7.6 P-1 页级拆分：方法体与拆分前 training_window.py 完全一致
（逐行搬运）；组合根 ProjectWindow 经 mixin 继承装配。
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget
from ...i18n import tr
from ..preflight import dataset_facts, param_findings, recommend_settings
from ...ui_style import bind_style, icon_path, token
from ...ui_widgets import Badge, HoverCard, load_prefs, relative_time, save_prefs, toast
from .common import (
    _PRESET_DEFS, _PRESET_DEFS_FREE, _t2, _t3, _cuda_available,
)

import logging

log = logging.getLogger(__name__)


# ============================================================
# 页面方法（自 training_window.py 逐行搬运，语义不变）
# ============================================================

class QuickPanelMixin:
    def _apply_recommendations(self):
        """智能推荐配置：保守规则，应用后在状态栏说明理由（toast 式）。"""
        self._save_params()
        facts = dataset_facts(self._dataset_samples_for_guard())
        recs = recommend_settings(self.project, facts, _cuda_available())
        if not recs:
            self.statusBar().showMessage(tr("当前配置已符合推荐，未做修改"), 5000)
            return
        for key, value, _reason in recs:
            self._apply_recommend_key(key, value)
        self._save_params()
        self.statusBar().showMessage(
            tr("已应用推荐（") + "；".join(r for _, _, r in recs) + tr("）"), 15000)
    def _apply_recommend_key(self, key, value):
        if key == "freeze_backbone":
            self.par_freeze.setChecked(True)
        elif key == "industrial_opt":
            self.par_industrial.setChecked(True)
        elif key == "rpn_reduced":
            self.par_rpn.setChecked(True)
        elif key == "use_amp":
            self.par_amp.setChecked(bool(value))
        elif key == "epochs":  # 轮次智能推荐
            self.par_epochs.setValue(int(value))
        elif key == "augment_probs":  # 已启用算子的概率提高到推荐值
            for op, prob in self.augment_prob_spins.items():
                if self.augment_controls[op][0].isChecked():
                    prob.setValue(max(prob.value(), float(value)))
    # ================= Quick/专家双轨（v0.5 UI 升级期3，D4）=================
    def _on_mode_quick(self):
        self._set_param_mode(True)
    def _on_mode_expert(self):
        self._set_param_mode(False)
    def _set_param_mode(self, quick: bool, save: bool = True):
        """快速档 ⇄ 专家模式切换；专家模式即原完整折叠表单（零迁移成本）。"""
        self.quick_panel.setVisible(quick)
        self.expert_host.setVisible(not quick)
        self.btn_expand_all.setVisible(not quick)
        self.btn_mode_quick.setChecked(quick)
        self.btn_mode_expert.setChecked(not quick)
        if quick:
            self._quick_from_experts()
        else:
            self._experts_from_quick()
        if save:
            prefs = load_prefs()
            prefs["train_param_mode"] = "quick" if quick else "expert"
            save_prefs(prefs)
    def _quick_row(self, label: str, widget, unit: str = "",
                   badge_key: str = "", tip: str = "") -> QWidget:
        """Quick 档行化参数行：label 上（可带帮助图标）、控件下 + 单位。"""
        host = QWidget()
        v = QVBoxLayout(host)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        top = QHBoxLayout()
        top.setSpacing(6)
        lbl = QLabel(label)
        lbl.setStyleSheet(
            "font-weight: bold; border:none; background:transparent;")
        top.addWidget(lbl)
        if tip:
            info = QLabel("ⓘ")
            info.setToolTip(tip)
            bind_style(info, lambda: f"color:{_t3()}; border:none; background:transparent;")
            top.addWidget(info)
        top.addStretch(1)
        if badge_key:
            badge = Badge(tr("已推荐"), "primary")
            badge.hide()
            self._rec_badges[badge_key] = badge
            top.addWidget(badge)
        v.addLayout(top)
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(widget, stretch=1)
        if unit:
            u = QLabel(unit)
            bind_style(u, lambda: f"color:{_t2()}; border:none; background:transparent;")
            row.addWidget(u)
        v.addLayout(row)
        return host
    def _make_preset_card(self, key: str, name: str, desc: str,
                          values: dict, color: str) -> QFrame:
        """预设档位卡：点击一键套用并联动智能推荐。"""
        card = QFrame()
        card.setObjectName("card")
        card.setCursor(Qt.PointingHandCursor)
        card.setFixedSize(216, 92)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 10, 12, 8)
        lay.setSpacing(3)
        t = QLabel(name)
        t.setStyleSheet(f"font-size:16px; font-weight:bold; color:{color};"
                        "border:none; background:transparent;")
        d = QLabel(desc)
        d.setWordWrap(True)
        bind_style(d, lambda: f"font-size:15px; color:{_t2()};"
                              "border:none; background:transparent;")
        lay.addWidget(t)
        lay.addWidget(d, stretch=1)
        self._preset_values[key] = values
        self._preset_labels[key] = (t, d)
        card.mouseReleaseEvent = self._make_preset_click(key)
        return card
    def _make_preset_click(self, key: str):
        def _click(event):
            if event.button() == Qt.LeftButton:
                self._apply_preset(key)
        return _click
    def _apply_preset_cards(self, defs: list) -> None:
        """换预设卡文案与数值（轮次 ⇄ 免训练两套语义切换，V0.7.4 一.1）。

        文案经 tr() 现译：卡在窗口构建后可能被语言重建/localize 之外的路
        径换过文案，运行时换装必须自带翻译。"""
        for key, name, desc, values, color in defs:
            pair = self._preset_labels.get(key)
            if pair is None:
                continue
            t, d = pair
            t.setText(tr(name))
            t.setStyleSheet(f"font-size:16px; font-weight:bold; color:{color};"
                            "border:none; background:transparent;")
            d.setText(tr(desc))
            self._preset_values[key] = values
    def _fill_quick_controls(self):
        """把专家表单的模型/设备条目同步到 Quick 控件并回填当前值。"""
        self.q_model.blockSignals(True)
        self.q_model.clear()
        for i in range(self.par_model.count()):
            self.q_model.addItem(self.par_model.itemText(i),
                                 self.par_model.itemData(i))
        self.q_model.blockSignals(False)
        self.q_device.blockSignals(True)
        self.q_device.clear()
        for i in range(self.par_device.count()):
            self.q_device.addItem(self.par_device.itemText(i),
                                  self.par_device.itemData(i))
        self.q_device.blockSignals(False)
        self._quick_from_experts()
    def _quick_from_experts(self):
        """par_* → q_*（专家表单为唯一数据源）。"""
        self.q_model.blockSignals(True)
        idx = self.q_model.findData(self.par_model.currentData())
        if idx >= 0:
            self.q_model.setCurrentIndex(idx)
        self.q_model.blockSignals(False)
        self.q_epochs.setValue(self.par_epochs.value())
        self.q_batch.setValue(self.par_batch.value())
        self.q_lr.setValue(self.par_lr.value())
        self.q_kshot.setValue(self.par_kshot.value())
        self.q_augcount.setValue(self.par_augcount.value())
        dev = self.q_device.findData(self.par_device.currentData())
        if dev >= 0:
            self.q_device.setCurrentIndex(dev)
        self._sync_quick_size()
    def _experts_from_quick(self):
        """q_* → par_*（Quick 编辑后回写专家表单；lr 不被模型建议覆盖）。"""
        self.par_model.blockSignals(True)
        idx = self.par_model.findData(self.q_model.currentData())
        if idx >= 0:
            self.par_model.setCurrentIndex(idx)
        self.par_model.blockSignals(False)
        self.par_epochs.setValue(self.q_epochs.value())
        self.par_batch.setValue(self.q_batch.value())
        self.par_lr.setValue(self.q_lr.value())
        self.par_kshot.setValue(self.q_kshot.value())
        self.par_augcount.setValue(self.q_augcount.value())
        dev = self.par_device.findData(self.q_device.currentData())
        if dev >= 0:
            self.par_device.setCurrentIndex(dev)
    def _sync_quick_size(self):
        text = self.input_size_hint.text().strip()
        self.q_size.setText(text if text else "模型默认尺寸")
    def _apply_param_kv(self, key: str, value):
        """预设/推荐的参数键 → 专家表单控件（单一数据源仍为 par_*）。"""
        if key == "epochs":
            self.par_epochs.setValue(int(value))
        elif key == "batch":
            self.par_batch.setValue(int(value))
        elif key == "lr":
            self.par_lr.setValue(float(value))
        elif key == "freeze_backbone":
            self.par_freeze.setChecked(bool(value))
        elif key == "industrial_opt":
            self.par_industrial.setChecked(bool(value))
        elif key == "use_amp":
            self.par_amp.setChecked(bool(value))
        elif key == "tf32":
            self.par_tf32.setChecked(bool(value))
        elif key == "patience":
            self.par_patience.setValue(int(value))
        elif key == "k_shot":
            self.par_kshot.setValue(int(value))
        elif key == "aug_count":
            self.par_augcount.setValue(int(value))
    def _apply_preset(self, key: str):
        values = self._preset_values.get(key)
        if not values:
            return
        name = next((_n for _k, _n, _d, _v, _c
                     in _PRESET_DEFS + _PRESET_DEFS_FREE if _k == key), key)
        for k, v in values.items():
            self._apply_param_kv(k, v)
        rec_n = 0
        try:   # 联动智能推荐（数据集事实驱动；失败不影响套用本身）
            facts = dataset_facts(self._dataset_samples_for_guard())
            recs = recommend_settings(self.project, facts,
                                      _cuda_available())
            for rkey, rvalue, _reason in recs:
                self._apply_recommend_key(rkey, rvalue)
                rec_n += 1
        except Exception:
            log.warning("智能推荐联动失败（预设已套用）", exc_info=True)
        self._quick_from_experts()   # 预设已写 par_* → 同步到 Quick 控件
        quick_keys = ({"k_shot", "aug_count"} if self._train_mode == "free"
                      else {"epochs", "batch", "lr"}) & set(values)
        self._mark_recommended(sorted(quick_keys))
        self._save_params()
        msg = f"{tr('已套用预设')}「{tr(name)}」"
        if rec_n:
            msg += f"，并联动智能推荐 {rec_n} 项"
        toast(self, msg, "success")
    def _reset_quick_defaults(self):
        d = getattr(self, "_param_defaults", None)
        if not d:
            return
        self.par_model.blockSignals(True)
        idx = self.par_model.findData(d.get("model"))
        if idx >= 0:
            self.par_model.setCurrentIndex(idx)
        self.par_model.blockSignals(False)
        self.par_epochs.setValue(d.get("epochs", self.par_epochs.value()))
        self.par_batch.setValue(d.get("batch", self.par_batch.value()))
        self.par_lr.setValue(d.get("lr", self.par_lr.value()))
        self.par_kshot.setValue(d.get("k_shot", self.par_kshot.value()))
        self.par_augcount.setValue(d.get("aug_count", self.par_augcount.value()))
        dev = self.par_device.findData(d.get("device"))
        if dev >= 0:
            self.par_device.setCurrentIndex(dev)
        self._refresh_input_size_hint()
        self._quick_from_experts()   # 默认值已写 par_* → 同步到 Quick 控件
        self._clear_recommended()
        self._save_params()
        toast(self, "已恢复默认参数", "info")
    def _mark_recommended(self, keys):
        for k in keys:
            b = self._rec_badges.get(k)
            if b is not None:
                b.show()
    def _clear_recommended(self):
        for b in self._rec_badges.values():
            b.hide()
    def _toggle_all_sections(self):
        """全部展开 ⇄ 全部折叠（再次点击全部折叠）。
        整批挂起重绘，避免连续三个分区逐个回流造成卡顿。"""
        sections = (self.sec_basic, self.sec_data, self.sec_adv)
        target = any(s.content.isHidden() for s in sections)
        win = self.window()
        win.setUpdatesEnabled(False)
        try:
            for sec in sections:
                sec.set_expanded(target)
        finally:
            win.setUpdatesEnabled(True)
        self.btn_expand_all.setText("全部折叠" if target else "全部展开")


# ---- Cython 编译态槽包装：mixin 方法经继承进入窗口类，编译态同样需要
# 参数裁剪包装（见 xccv/qt_slots.py；幂等）----
from xccv.qt_slots import apply_auto_slot_class  # noqa: E402

apply_auto_slot_class(QuickPanelMixin)

