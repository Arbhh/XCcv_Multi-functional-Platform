"""O2（V0.7）+ V0.7.1 用户勘误：批量分数/维度分布 + 阈值面板。

批量完成后，把 GT 匹配统计（engine.metrics.batch_match 的 match_stats）
画成"带平滑线的散点图"（用户 V0.7.1 设定）：正常 / 过检 / 漏检三条不同
颜色的曲线。拖动阈值线或在输入框键入数值，即零推理成本实时刷新该阈值
下的精确率 / 召回率 / OK 误检。

V0.7.1 坐标轴约定（用户规定，固定不随拖动变化）：
- y 轴 = 检出数，从 0 起，上限随本批数据的最大分箱计数自适应外扩；
- x 轴按维度：置信度固定 0~1（两位小数）；面积/短边/长边从 0 到本批
  最大值外扩一部分（整数刻度）；灰度差从 0 到最大值外扩（两位小数）；
- 散点过多时自动分箱（如面积按区间聚合）后连线，轴上限取整到箱宽整数倍；
- 三条曲线：正常=命中 GT 的检出（TP）；过检=误检（含 OK 图误检）；
  漏检=漏检 GT 框自身的几何维度分布（仅面积/短边/长边可画——分数与
  灰度类维度 GT 侧无值）。

多维度阈值（O2 扩展）：每维度独立记忆阈值与"参与过滤"开关；拖线或输入
自动启用该维度；"生成流程图模板"把启用维度编排成 IR 流程图
（threshold_flow）落在 <模型目录>/flow/；"应用到检出流程"维持只写 conf 维。
前景占比维度按用户 V0.7.1 指示从面板移除（流程图节点保留不受影响）。

纯展示组件：数据经 set_stats 注入；applyRequested/generateFlowRequested
由宿主窗口接线落盘。UI 文案遵循项目总体约定：简短精确、不用括号补充，
详细说明放 tooltip 与操作手册。
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
                             QGridLayout, QHBoxLayout, QLabel, QPushButton,
                             QVBoxLayout, QWidget)

from ..engine.metrics.batch_match import threshold_metrics, threshold_metrics_for
from ..ui_style import token

_GREEN = "#2E9E5B"   # 正常（TP，命中 GT）
_RED = "#D64545"     # 过检（FP，含 OK 图误检）
_BLUE = "#25406B"    # 漏检（漏检 GT 框几何维度）

# 维度规格：key → (显示名, 小数位, 步长, 无数据时的 x 轴上限)
# score 走 0~1 固定通道；面积/短边/长边整数、灰度差两位小数（用户口径）
_DIM_SPECS = {
    "score": ("置信度", 2, 0.05, 1.0),
    "area": ("面积", 0, 10.0, 1000.0),
    "short": ("短边", 0, 5.0, 200.0),
    "long": ("长边", 0, 5.0, 400.0),
    "gray_diff": ("灰度差", 2, 1.0, 100.0),
}
_DIM_ORDER = ("score", "area", "short", "long", "gray_diff")
_DIM_DEFAULTS = {"area": 0.0, "short": 0.0, "long": 0.0, "gray_diff": 15.0}
# 漏检曲线仅几何维度可画（GT 框没有分数/前景/灰度值）
_MISSABLE = ("area", "short", "long")

_TICK_TARGET = 6          # x 轴主刻度目标个数
_Y_PAD = 1.15             # y 轴上限外扩系数
_X_PAD = 1.10             # 非 score 维 x 轴上限外扩系数
_N_BINS = 20              # 非 score 维目标分箱数


def _dim_label(key: str) -> str:
    return _DIM_SPECS.get(key, (key,))[0]


def _nice_ceil(v: float) -> float:
    """向上取到 1/2/2.5/5×10^k 的"好看"刻度（坐标轴上限与箱宽外扩用）。"""
    if v <= 0:
        return 0.0
    exp = 10.0 ** np.floor(np.log10(v))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if v <= m * exp:
            return float(m * exp)
    return 10.0 * float(exp)


def _axis_ticks(x_max: float, decimals: int) -> list:
    """固定 x 轴主刻度 [(值, 文本)...]，格式随维度小数位（用户口径）。"""
    step = _nice_ceil(x_max / _TICK_TARGET) if x_max > 0 else 1.0
    vals = np.arange(0.0, x_max + step * 0.5, step)
    out = [(round(float(v), decimals), f"{v:.{decimals}f}")
           for v in vals if v <= x_max + 1e-9]
    return [out]   # pyqtgraph setTicks 的层级结构


def _smooth(counts: np.ndarray) -> np.ndarray:
    """分箱计数轻平滑（[1,2,1] 卷积两轮）——平滑折线不丢峰位。"""
    k = np.array([1.0, 2.0, 1.0])
    k /= k.sum()
    out = counts.astype(float)
    for _ in range(2):
        pad = np.concatenate(([out[0]], out, [out[-1]]))
        out = np.convolve(pad, k, mode="valid")
    return out


class ThresholdPanel(QWidget):
    """分数/维度分布平滑散点线图 + 可拖阈值线 + 阈值输入框 + 实时指标。"""

    thresholdChanged = pyqtSignal(float)
    applyRequested = pyqtSignal(float)
    generateFlowRequested = pyqtSignal(dict)
    # 项目复盘页（用户勘误）：任一维度阈值/启用状态变化 → {dim: thr}（启用集）
    thresholdsChanged = pyqtSignal(dict)

    def __init__(self, parent=None, compact: bool = False):
        """compact=True：上下布局（图整幅在上、指标/按钮在下）。

        项目复盘页左列较窄，横排布局会把图挤成窄条（再勘误 ②）；
        测试平台旧内嵌处维持 compact=False 的横排原样。
        """
        super().__init__(parent)
        self._stats: dict = {}
        self._current = "score"
        self._x_max = 1.0     # 当前维度固定 x 轴上限（V0.7.1 坐标轴约定）
        # 每维度独立状态：{key: {"thr": float, "enabled": bool}}
        self._dim_state: dict = {"score": {"thr": 0.25, "enabled": True}}
        lay = (QVBoxLayout(self) if compact else QHBoxLayout(self))
        lay.setContentsMargins(0, 0, 0, 0)

        self.plot = pg.PlotWidget()
        self.plot.setBackground(None)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("left", "检出数")
        self.plot.addLegend(offset=(8, 6))
        self._line = pg.InfiniteLine(
            pos=0.25, angle=90, movable=True,
            pen=pg.mkPen(token("primary"), width=2, style=Qt.DashLine))
        self.plot.addItem(self._line)
        self._line.sigPositionChanged.connect(self._on_line_moved)
        if compact:
            self.plot.setMinimumHeight(190)
            self.plot.setMaximumHeight(230)
            lay.addWidget(self.plot, stretch=1)
            right = QGridLayout()
            right.setHorizontalSpacing(10)
            right.setVerticalSpacing(6)
        else:
            lay.addWidget(self.plot, stretch=3)
            right = QFormLayout()

        self.cmb_dim = QComboBox()
        for key in _DIM_ORDER:
            self.cmb_dim.addItem(_dim_label(key), key)
        self.cmb_dim.currentIndexChanged.connect(self._on_dim_changed)
        # 阈值数字输入框（V0.7.1 用户追加：拖线之外可直接键入）
        self.spin_thr = QDoubleSpinBox()
        self.spin_thr.setRange(0.0, 1.0)
        self.spin_thr.setDecimals(2)
        self.spin_thr.setValue(0.25)
        self.spin_thr.setKeyboardTracking(False)
        self.spin_thr.valueChanged.connect(self._on_spin_changed)
        self.cb_enable = QCheckBox("参与过滤")
        self.cb_enable.stateChanged.connect(self._on_enable_toggled)
        self.lbl_precision = QLabel("—")
        self.lbl_recall = QLabel("—")
        self.lbl_ok = QLabel("—")
        if compact:
            # 行1：维度 + 阈值输入 + 参与过滤；行2：三指标；行3：两按钮
            right.addWidget(QLabel("维度"), 0, 0)
            right.addWidget(self.cmb_dim, 0, 1)
            right.addWidget(QLabel("阈值"), 0, 2)
            right.addWidget(self.spin_thr, 0, 3)
            right.addWidget(self.cb_enable, 0, 4, 1, 2)
            right.addWidget(QLabel("精确率"), 1, 0)
            right.addWidget(self.lbl_precision, 1, 1)
            right.addWidget(QLabel("召回率"), 1, 2)
            right.addWidget(self.lbl_recall, 1, 3)
            right.addWidget(QLabel("OK误检"), 1, 4)
            right.addWidget(self.lbl_ok, 1, 5)
            right.setColumnStretch(6, 1)
        else:
            for row, w in (("维度", self.cmb_dim),
                           ("阈值", self.spin_thr),
                           ("", self.cb_enable),
                           ("精确率", self.lbl_precision),
                           ("召回率", self.lbl_recall),
                           ("OK误检", self.lbl_ok)):
                if row:
                    right.addRow(row, w)
                else:
                    right.addRow(w)
        lay.addLayout(right)

        self.lbl_hint = QLabel("")
        self.lbl_hint.setWordWrap(True)
        lay.addWidget(self.lbl_hint)

        btn_row = QHBoxLayout()
        self.btn_apply = QPushButton("应用到检出流程")
        self.btn_apply.setToolTip(
            "写入该项目「检出流程.json」的置信度过滤节点（含变更留痕备注）；\n"
            "测试平台与部署服务此后按该阈值过滤检出")
        self.btn_apply.clicked.connect(self._emit_apply_conf)
        btn_row.addWidget(self.btn_apply)
        self.btn_flow = QPushButton("生成流程图模板")
        self.btn_flow.setToolTip(
            "在模型存放文件夹下新建 flow/ 子文件夹，按当前启用的维度阈值"
            "生成流程图模板：<时间>_<模型名>.json（模型路径已填当前权重，"
            "生成即可在流程图平台打开运行）")
        self.btn_flow.clicked.connect(self._emit_generate_flow)
        btn_row.addWidget(self.btn_flow)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)

        # 上线提示：阈值仅对本批数据有效，换产线需复验
        self.lbl_warn = QLabel(
            "⚠ 阈值仅对本批数据有效，上线前请用新产线图复验")
        self.lbl_warn.setWordWrap(True)
        self.lbl_warn.setStyleSheet(f"color: {token('warn')};")
        lay.addWidget(self.lbl_warn)

    # ---- 数据注入与刷新 ----
    def set_stats(self, stats: dict, conf: float):
        """注入 match_stats；按维度可用性重建下拉（无数据维度不出现）。"""
        self._stats = stats or {}
        self._dim_state = {"score": {"thr": float(conf), "enabled": True}}
        dims = self._stats.get("dims") or {}
        for key in _DIM_ORDER[1:]:
            d = dims.get(key) or {}
            has = any(d.get(b) for b in ("green", "red_defect", "red_ok"))
            if has:
                self._dim_state[key] = {
                    "thr": float(_DIM_DEFAULTS.get(key, 0.0)), "enabled": False}
        combo = self.cmb_dim
        combo.blockSignals(True)
        combo.clear()
        for key in _DIM_ORDER:
            if key == "score" or key in self._dim_state:
                combo.addItem(_dim_label(key), key)
        combo.blockSignals(False)
        self._current = "score"
        combo.setCurrentIndex(0)
        self._sync_dim_widgets()
        self._redraw()
        if not self._stats.get("gt_available"):
            self.lbl_hint.setText(
                "批量目录未找到同名标注 JSON：分布与统计需要真值标注，"
                "请补标注或改用数据集目录批量。")
            self.btn_apply.setEnabled(True)
            self.btn_flow.setEnabled(True)
            return
        self.lbl_hint.setText("")
        self.btn_apply.setEnabled(True)
        self.btn_flow.setEnabled(True)

    def current_threshold(self) -> float:
        st = self._dim_state.get(self._current)
        return float(st["thr"]) if st else 0.25

    def enabled_dims(self) -> dict:
        """启用且阈值>0 的维度 → {dim: thr}（生成流程图模板用）。"""
        return {k: float(v["thr"]) for k, v in self._dim_state.items()
                if v.get("enabled") and float(v["thr"]) > 0}

    def _emit_thresholds_changed(self):
        self.thresholdsChanged.emit(self.enabled_dims())

    # ---- 交互 ----
    def _emit_apply_conf(self):
        # 语义锁定 conf 维（其他维度走"生成流程图模板"）
        self.applyRequested.emit(float(self._dim_state["score"]["thr"]))

    def _emit_generate_flow(self):
        dims = self.enabled_dims()
        if not dims:
            self.lbl_hint.setText("请先拖动阈值线或键入阈值启用至少一个维度")
            return
        self.generateFlowRequested.emit(dims)

    def _series_values(self, key: str) -> list:
        """当前维度的曲线数据 → [(名称, 颜色, 值列表)]。

        score 维取顶层分数表；其余维度取 dims 表；过检合并缺陷图误检与
        OK 图误检两条来源；漏检线只给几何维度（dims_missed）。
        """
        if key == "score":
            green = list(self._stats.get("green") or [])
            over = (list(self._stats.get("red_defect") or [])
                    + list(self._stats.get("red_ok") or []))
        else:
            d = (self._stats.get("dims") or {}).get(key) or {}
            green = list(d.get("green") or [])
            over = (list(d.get("red_defect") or [])
                    + list(d.get("red_ok") or []))
        series = [("正常", _GREEN, green), ("过检", _RED, over)]
        if key in _MISSABLE:
            missed = (self._stats.get("dims_missed") or {}).get(key) or []
            if missed:
                series.append(("漏检", _BLUE, missed))
        return series

    def _plan_axis(self, key: str) -> tuple:
        """固定轴规划 → (x 上限, 分箱边界)。score 恒 [0,1] 按 0.05 分箱；
        其余维度约 20 箱、箱宽取整档，轴上限=箱宽×箱数（边界不越轴）。"""
        min_w = 1.0 if _DIM_SPECS[key][1] == 0 else 0.01
        if key == "score":
            return 1.0, np.linspace(0.0, 1.0, 21)
        vals = [v for _n, _c, vs in self._series_values(key) for v in vs]
        if not vals:
            x_max = float(_DIM_SPECS[key][3])
            return x_max, np.linspace(0.0, x_max, _N_BINS + 1)
        raw = max(vals) * _X_PAD
        w = max(_nice_ceil(raw / _N_BINS), min_w)
        n = max(2, int(np.ceil(raw / w)))
        x_max = w * n
        bins = np.linspace(0.0, x_max, n + 1)
        return x_max, bins

    def _on_dim_changed(self, idx: int):
        key = self.cmb_dim.itemData(idx)
        if not key or key == self._current:
            return
        self._current = str(key)
        self._sync_dim_widgets()
        self._redraw()

    def _sync_dim_widgets(self):
        """维度切换后同步输入框（范围/小数位/步长/值）与参与过滤勾选。"""
        key = self._current
        st = self._dim_state.get(key)
        self._x_max, _bins = self._plan_axis(key)
        _name, dec, step, _d = _DIM_SPECS[key]
        spin = self.spin_thr
        spin.blockSignals(True)
        spin.setDecimals(dec)
        spin.setSingleStep(step)
        spin.setRange(0.0, self._x_max)
        spin.setValue(float(st["thr"]) if st else 0.0)
        spin.blockSignals(False)
        self._line.blockSignals(True)
        self._line.setValue(float(st["thr"]) if st else 0.0)
        self._line.blockSignals(False)
        self.cb_enable.blockSignals(True)
        self.cb_enable.setChecked(bool(st and st.get("enabled")))
        self.cb_enable.blockSignals(False)
        self.plot.setLabel("bottom", _dim_label(key))

    def _on_enable_toggled(self, state: int):
        st = self._dim_state.get(self._current)
        if st is not None:
            st["enabled"] = bool(state)
        self._refresh_numbers()
        self._emit_thresholds_changed()

    def _apply_threshold(self, v: float):
        """阈值统一入口（拖线/键入共用）：钳回固定轴范围 → 取整到维度
        小数位 → 回写线与输入框 → 启用当前维 → 刷新指标并广播。"""
        # 拖拽/键盘可能把线带出固定轴范围
        # → 钳回 [0, x_max]（score 恒 [0,1]）后再入状态
        v = min(max(float(v), 0.0), self._x_max)
        _name, dec, _step, _d = _DIM_SPECS[self._current]
        v = round(v, dec)
        self._line.blockSignals(True)
        self._line.setValue(v)
        self._line.blockSignals(False)
        self.spin_thr.blockSignals(True)
        self.spin_thr.setValue(v)
        self.spin_thr.blockSignals(False)
        st = self._dim_state.setdefault(
            self._current, {"thr": 0.0, "enabled": False})
        st["thr"] = v
        if not st.get("enabled"):     # 拖线/输入即启用（一键成图约定）
            st["enabled"] = True
            self.cb_enable.blockSignals(True)
            self.cb_enable.setChecked(True)
            self.cb_enable.blockSignals(False)
        self._refresh_numbers()
        self.thresholdChanged.emit(self.current_threshold())
        self._emit_thresholds_changed()

    def _on_line_moved(self):
        self._apply_threshold(float(self._line.value()))

    def _on_spin_changed(self, v: float):
        self._apply_threshold(float(v))

    # ---- 绘制 ----
    def _redraw(self):
        self.plot.clear()
        leg = self.plot.getPlotItem().legend
        if leg is not None:
            leg.clear()   # pg 图例不随 plot.clear() 清空，重绘前手动清
        self.plot.addItem(self._line)
        stats = self._stats
        key = self._current
        if not stats.get("gt_available"):
            self._x_max, _ = self._plan_axis(key)
            self._apply_fixed_axes(key)
            self.plot.setYRange(0, 4.0, padding=0)
            self._refresh_numbers()
            return
        series = self._series_values(key)
        self._x_max, bins = self._plan_axis(key)
        self._apply_fixed_axes(key)
        centers = (bins[:-1] + bins[1:]) / 2
        clip = (lambda v: np.clip(np.asarray(v, dtype=float), 0.0, 1.0)
                if key == "score" else np.asarray(v, dtype=float))
        top = 1
        for name, color, values in series:
            if not values:
                continue
            counts, _ = np.histogram(clip(values), bins=bins)
            if not counts.any():
                continue
            top = max(top, int(counts.max()))
            self.plot.plot(
                centers, _smooth(counts), pen=pg.mkPen(color, width=2),
                symbol="o", symbolSize=6, symbolBrush=pg.mkBrush(color),
                symbolPen=pg.mkPen(None), name=name, antialias=True)
        y_max = max(_nice_ceil(top * _Y_PAD), 4)
        self.plot.setYRange(0, float(y_max), padding=0)
        self._refresh_numbers()

    def _apply_fixed_axes(self, key: str):
        """固定坐标轴（V0.7.1 约定）：轴不随拖动/重绘变化，只有阈值线可动。"""
        x_max = self._x_max
        dec = _DIM_SPECS[key][1]
        self.plot.setXRange(0.0, x_max, padding=0)
        self.plot.getAxis("bottom").setTicks(_axis_ticks(x_max, dec))
        self.plot.getAxis("bottom").setWidth(56)
        self.plot.getAxis("left").setWidth(42)

    def _refresh_numbers(self):
        key = self._current
        t = self.current_threshold()
        if key == "score":
            m = threshold_metrics(self._stats, t)
        else:
            m = threshold_metrics_for(self._stats, key, t)
        self.lbl_precision.setText(
            f"{m['precision']:.3f}" if m["precision"] is not None else "—")
        self.lbl_recall.setText(
            f"{m['recall']:.3f}" if m["recall"] is not None else "—")
        self.lbl_ok.setText(
            f"{m['ok_fp_per_img']:.2f}" if m["ok_fp_per_img"] is not None else "—")
