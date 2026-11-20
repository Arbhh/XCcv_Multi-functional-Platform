"""维度阈值扫描曲线面板（0.7.2 重设计）：维度阈值扫描曲线面板。

曲线含义（V0.7.2 实测修复定稿）——**图口径**：每条曲线给出阈值=x 时
对应**分组的图数**，与右侧分组/统计行完全同口径：
- 正确：该阈值下全命中且无误检的图数；
- 过检：存在误检（含 OK 图误检）的图数；
- 漏检：无过检且有 GT 未命中的图数；
恒有 正确+过检+漏检 = 已评估图数。未评估范围（无 GT 生产图）：
OK/NG 两组图数（NG=有检出通过阈值的图数）。
拖动阈值线或键入数值，三条曲线上的读数即该阈值处的实际分组数量，
与右侧分组同口径（其他已启用维度按各自阈值联合过滤）。
「精确率/召回率/OK误检」数字行维持检出/GT 实例口径（与训练评估同源）。

坐标轴约定（V0.7.1，固定不随拖动变化）：
- y 轴 = 数量（图数），从 0 起，上限随曲线峰值外扩；
- x 轴按维度：置信度固定 0~1（两位小数）；面积/短边/长边从 0 到本批
  最大值外扩一部分（整数刻度）；灰度差从 0 到最大值外扩（两位小数）；
- 轴规划取全量数据集（已评估+未评估一起），已评估/未评估切换不重排轴；
- 只有阈值线可拖。已评估/未评估切换由宿主页控制（set_scope）。

多维度阈值：每维度独立记忆阈值与"参与过滤"开关；拖线或输入自动启用
该维度；"生成流程图模板"把启用维度编排成 IR 流程图（threshold_flow）
落在 <模型目录>/flow/；"应用到检出流程"维持只写 conf 维。

纯展示组件：数据经 set_stats 注入；applyRequested/generateFlowRequested
由宿主窗口接线落盘。UI 文案遵循项目总体约定：简短精确、不用括号补充，
详细说明放 tooltip 与操作手册。
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
                             QGridLayout, QHBoxLayout, QLabel, QPushButton,
                             QVBoxLayout, QWidget)

from ..engine.metrics.batch_match import sweep_metrics, sweep_unjudged
from ..ui_style import token

_GREEN = "#2E9E5B"   # 正常 / OK
_RED = "#D64545"     # 过检 / NG
_BLUE = "#25406B"    # 漏检

# 维度规格：key → (显示名, 小数位, 步长, 无数据时的 x 轴上限)
# score 走 0~1 固定通道；面积/短边/长边整数、灰度差两位小数（既定口径）
_DIM_SPECS = {
    "score": ("置信度", 2, 0.05, 1.0),
    "area": ("面积", 0, 10.0, 1000.0),
    "short": ("短边", 0, 5.0, 200.0),
    "long": ("长边", 0, 5.0, 400.0),
    "gray_diff": ("灰度差", 2, 1.0, 100.0),
}
_DIM_ORDER = ("score", "area", "short", "long", "gray_diff")
_DIM_DEFAULTS = {"area": 0.0, "short": 0.0, "long": 0.0, "gray_diff": 15.0}

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
    """固定 x 轴主刻度 [(值, 文本)...]，格式随维度小数位（既定口径）。"""
    step = _nice_ceil(x_max / _TICK_TARGET) if x_max > 0 else 1.0
    vals = np.arange(0.0, x_max + step * 0.5, step)
    out = [(round(float(v), decimals), f"{v:.{decimals}f}")
           for v in vals if v <= x_max + 1e-9]
    return [out]   # pyqtgraph setTicks 的层级结构


class ThresholdPanel(QWidget):
    """维度阈值扫描曲线 + 可拖阈值线 + 阈值输入框 + 实时指标。

    曲线读数 = 阈值处 正常/过检/漏检（已评估）或 OK/NG（未评估）总数。
    """

    thresholdChanged = pyqtSignal(float)
    applyRequested = pyqtSignal(float)
    generateFlowRequested = pyqtSignal(dict)
    # 项目复盘页：任一维度阈值/启用状态变化 → {dim: thr}（启用集）
    thresholdsChanged = pyqtSignal(dict)

    def __init__(self, parent=None, compact: bool = False):
        """compact=True：上下布局（图整幅在上、指标/按钮在下）。

        项目复盘页左列较窄，横排布局会把图挤成窄条；
        测试平台旧内嵌处维持 compact=False 的横排原样。
        """
        super().__init__(parent)
        self._stats: dict = {}
        self._current = "score"
        self._x_max = 1.0     # 当前维度固定 x 轴上限（V0.7.1 坐标轴约定）
        self._scope = "evaluated"   # evaluated=正常/过检/漏检；unjudged=OK/NG
        # 每维度独立状态：{key: {"thr": float, "enabled": bool}}
        self._dim_state: dict = {"score": {"thr": 0.25, "enabled": True}}
        lay = (QVBoxLayout(self) if compact else QHBoxLayout(self))
        lay.setContentsMargins(0, 0, 0, 0)

        self.plot = pg.PlotWidget()
        self.plot.setBackground(None)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        from ..i18n import tr

        self.plot.setLabel("left", tr("数量"))
        self.plot.setToolTip(tr(
            "曲线=各分组图数随阈值变化（与右侧分组同口径）；"
            "精确率/召回率=检出与GT实例口径"))
        self.plot.addLegend(offset=(8, 6))
        self._line = pg.InfiniteLine(
            pos=0.25, angle=90, movable=True,
            pen=pg.mkPen(token("primary"), width=2, style=Qt.DashLine),
            # V0.7.4 二.1：拖动钳回固定轴（线滑出轴会被视图裁剪成"看不见
            # 抓不到"；实测现象"滑到最右端后滑不回来"的根因）；hoverPen 加宽
            # 同时扩大抓取判定区（适度）与悬停反馈
            bounds=(0.0, 1.0),
            hoverPen=pg.mkPen(token("primary"), width=10, style=Qt.DashLine))
        self.plot.addItem(self._line)
        self._line.sigPositionChanged.connect(self._on_line_moved)
        self._line.sigPositionChangeFinished.connect(self._on_drag_finished)
        self._thr_emit_timer = QTimer(self)
        self._thr_emit_timer.setSingleShot(True)
        self._thr_emit_timer.setInterval(150)
        self._thr_emit_timer.timeout.connect(self._on_drag_finished)
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
        from ..i18n import tr as _tr

        for key in _DIM_ORDER:
            self.cmb_dim.addItem(_tr(_dim_label(key)), key)
        self.cmb_dim.currentIndexChanged.connect(self._on_dim_changed)
        # 阈值数字输入框（V0.7.1 追加：拖线之外可直接键入）
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
            self.lbl_p_name = QLabel("精确率")
            self.lbl_r_name = QLabel("召回率")
            self.lbl_o_name = QLabel("OK误检")
            right.addWidget(QLabel("维度"), 0, 0)
            right.addWidget(self.cmb_dim, 0, 1)
            right.addWidget(QLabel("阈值"), 0, 2)
            right.addWidget(self.spin_thr, 0, 3)
            right.addWidget(self.cb_enable, 0, 4, 1, 2)
            right.addWidget(self.lbl_p_name, 1, 0)
            right.addWidget(self.lbl_precision, 1, 1)
            right.addWidget(self.lbl_r_name, 1, 2)
            right.addWidget(self.lbl_recall, 1, 3)
            right.addWidget(self.lbl_o_name, 1, 4)
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
            self.lbl_p_name = self.lbl_r_name = self.lbl_o_name = None
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

        # 上线检查单（产线过漏杀五因；详细说明见操作手册）
        self.lbl_warn = QLabel(
            "⚠ 阈值仅对本批数据有效，上线前请用新产线图复验")
        self.lbl_warn.setWordWrap(True)
        self.lbl_warn.setStyleSheet(f"color: {token('warn')};")
        lay.addWidget(self.lbl_warn)

    # ---- 分数语义（§三，V0.7.3）：conf/prob 固定 0~1 轴；z 无上界动态轴 ----
    def _score_unbounded(self) -> bool:
        rng = (self._stats.get("score_semantics") or {}).get("range") or [0.0, 1.0]
        return len(rng) > 1 and rng[1] is None

    def _score_label(self) -> str:
        from ..i18n import tr

        sem = self._stats.get("score_semantics") or {}
        return str(tr(sem.get("name") or _DIM_SPECS["score"][0]))

    def _score_spec(self) -> tuple:
        """score 维规格 (显示名, 小数位, 步长, 默认轴上限)，随语义切换。"""
        if self._score_unbounded():
            return (self._score_label(), 2, 0.5, 10.0)
        return (_DIM_SPECS["score"][0],) + _DIM_SPECS["score"][1:]

    def _initial_score_thr(self, conf: float) -> float:
        """score 维初始阈值：z 语义按 conf→z=1/conf−1 换算（与预测同式）。"""
        if not self._score_unbounded():
            return float(conf)
        return 1.0 / min(max(float(conf), 1e-3), 1.0) - 1.0

    # ---- 数据注入与刷新 ----
    def set_stats(self, stats: dict, conf: float):
        """注入 match_stats；按维度可用性重建下拉（无数据维度不出现）。"""
        self._stats = stats or {}
        self._dim_state = {"score": {"thr": self._initial_score_thr(conf),
                                     "enabled": True}}
        for key in _DIM_ORDER[1:]:
            d = (self._stats.get("dims") or {}).get(key) or {}
            has_dims = any(d.get(b) for b in ("green", "red_defect", "red_ok"))
            if has_dims or self._axis_values(key):
                self._dim_state[key] = {
                    "thr": float(_DIM_DEFAULTS.get(key, 0.0)), "enabled": False}
        combo = self.cmb_dim
        from ..i18n import tr

        combo.blockSignals(True)
        combo.clear()
        for key in _DIM_ORDER:
            if key == "score" or key in self._dim_state:
                combo.addItem(tr(self._score_label() if key == "score"
                                 else _dim_label(key)), key)
        combo.blockSignals(False)
        self._current = "score"
        combo.setCurrentIndex(0)
        self._sync_dim_widgets()
        self._redraw()
        from ..i18n import tr

        if not self._stats.get("gt_available"):
            self.lbl_hint.setText(tr(
                "批量目录未找到同名标注 JSON：分布与统计需要真值标注，"
                "请补标注或改用数据集目录批量。"))
            self.btn_apply.setEnabled(True)
            self.btn_flow.setEnabled(True)
            return
        self.lbl_hint.setText("")
        self.btn_apply.setEnabled(True)
        self.btn_flow.setEnabled(True)

    def set_scope(self, scope: str):
        """评估范围切换（V0.7.2 任务③）：evaluated=正常/过检/漏检；
        unjudged=未评估生产图 OK/NG。轴按数据集不随范围变化，
        切换即重绘曲线并同步输入框。"""
        scope = "unjudged" if scope == "unjudged" else "evaluated"
        if scope == self._scope:
            return
        self._scope = scope
        self._sync_dim_widgets()
        self._redraw()

    def current_threshold(self) -> float:
        st = self._dim_state.get(self._current)
        return float(st["thr"]) if st else 0.25

    def enabled_dims(self) -> dict:
        """启用且阈值>0 的维度 → {dim: thr}（生成流程图模板用）。"""
        return {k: float(v["thr"]) for k, v in self._dim_state.items()
                if v.get("enabled") and float(v["thr"]) > 0}

    def _emit_thresholds_changed(self, debounce: bool = False):
        """广播启用维度阈值（V0.7.4 二.1：真实拖线期间防抖 150ms——
        sigPositionChanged 每像素一发的重分组会把复盘页拖垮；程序化
        setValue 与拖线结束立即广播）。"""
        if debounce:
            self._thr_emit_timer.start()
        else:
            self._thr_emit_timer.stop()
            self.thresholdsChanged.emit(self.enabled_dims())

    def _on_drag_finished(self, *_):
        self._emit_thresholds_changed()

    # ---- 交互 ----
    def _emit_apply_conf(self):
        # 语义锁定 conf 维（其他维度走"生成流程图模板"）
        self.applyRequested.emit(float(self._dim_state["score"]["thr"]))

    def _emit_generate_flow(self):
        dims = self.enabled_dims()
        if not dims:
            from ..i18n import tr

            self.lbl_hint.setText(
                tr("请先拖动阈值线或键入阈值启用至少一个维度"))
            return
        self.generateFlowRequested.emit(dims)

    # ---- 数据通道 ----
    def _axis_values(self, key: str) -> list:
        """本批数据该维度全部检出值（x 轴规划用，含未过阈值的）。

        V0.7.2 实测修复：坐标轴按**数据集**规划（V0.7.1 约定"轴只随
        数据集变化"），已评估/未评估切换不重排轴——否则阈值线会被切出
        视图、输入框范围失真。已评估/未评估只影响曲线口径与计数。
        per_image 无逐检出记录时回退 dims 汇总表（旧缓存/轻量 stats 兼容）。"""
        from ..engine.metrics.batch_match import DET_COL

        col = DET_COL.get(key)
        if col is None:
            return []
        vals = []
        for img in (self._stats.get("per_image") or []):
            for rec in img.get("dets") or []:
                v = rec[col]
                if v is not None:
                    vals.append(float(v))
        if vals:
            return vals
        d = (self._stats.get("dims") or {}).get(key) or {}
        out = []
        for b in ("green", "red_defect", "red_ok"):
            out.extend(float(v) for v in (d.get(b) or []))
        return out

    def _sweep_fixed(self) -> dict:
        """其他已启用维度的阈值（联合过滤口径，与复盘分组同源）。"""
        return {k: float(v["thr"]) for k, v in self._dim_state.items()
                if k != self._current and v.get("enabled")
                and float(v["thr"]) > 0}

    def _plan_axis(self, key: str) -> tuple:
        """固定轴规划 → (x 上限, 扫描点)。score 的 conf/prob 语义恒 [0,1]
        按 0.05；z 语义（§三）与其余维度一样按数据动态规划（约 20 档、
        档宽取整，轴上限=档宽×档数，边界不越轴；无数据退默认上限）。"""
        if key == "score":
            _name, _dec, _step, default_max = self._score_spec()
            if not self._score_unbounded():
                return 1.0, np.linspace(0.0, 1.0, 21)
            vals = self._axis_values(key)
            if not vals:
                return float(default_max), np.linspace(
                    0.0, float(default_max), _N_BINS + 1)
            raw = max(vals) * _X_PAD
            w = max(_nice_ceil(raw / _N_BINS), 0.5)
            n = max(2, int(np.ceil(raw / w)))
            x_max = w * n
            return x_max, np.linspace(0.0, x_max, n + 1)
        min_w = 1.0 if _DIM_SPECS[key][1] == 0 else 0.01
        vals = self._axis_values(key)
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
        from ..i18n import tr

        key = self._current
        st = self._dim_state.get(key)
        self._x_max, _bins = self._plan_axis(key)
        # V0.7.4 二.1：拖动范围随轴上限更新（原生钳制，线永不滑出轴）
        self._line.setBounds((0.0, float(self._x_max)))
        if key == "score":
            _name, dec, step, _d = self._score_spec()
        else:
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
        self.plot.setLabel(
            "bottom", tr(self._score_label() if key == "score"
                         else _dim_label(key)))

    def _on_enable_toggled(self, state: int):
        st = self._dim_state.get(self._current)
        if st is not None:
            st["enabled"] = bool(state)
        self._refresh_numbers()
        self._emit_thresholds_changed()

    def _apply_threshold(self, v: float, debounce: bool = False):
        """阈值统一入口（拖线/键入共用）：钳回固定轴范围 → 取整到维度
        小数位 → 回写线与输入框 → 启用当前维 → 刷新指标并广播。"""
        # 实测：拖拽/键盘可把线带出固定轴范围
        # → 钳回 [0, x_max]（score 恒 [0,1]）后再入状态
        v = min(max(float(v), 0.0), self._x_max)
        _name, dec, _step, _d = (self._score_spec() if self._current == "score"
                                 else _DIM_SPECS[self._current])
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
        self._emit_thresholds_changed(debounce=debounce)

    def _on_line_moved(self):
        # V0.7.4 二.1：真实拖线期间的广播走防抖（moving=拖线中；程序化
        # setValue 立即广播，回归锁 PanelSignalTest 依赖同步语义）
        self._apply_threshold(float(self._line.value()),
                              debounce=bool(getattr(self._line, "moving", False)))

    def _on_spin_changed(self, v: float):
        self._apply_threshold(float(v))

    # ---- 绘制 ----
    def _redraw(self):
        self.plot.clear()
        leg = self.plot.getPlotItem().legend
        if leg is not None:
            leg.clear()   # pg 图例不随 plot.clear() 清空，重绘前手动清
        self.plot.addItem(self._line)
        key = self._current
        self._x_max, xs = self._plan_axis(key)
        self._line.setBounds((0.0, float(self._x_max)))   # V0.7.4 二.1
        self._apply_fixed_axes(key)
        if self._scope == "unjudged":
            sw = sweep_unjudged(self._stats, key, xs, self._sweep_fixed())
            series = [("OK", _GREEN, sw["ok"]), ("NG", _RED, sw["ng"])]
        else:
            # V0.7.2 实测修复：曲线=**图口径**——该阈值下三组的图数，
            # 与右侧分组/统计行完全同口径（正确 31/过检 53 即曲线读数）
            from .review_logic import sweep_group_counts
            from ..i18n import tr

            sw = sweep_group_counts(self._stats, key, xs, self._sweep_fixed())
            over_name = (tr("错误") if self._stats.get("task") == "classification"
                         else tr("过检"))
            series = [(tr("正确"), _GREEN, sw["correct"]),
                      (over_name, _RED, sw["over"]),
                      (tr("漏检"), _BLUE, sw["missed"])]
            if not self._stats.get("gt_available"):
                # 无 GT：生产图无真值，全在未评估性质——只画有检出过阈图数
                swu = sweep_unjudged(self._stats, key, xs, self._sweep_fixed())
                series = [(tr("过检"), _RED, swu["ng"])]
        top = 1
        for name, color, ys in series:
            if not ys or max(ys) <= 0:
                continue
            top = max(top, int(max(ys)))
            self.plot.plot(np.asarray(xs, dtype=float), np.asarray(ys, float),
                           pen=pg.mkPen(color, width=2),
                           symbol="o", symbolSize=5,
                           symbolBrush=pg.mkBrush(color),
                           symbolPen=pg.mkPen(None), name=name, antialias=True)
        y_max = max(_nice_ceil(top * _Y_PAD), 4)
        self.plot.setYRange(0, float(y_max), padding=0)
        self._refresh_numbers()

    def _apply_fixed_axes(self, key: str):
        """固定坐标轴（V0.7.1 约定）：轴不随拖动/重绘变化，只有阈值线可动。"""
        x_max = self._x_max
        dec = _DIM_SPECS[key][1]
        self.plot.setXRange(0.0, x_max, padding=0)
        # 注意：底部轴禁止 setWidth——那会钳死轴控件宽度，刻度文字全部
        # 挤进左端几十像素（V0.7.2 任务①根因，P237）；左轴保留预留宽度
        self.plot.getAxis("bottom").setTicks(_axis_ticks(x_max, dec))
        self.plot.getAxis("left").setWidth(42)

    def _refresh_numbers(self):
        key = self._current
        t = self.current_threshold()
        if self._scope == "unjudged":
            m = sweep_metrics(self._stats, key, t, self._sweep_fixed())
            if self.lbl_p_name is not None:
                self.lbl_p_name.setText("NG")
                self.lbl_r_name.setText("OK")
                self.lbl_o_name.setText("")
            self.lbl_precision.setText(str(m["ng_img"]))
            self.lbl_recall.setText(str(m["ok_img"]))
            self.lbl_ok.setText("—")
            return
        from ..engine.metrics.batch_match import (threshold_metrics,
                                                  threshold_metrics_for)

        if not self._has_per_dets():
            # 无逐检出记录（旧缓存/轻量 stats）：回退 dims 汇总口径
            m = (threshold_metrics(self._stats, t) if key == "score"
                 else threshold_metrics_for(self._stats, key, t))
        else:
            m = sweep_metrics(self._stats, key, t, self._sweep_fixed())
        if self.lbl_p_name is not None:
            self.lbl_p_name.setText("精确率")
            self.lbl_r_name.setText("召回率")
            self.lbl_o_name.setText("OK误检")
        self.lbl_precision.setText(
            f"{m['precision']:.3f}" if m["precision"] is not None else "—")
        self.lbl_recall.setText(
            f"{m['recall']:.3f}" if m["recall"] is not None else "—")
        self.lbl_ok.setText(
            f"{m['ok_fp_per_img']:.2f}" if m["ok_fp_per_img"] is not None
            else "—")

    def _has_per_dets(self) -> bool:
        from ..engine.metrics.batch_match import _is_unjudged

        return any(not _is_unjudged(img) and img.get("dets")
                   for img in (self._stats.get("per_image") or []))
