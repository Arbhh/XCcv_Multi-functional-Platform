"""「传统平台」算子工作台（Q3，；V0.6 F2-4 布局修订）。

三区布局（数据源 = engine.flow 注册表，无新引擎）：
- 左：算子库——注册表全量算子，按分类过滤，点选显示功能说明与参数 schema；
- 中：单算子试跑——参数表单（schema 自动渲染）→ 选图 → 执行；中下部为
  大尺寸**运行预览**（随窗口自适应；滚轮缩放、拖拽平移、双击恢复适应）：
  有图像输出的算子显示输出图，定位/测量/过滤等无图像输出的算子显示
  「输入+结果叠加」；其下为自定义算子按钮排（导入/编辑/刷新）；
- 右：算子组合测试——依次加入算子自动按端口类型串链 → 运行；右下为
  **运行日志**（单算子试跑与组合链共用：节点级日志 + 非图像结果）。
  组合链运行后预览末节点；点选链中算子可切换预览该步结果。

实现纪律：纯逻辑都在 engine.flow（本文件只做 Qt 壳 + 预览取图）；所有 UI
方法结尾统一经 qt_slots 包装（模块末尾 apply_auto_slot_module）。
"""

import logging
log = logging.getLogger(__name__)
import time

import numpy as np
from PyQt5.QtCore import QRectF, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGraphicsScene, QGraphicsView, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton,
    QScrollArea, QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

from .engine.flow.ir import (
    Geometry, ImageData, ImageList, MeasureResult, NumberVal, ResultList,
)
from .i18n import tr
from .modeltools.flow_session import FlowSession, node_library_payload

_PREVIEW_MAX_SIDE = 4096   # 预览图长边上限（超大图降采样，防显存/内存暴涨）
_PREVIEW_HINT = ("运行预览：试跑后在此显示结果——有图像输出的算子显示输出图，"
                 "定位/测量/过滤等算子显示「输入+结果叠加」")


def _to_bgr_u8(mat) -> np.ndarray:
    """预览显示用：ImageData 矩阵 → 连续 BGR uint8（契约外位深/通道的防御）。"""
    import cv2

    m = np.asarray(mat)
    if m.dtype == bool:
        m = m.astype(np.uint8) * 255
    elif m.dtype != np.uint8:
        m = cv2.normalize(m.astype(np.float32), None, 0, 255,
                          cv2.NORM_MINMAX).astype(np.uint8)
    if m.ndim == 3 and m.shape[2] == 1:
        m = m[:, :, 0]
    if m.ndim == 2:
        m = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    elif m.ndim == 3 and m.shape[2] == 4:
        m = cv2.cvtColor(m, cv2.COLOR_BGRA2BGR)
    return np.ascontiguousarray(m)


def _first_image(outputs: dict):
    """输出字典中第一个非空图像：(端口名, ndarray) 或 (None, None)。"""
    for name, v in outputs.items():
        if isinstance(v, ImageData) and v.mat is not None and v.mat.size:
            return name, v.mat
    return None, None


def build_preview(executor, node_id: int):
    """节点运行预览（纯逻辑，可在工作线程调用）→ (BGR uint8 ndarray | None, 说明)。

    - 节点有图像输出 → 首个图像输出；
    - 无图像输出（定位/测量/过滤/逻辑类）→ 结果叠加到其输入图上。工作台的
      图恒为线性链（#1 图像头，#2.. 依次串接），向上游逐个节点找最近的
      图像输出即为该节点的输入底图。
    不依赖快照仓：同参数重跑整链命中会话缓存时执行器不重存快照。
    """
    from .engine.flow.snapshot import downscale, overlay_result

    if executor is None:
        return None, tr("尚未运行")
    results = executor.results
    res = results.get(int(node_id))
    if res is None or getattr(res, "skipped", False):
        return None, tr("该节点未执行（上游失败被跳过，原因见右下角日志）")
    if not res.ok:
        return None, tr("运行失败（原因见右下角日志）")
    name, mat = _first_image(res.outputs)
    if mat is not None:
        # 图像直通节点（检出/定位类）结果仍有价值：有几何/检出输出时优先
        # 展示「输出+结果叠加」（F2-4 语义在端口统一后保持）
        drawable = any(
            isinstance(v, Geometry)
            or (getattr(v, "boxes", None) is not None and not isinstance(v, type(None)))
            for v in res.outputs.values())
        if drawable:
            try:
                vis = overlay_result(_to_bgr_u8(mat), res.outputs)
                n_boxes = sum(len(v.boxes) for v in res.outputs.values()
                              if getattr(v, "boxes", None) is not None)
                h, w = mat.shape[:2]
                note = (f"{tr('输出+结果叠加')}（{n_boxes} {tr('个目标')}）" if n_boxes
                        else tr("输出+结果叠加")) + f" {w}×{h}"
                return np.ascontiguousarray(downscale(vis, _PREVIEW_MAX_SIDE)), note
            except Exception:
                log.debug("预览叠加渲染失败（回退纯输出预览）")
        h, w = mat.shape[:2]
        return (np.ascontiguousarray(downscale(_to_bgr_u8(mat), _PREVIEW_MAX_SIDE)),
                f"{tr('输出图像')}「{name}」 {w}×{h}")
    list_preview = _list_preview(results, int(node_id), res)
    if list_preview is not None:
        return list_preview
    base = None
    for nid in range(int(node_id) - 1, 0, -1):
        r = results.get(nid)
        if r is not None and r.ok:
            _n, base = _first_image(r.outputs)
            if base is not None:
                break
    if base is None:
        return None, tr("无图像可预览（结果见右下角日志）")
    n_boxes = 0
    drawable = False
    for v in res.outputs.values():
        if isinstance(v, Geometry):
            drawable = True
        else:
            boxes = getattr(v, "boxes", None)
            if boxes is not None:
                drawable = True
                n_boxes += len(boxes)
    vis = overlay_result(_to_bgr_u8(base), res.outputs)
    h, w = base.shape[:2]
    if not drawable:
        note = f"{tr('输入图（结果为数值，见右下角日志）')} {w}×{h}"
    elif n_boxes:
        note = f"{tr('输入+结果叠加')}（{n_boxes} {tr('个目标')}） {w}×{h}"
    else:
        note = f"{tr('输入+结果叠加')} {w}×{h}"
    return np.ascontiguousarray(downscale(vis, _PREVIEW_MAX_SIDE)), note


def _list_preview(results: dict, node_id: int, res):
    """列表端口预览：图像列表 → 拼图；结果列表 → 叠加到上游对应裁图后拼图。

    返回 (ndarray, 说明) 或 None（非列表输出，交回常规路径）。
    """
    from .engine.flow.snapshot import mosaic, overlay_result

    for name, v in res.outputs.items():
        if isinstance(v, ImageList):
            vis = mosaic([_to_bgr_u8(it.mat) for it in v.items if it.mat.size])
            return (np.ascontiguousarray(vis),
                    f"{tr('图像列表')}「{name}」{len(v)} {tr('张')}"
                    f"（{tr('拼图预览，最多显示 16 张')}）")
    rl = next((v for v in res.outputs.values() if isinstance(v, ResultList)), None)
    if rl is None:
        return None
    src = None
    for nid in range(node_id - 1, 0, -1):
        r = results.get(nid)
        if r is not None and r.ok:
            src = next((v for v in r.outputs.values() if isinstance(v, ImageList)), None)
            if src is not None:
                break
    kinds = sorted({it.kind for it in rl.items})
    summary = f"{tr('逐图结果')} {len(rl)} {tr('条')}（{'/'.join(kinds) or tr('空')}）"
    if src is None or not len(src):
        return None, f"{summary}（{tr('无对应图像列表可预览')}）"
    by_index = {it.meta.get("index", i): it for i, it in enumerate(rl.items)}
    vis = mosaic([overlay_result(_to_bgr_u8(img.mat), {"r": by_index[k].value})
                  if k in by_index else _to_bgr_u8(img.mat)
                  for k, img in enumerate(src.items)])
    return np.ascontiguousarray(vis), f"{summary}，{tr('逐张叠加拼图')}"


def _to_pixmap(img: np.ndarray) -> QPixmap:
    """BGR uint8 连续数组 → QPixmap（fromImage 深拷贝，数组可随即释放）。"""
    h, w = img.shape[:2]
    return QPixmap.fromImage(QImage(img.data, w, h, 3 * w, QImage.Format_BGR888))


class _RunWorker(QThread):
    """试跑工作线程（T2-2：大图/慢算子不再阻塞主线程）。

    FlowSession.run 与预览取图（叠加绘制）都在工作线程执行；结果经信号回
    主线程渲染（主线程读 executor.results 时工作线程已结束，无并发读）。
    """

    done = pyqtSignal(object)   # summary dict，附 "preview": (ndarray|None, 说明)
    failed = pyqtSignal(str)

    def __init__(self, session, graph: dict, label: str, focus: int, parent=None):
        super().__init__(parent)
        self._session = session
        self._graph = graph
        self._label = label
        self._focus = int(focus)

    def run(self):
        try:
            summary = self._session.run(self._graph, label=self._label)
        except Exception as e:
            log.warning("工作台预览构建失败（failed 信号上报）: %s", e, exc_info=True)
            self.failed.emit(str(e))
            return
        try:
            summary["preview"] = build_preview(self._session.executor, self._focus)
        except Exception as e:  # 预览失败不影响运行结果
            summary["preview"] = (None, f"{tr('预览生成失败：')}{e}")
        self.done.emit(summary)


class _PreviewView(QGraphicsView):
    """运行预览：等比适应窗口；滚轮缩放（以鼠标为中心）、左键拖拽平移、双击恢复适应。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = None
        self._user_zoomed = False
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setMinimumHeight(160)

    def set_pixmap(self, pixmap):
        self._scene.clear()
        self._item = None
        self._user_zoomed = False
        if pixmap is None or pixmap.isNull():
            self._scene.setSceneRect(QRectF())
            return
        self._item = self._scene.addPixmap(pixmap)
        self._item.setTransformationMode(Qt.SmoothTransformation)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.fit()

    def has_image(self) -> bool:
        return self._item is not None

    def image_size(self):
        """当前预览图 (宽, 高)；无图为 None。"""
        if self._item is None:
            return None
        pm = self._item.pixmap()
        return pm.width(), pm.height()

    def fit(self):
        if self._item is None:
            return
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        if self._item is None:
            return
        self._user_zoomed = True
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, event):
        self._user_zoomed = False
        self.fit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._user_zoomed:
            self.fit()


def _format_value(v) -> str:
    """非图像输出 → 人读文本（日志区展示）。"""
    if isinstance(v, ImageData):
        return f"{tr('图像')} {v.mat.shape[1]}×{v.mat.shape[0]}"
    if isinstance(v, Geometry):
        return f"{tr('几何')} {v.kind}：{np.round(np.asarray(v.data, dtype=float), 2).tolist()}"
    if isinstance(v, (MeasureResult, NumberVal)):
        return str(v.to_json())
    if isinstance(v, ImageList):
        return f"{tr('图像列表')} {len(v)} {tr('张')}；{tr('原图矩形')}={[it.meta.get('roi_rect') for it in v.items][:8]}" \
            + ("…" if len(v) > 8 else "")
    if isinstance(v, ResultList):
        kinds = {}
        for it in v.items:
            kinds[it.kind] = kinds.get(it.kind, 0) + 1
        return f"{tr('逐图结果')} {len(v)} {tr('条')}；{tr('按种类')}={kinds}"
    boxes = getattr(v, "boxes", None)
    if boxes is not None:
        return f"{tr('检出')} {len(boxes)} {tr('个目标')}；boxes={[[round(float(x), 1) for x in b] for b in boxes]}"
    return repr(v)


# ---- 参数编辑控件：禁用滚轮调值----
# 滚轮一律 ignore 向上传播 → 滚动参数面板，数值只允许键入/点按钮修改。
class _NoWheelSpin(QSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class _NoWheelDoubleSpin(QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class _NoWheelCombo(QComboBox):
    def wheelEvent(self, event):
        event.ignore()


class OperatorWorkbench(QWidget):
    """主界面「传统平台」页：算子库 / 单算子试跑+运行预览 / 组合测试+运行日志。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        lib = node_library_payload()
        self._lib = {f"xccv2/{n['name']}": n for n in lib["nodes"]}
        self._session = FlowSession()
        self._param_edits: dict = {}
        self._chain_types: list = []
        self._run_thread = None  # 试跑工作线程（运行中禁再次启动，T2-2）
        self._preview_owner = None   # 预览/会话结果归属：("single", None) | ("chain", 链签名)
        self._pending_chain_sig = None
        self._build_ui()
        self._refresh_ops("全部")
        from .i18n import localize_widget_tree  # 内嵌工作台：英文替换层

        localize_widget_tree(self)

    # ---------- UI ----------
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        split = QSplitter(self)
        root.addWidget(split)

        # ---- 左：算子库 ----
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.cat_combo = QComboBox()
        cats = ["全部", "图像源", "预处理", "定位", "测量", "逻辑", "检出过滤", "AI",
                "通用", "输出", "自定义"]
        from .i18n import tr

        for c in cats:   # 过滤按 userData（原始中文）匹配，显示走 tr
            self.cat_combo.addItem(tr(c), c)
        self.cat_combo.currentIndexChanged.connect(
            lambda _i: self._refresh_ops(self.cat_combo.currentData()))
        lv.addWidget(self.cat_combo)
        self.op_list = QListWidget()
        self.op_list.currentItemChanged.connect(self._on_op_selected)
        lv.addWidget(self.op_list, stretch=1)
        self.detail_box = QPlainTextEdit()
        self.detail_box.setReadOnly(True)
        self.detail_box.setPlaceholderText(tr("点选算子查看功能说明与参数定义"))
        self.detail_box.setMaximumHeight(170)
        lv.addWidget(self.detail_box)
        split.addWidget(left)

        # ---- 中：单算子试跑（上：参数/选图/试跑；下：运行预览 + 自定义算子按钮排）----
        mid = QSplitter(Qt.Vertical)
        mid.setChildrenCollapsible(False)
        mid_top = QWidget()
        mv = QVBoxLayout(mid_top)
        mv.setContentsMargins(0, 0, 0, 0)
        mv.addWidget(QLabel(tr("单算子试跑")))
        form_group = QGroupBox(tr("参数（选算子后自动生成）"))
        self.form_area = QScrollArea()
        self.form_area.setWidgetResizable(True)
        self.form_host = QWidget()
        self.form_lay = QFormLayout(self.form_host)
        self.form_lay.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.form_lay.setHorizontalSpacing(8)
        self.form_area.setWidget(self.form_host)
        fl = QVBoxLayout(form_group)
        fl.addWidget(self.form_area)
        mv.addWidget(form_group, stretch=1)
        img_row = QHBoxLayout()
        self.run_image_edit = QLineEdit()
        self.run_image_edit.setPlaceholderText(tr("试跑图片路径（必填）"))
        btn_img = QPushButton(tr("浏览…"))
        btn_img.clicked.connect(self._pick_run_image)
        img_row.addWidget(self.run_image_edit, stretch=1)
        img_row.addWidget(btn_img)
        mv.addLayout(img_row)
        btn_run = QPushButton(tr("▶ 试跑所选算子"))
        btn_run.clicked.connect(self._run_single)
        self.btn_run = btn_run  # T2-2：运行中禁用
        mv.addWidget(btn_run)
        mid.addWidget(mid_top)

        mid_bottom = QWidget()
        pv = QVBoxLayout(mid_bottom)
        pv.setContentsMargins(0, 0, 0, 0)
        preview_group = QGroupBox(tr("运行预览"))
        pg = QVBoxLayout(preview_group)
        from .i18n import tr

        self.preview_caption = QLabel(tr(_PREVIEW_HINT))
        self.preview_caption.setWordWrap(True)
        pg.addWidget(self.preview_caption)
        self.preview = _PreviewView()
        self.preview.setToolTip(tr("滚轮缩放、左键拖拽平移、双击恢复适应窗口"))
        pg.addWidget(self.preview, stretch=1)
        pv.addWidget(preview_group, stretch=1)
        # 自定义算子按钮排（V0.6 T4-5：试跑结果预览之后）
        op_row = QHBoxLayout()
        self.btn_import_op = QPushButton(tr("导入算子"))
        self.btn_import_op.setToolTip(tr("导入 .xcvop 自定义算子包"))
        self.btn_import_op.clicked.connect(self._import_op)
        self.btn_edit_op = QPushButton(tr("编辑算子"))
        self.btn_edit_op.setToolTip(tr("编辑所选自定义算子；选中内置算子或未选时新建"))
        self.btn_edit_op.clicked.connect(self._edit_op)
        self.btn_reload_op = QPushButton(tr("刷新"))
        self.btn_reload_op.setToolTip(tr("重新读取 platform_data/operators 下的自定义算子（热重载）"))
        self.btn_reload_op.clicked.connect(lambda: self._reload_ops())
        for b in (self.btn_import_op, self.btn_edit_op, self.btn_reload_op):
            op_row.addWidget(b)
        pv.addLayout(op_row)
        self._editor = None
        mid.addWidget(mid_bottom)
        mid.setStretchFactor(0, 2)
        mid.setStretchFactor(1, 3)
        split.addWidget(mid)

        # ---- 右：组合测试（上）+ 运行日志（下，单算子/组合链共用）----
        right = QSplitter(Qt.Vertical)
        right.setChildrenCollapsible(False)
        right_top = QWidget()
        rv = QVBoxLayout(right_top)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(QLabel(tr("算子组合测试（按加入顺序自动串链）")))
        self.chain_list = QListWidget()
        self.chain_list.setToolTip(tr("运行组合链后，点选链中算子可在中间预览该步结果"))
        self.chain_list.currentRowChanged.connect(self._on_chain_row)
        rv.addWidget(self.chain_list, stretch=1)
        chain_btns = QHBoxLayout()
        btn_add = QPushButton(tr("加入所选算子"))
        btn_add.clicked.connect(self._chain_add)
        btn_del = QPushButton(tr("移除选中"))
        btn_del.clicked.connect(self._chain_remove)
        btn_clear = QPushButton(tr("清空"))
        btn_clear.clicked.connect(self._chain_clear)
        for b in (btn_add, btn_del, btn_clear):
            chain_btns.addWidget(b)
        rv.addLayout(chain_btns)
        btn_chain_run = QPushButton(tr("▶ 运行组合链"))
        btn_chain_run.clicked.connect(self._run_chain)
        self.btn_chain_run = btn_chain_run  # T2-2：运行中禁用
        rv.addWidget(btn_chain_run)
        right.addWidget(right_top)

        log_group = QGroupBox(tr("运行日志（单算子试跑 / 组合链共用）"))
        lg = QVBoxLayout(log_group)
        self.log_out = QPlainTextEdit()
        self.log_out.setReadOnly(True)
        self.log_out.setPlaceholderText(
            tr("每个节点的执行结果与日志（[ERROR] 算子名(#id)：原因）；"
               "几何/数值/检出等非图像结果也列在这里"))
        lg.addWidget(self.log_out, stretch=1)
        log_btns = QHBoxLayout()
        log_btns.addStretch(1)
        self.btn_clear_log = QPushButton(tr("清空日志"))
        self.btn_clear_log.clicked.connect(self._clear_log)
        log_btns.addWidget(self.btn_clear_log)
        lg.addLayout(log_btns)
        right.addWidget(log_group)
        split.addWidget(right)
        split.setSizes([280, 560, 400])
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 1)
        # 兼容别名：既有测试/探针按旧名访问（日志区与预览区已合并为一处）
        self.single_out = self.chain_out = self.log_out
        self.single_preview = self.preview

    def _clear_log(self):
        self.log_out.clear()

    # ---------- 算子库 ----------
    def set_search_query(self, text: str) -> None:
        """主界面全局搜索：按算子名/说明过滤算子列表（跨类别）。"""
        self._query = (text or "").strip().lower()
        self._refresh_ops(self.cat_combo.currentData())

    def _refresh_ops(self, category: str):
        from .i18n import tr

        query = getattr(self, "_query", "")
        self.op_list.blockSignals(True)
        self.op_list.clear()
        for type_name, info in self._lib.items():
            if category != "全部" and info["category"] != category:
                continue
            if query:
                hay = (info["name"] + tr(info["name"])
                       + (info.get("desc") or "")).lower()
                if query not in hay:
                    continue
            item = QListWidgetItem(
                f"{tr(info['name'])}　·　{tr(info['category'])}")
            item.setData(Qt.UserRole, type_name)
            item.setToolTip(tr(info.get("desc") or ""))
            self.op_list.addItem(item)
        self.op_list.blockSignals(False)

    def _on_op_selected(self, cur, _prev=None):
        self._param_edits = {}
        while self.form_lay.rowCount():
            self.form_lay.removeRow(0)
        if cur is None:
            return
        info = self._lib[cur.data(Qt.UserRole)]
        lines = [f"{tr(info['name'])}（{tr(info['category'])}）", tr(info.get("desc") or ""),
                 "", tr("参数：")]
        for key, spec in info["params"].items():
            lines.append(f"  · {tr(spec['label'])}（{spec['type']}）：{tr(spec['tooltip']) or '—'}")
        self.detail_box.setPlainText("\n".join(lines))
        self._build_param_form(info)

    def _build_param_form(self, info: dict):
        """按 schema 渲染参数表单（属性面板同源，Python 唯一事实源）。"""
        for key, spec in info["params"].items():
            label = QLabel(tr(spec["label"]))
            label.setToolTip(tr(spec["tooltip"] or ""))
            edit = self._make_edit(spec)
            self._param_edits[key] = (spec, edit)
            self.form_lay.addRow(label, edit)

    @staticmethod
    def _make_edit(spec: dict):
        t = spec["type"]
        if t == "int":
            w = _NoWheelSpin()
            lo, hi = spec["range"] or (0, 999999)
            w.setRange(int(lo), int(hi))
            w.setValue(int(spec["default"]))
            w.setMinimumWidth(120)
            return w
        if t == "float":
            w = _NoWheelDoubleSpin()
            lo, hi = spec["range"] or (-1e9, 1e9)
            w.setRange(float(lo), float(hi))
            w.setDecimals(3)
            w.setValue(float(spec["default"]))
            w.setMinimumWidth(120)
            return w
        if t == "bool":
            w = QCheckBox()
            w.setChecked(bool(spec["default"]))
            return w
        if t == "enum" and all(isinstance(o, str) for o in (spec["options"] or [])):
            w = _NoWheelCombo()
            w.addItems([str(o) for o in spec["options"]])
            w.setCurrentText(str(spec["default"]))
            return w
        if t == "enum":  # 非字符串枚举：下拉显示、按索引取回原值
            w = _NoWheelCombo()
            w.addItems([str(o) for o in spec["options"]])
            w._xccv_options = list(spec["options"])  # 原值映射
            try:
                w.setCurrentIndex(list(spec["options"]).index(spec["default"]))
            except ValueError:
                w.setCurrentIndex(0)
            return w
        return QLineEdit(str(spec["default"] if spec["default"] is not None else ""))

    def _collect_params(self, info: dict) -> dict:
        out = {}
        for key, (spec, edit) in self._param_edits.items():
            t = spec["type"]
            if t == "int":
                out[key] = int(edit.value())
            elif t == "float":
                out[key] = float(edit.value())
            elif t == "bool":
                out[key] = bool(edit.isChecked())
            elif t == "enum" and all(isinstance(o, str) for o in (spec["options"] or [])):
                out[key] = edit.currentText()
            elif t == "enum":
                out[key] = getattr(edit, "_xccv_options", [])[edit.currentIndex()]
            else:
                out[key] = edit.text().strip()
        return out

    # ---------- 单算子试跑 ----------
    def _pick_run_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("选择试跑图片"), "", tr("图片 (*.jpg *.jpeg *.png *.bmp)"))
        if path:
            self.run_image_edit.setText(path)

    # ---------- 自定义算子（V0.6 T4-4/T4-5） ----------
    def _custom_root(self):
        """自定义算子目录（None=platform_data/operators；测试可替换 custom_ops.operators_dir）。"""
        return None

    def _import_op(self):
        from .engine.flow import custom_ops

        path, _ = QFileDialog.getOpenFileName(self, tr("导入自定义算子"), "",
                                              tr("XCcv 自定义算子 (*.xcvop)"))
        if not path:
            return
        try:
            op_id = custom_ops.import_xcvop(path, self._custom_root(), overwrite=False)
        except FileExistsError:
            if QMessageBox.question(self, tr("导入算子"), tr("同名算子已存在，是否覆盖？")) \
                    != QMessageBox.Yes:
                return
            op_id = custom_ops.import_xcvop(path, self._custom_root(), overwrite=True)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, tr("导入算子"), f"{tr('导入失败：')}{e}")
            return
        self._reload_ops(select=custom_ops.PREFIX + op_id)

    def _edit_op(self):
        """所选为自定义算子 → 编辑；否则新建（非模态，边改边试跑）。"""
        from .engine.flow import custom_ops
        from .operator_editor import OperatorEditorDialog

        info = self._current_info()
        op_def = None
        if info is not None and info["name"].startswith(custom_ops.PREFIX):
            try:
                op_def = custom_ops.read_op(info["name"][len(custom_ops.PREFIX):],
                                            self._custom_root())
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(self, tr("编辑算子"), f"{tr('读取失败：')}{e}")
                return
        if self._editor is not None:
            self._editor.close()
        self._editor = OperatorEditorDialog(self, op_def, self._custom_root())
        self._editor.saved.connect(self._on_editor_saved)
        self._editor.show()

    def _on_editor_saved(self, name: str):
        self._reload_ops(select=name, reload_disk=False)   # save_op 已热重载注册表

    def _reload_ops(self, select: str | None = None, reload_disk: bool = True):
        """热重载：重读算子目录 → 重建节点库/列表 → 新会话（防命中旧实现的缓存）。"""
        from .engine.flow import custom_ops

        errors = []
        if reload_disk:
            _loaded, errors = custom_ops.load_custom_ops(self._custom_root())
        lib = node_library_payload()
        self._lib = {f"xccv2/{n['name']}": n for n in lib["nodes"]}
        if self._run_thread is None:
            self._session = FlowSession()
            self._preview_owner = None   # 旧会话结果已不可点选预览
        self._refresh_ops(self.cat_combo.currentData())
        if select:
            for i in range(self.op_list.count()):
                item = self.op_list.item(i)
                if item.data(Qt.UserRole) == f"xccv2/{select}":
                    self.op_list.setCurrentItem(item)
                    break
        n_custom = sum(1 for k in self._lib if k.startswith("xccv2/" + custom_ops.PREFIX))
        from .ui_widgets import toast

        toast(self, f"{tr('算子库已刷新')}：{n_custom} {tr('个自定义算子')}",
              "success" if not errors else "warn")
        if errors:
            QMessageBox.warning(self, tr("自定义算子"),
                                tr("以下文件未加载：") + "\n" + "\n".join(errors[:10]))

    # ---------- 试跑执行（T2-2 后台化：QThread + 忙态 + 完成信号回填） ----------
    def _run_async(self, graph: dict, label: str, on_done, focus: int,
                   title: str) -> bool:
        """在工作线程跑 FlowSession.run（+预览取图）；运行中禁用试跑按钮并拒绝重入。"""
        if self._run_thread is not None:
            QMessageBox.information(self, tr("试跑"), tr("已有试跑在进行，请稍候"))
            return False
        self.btn_run.setEnabled(False)
        self.btn_chain_run.setEnabled(False)
        self.log_out.appendPlainText(f"—— {title}（{time.strftime('%H:%M:%S')}）——")
        self.preview_caption.setText(f"{tr('运行预览')}：{title} · {tr('运行中…')}")
        # 无父对象：工作台可能在线程运行中被关闭（parented 运行线程
        # 销毁即 Qt 崩溃）；finished→deleteLater 已接线兜底
        self._run_thread = _RunWorker(self._session, graph, label, focus)
        self._run_thread.done.connect(
            lambda summary: self._finish_run(on_done, summary, None))
        self._run_thread.failed.connect(
            lambda msg: self._finish_run(on_done, None, msg))
        self._run_thread.finished.connect(self._run_thread.deleteLater)  # A-3 纪律
        self._run_thread.finished.connect(self._clear_run_thread)
        self._run_thread.start()
        return True

    def _clear_run_thread(self):
        self._run_thread = None

    def _finish_run(self, on_done, summary, error):
        self.btn_run.setEnabled(True)
        self.btn_chain_run.setEnabled(True)
        if error is not None:
            self.log_out.appendPlainText(f"✘ {tr('运行失败：')}{error}")
            self.preview_caption.setText(
                f"{tr('运行预览')}：{tr('运行失败（原因见右下角日志）')}")
            QMessageBox.information(self, tr("试跑"), f"{tr('运行失败：')}{error}")
            return
        on_done(summary)

    def _run_single(self):
        info = self._current_info()
        if info is None:
            QMessageBox.information(self, tr("单算子试跑"), tr("请先在左侧选择算子"))
            return
        img_path = self.run_image_edit.text().strip()
        if not img_path:
            QMessageBox.information(self, tr("单算子试跑"), tr("请先选择试跑图片"))
            return
        # 该算子第一个必需 image 输入的槽位；没有 image 输入的算子不支持单跑
        slot = next((i for i, p in enumerate(info["inputs"])
                     if p["type"] == "image" and p["required"]), None)
        if slot is None:
            QMessageBox.information(
                self, tr("单算子试跑"),
                f"「{info['name']}」{tr('没有必需的图像输入，请在右侧组合测试中搭建链路')}")
            return
        params = self._collect_params(info)
        graph = {"nodes": [
            {"id": 1, "type": "xccv2/image_loader",
             "properties": {"path": img_path, "is_grayscale": False}},
            {"id": 2, "type": "xccv2/" + info["name"], "properties": params}],
            "links": [[1, 1, 0, 2, slot]]}
        self._single_title = f"{info['name']}(#2)"
        self._run_async(graph, f"工作台单算子:{info['name']}",
                        self._after_run_single, focus=2,
                        title=f"{tr('单算子试跑')}：{info['name']}")

    def _after_run_single(self, summary):
        self._show_summary(summary, self.log_out)
        self._show_outputs(2, self._single_title, summary.get("preview"))
        self._preview_owner = ("single", None)

    def _current_info(self):
        cur = self.op_list.currentItem()
        return self._lib[cur.data(Qt.UserRole)] if cur else None

    def _show_outputs(self, node_id: int, title: str, preview):
        """节点输出：图像/叠加图进中间运行预览，其余人读格式进右下日志。"""
        self._set_preview(title, preview)
        ex = self._session.executor
        res = ex.results.get(node_id) if ex is not None else None
        if res is None or not res.ok:
            return
        for name, v in res.outputs.items():
            if not isinstance(v, ImageData):
                self.log_out.appendPlainText(f"[{name}] {_format_value(v)}")

    def _set_preview(self, title: str, preview):
        img, note = preview if preview else (None, tr("无预览"))
        self.preview.set_pixmap(_to_pixmap(img) if img is not None else None)
        self.preview_caption.setText(f"{tr('运行预览')}：{title} · {note}")

    # ---------- 组合测试 ----------
    def _chain_signature(self) -> tuple:
        """链身份（加入/移除/清空即变化）：判断会话结果是否仍对应当前链。"""
        return tuple(id(info) for info in self._chain_types)

    def _on_chain_row(self, row: int):
        """组合链运行后且链未改动：点选链中算子 → 预览该步结果（#k+2）。"""
        if row < 0 or self._run_thread is not None:
            return
        if self._preview_owner != ("chain", self._chain_signature()):
            return
        nid = row + 2
        try:
            preview = build_preview(self._session.executor, nid)
        except Exception as e:  # noqa: BLE001
            preview = (None, f"{tr('预览生成失败：')}{e}")
        self._set_preview(f"{self._chain_types[row]['name']}(#{nid})", preview)

    def _chain_add(self):
        info = self._current_info()
        if info is None:
            QMessageBox.information(self, tr("组合测试"), tr("请先在左侧选择算子"))
            return
        self._chain_types.append(info)
        QListWidgetItem(info["name"], self.chain_list)

    def _chain_remove(self):
        row = self.chain_list.currentRow()
        if row >= 0:
            self._chain_types.pop(row)
            self.chain_list.takeItem(row)

    def _chain_clear(self):
        self._chain_types.clear()
        self.chain_list.clear()

    def _run_chain(self):
        if not self._chain_types:
            QMessageBox.information(self, tr("组合测试"), tr("请先加入算子"))
            return
        img_path = self.run_image_edit.text().strip()
        if not img_path:
            QMessageBox.information(self, tr("组合测试"), tr("请先选择试跑图片"))
            return
        graph, error = self._build_chain_graph(img_path)
        if error:
            QMessageBox.information(self, tr("组合测试"), error)
            return
        self._last_chain_graph = graph  # 完成回填时取末节点 id
        last = graph["nodes"][-1]
        self._chain_title = f"{last['type'].split('/', 1)[1]}(#{last['id']})"
        self._pending_chain_sig = self._chain_signature()
        self._run_async(graph, "工作台组合链", self._after_run_chain,
                        focus=int(last["id"]),
                        title=f"{tr('组合链')}：{len(graph['nodes']) - 1} {tr('个算子')}")

    def _after_run_chain(self, summary):
        self._show_summary(summary, self.log_out)
        last_id = max(int(n["id"]) for n in self._last_chain_graph["nodes"])
        self._show_outputs(last_id, self._chain_title, summary.get("preview"))
        self._preview_owner = ("chain", self._pending_chain_sig)

    def _build_chain_graph(self, img_path: str):
        """图像头 + 依次串链（类型化端口的组合规则）：

        - 下游每个输入端口：上一步有同类型输出 → 连上一步；同类输出按
          **消耗配对**（result/result2 各连一个不同输出，避免两路接同一处）；
        - 仍未连上的**图像**输入（必需或可选）→ 连图像头（原图）。例如
          「区域保留 → 按检出裁图」：检出连上一步、图像连原图；
        - 必需输入两者都连不上 → 报错（不生成跑不过的图）；下游与上一步一条线
          都没连上也报错（否则链断开）。
        """
        nodes = [{"id": 1, "type": "xccv2/image_loader",
                  "properties": {"path": img_path, "is_grayscale": False}}]
        links = []
        prev_type = "xccv2/image_loader"
        prev_id = 1
        nid = 2
        for info in self._chain_types:
            required = [p for p in info["inputs"] if p["required"]]
            if not required and not info["inputs"]:
                return None, f"「{info['name']}」{tr('没有输入端口，无法串链')}"
            prev_info = self._lib.get(prev_type)
            free_outputs: dict = {}    # 类型 -> 尚未被本节点消耗的输出槽列表
            linked_prev = False
            for slot, port in enumerate(info["inputs"]):
                pt = port["type"]
                pool = free_outputs.setdefault(
                    pt, [i for i, p in enumerate(prev_info["outputs"]) if p["type"] == pt])
                if pool:
                    out_slot = pool.pop(0)
                    links.append([len(links) + 1, prev_id, out_slot, nid, slot])
                    linked_prev = True
                elif port["type"] == "image" and prev_id != 1:
                    links.append([len(links) + 1, 1, 0, nid, slot])   # 原图
                elif port["required"]:
                    return None, (f"「{prev_info['name']}」{tr('没有')} {port['type']}"
                                  f" {tr('类型输出，无法连接')}「{info['name']}」")
            if not linked_prev:
                return None, (f"「{prev_info['name']}」{tr('的输出无法连接')}「{info['name']}」"
                              f"（{tr('端口类型不匹配')}）")
            params = self._collect_params(info) if info is self._current_info() \
                else {k: s["default"] for k, s in info["params"].items()}
            nodes.append({"id": nid, "type": "xccv2/" + info["name"],
                          "properties": dict(params)})
            prev_type, prev_id = "xccv2/" + info["name"], nid
            nid += 1
        # 新建图按当前端口布局生成：带 schema 标记，载入侧跳过旧布局迁移
        return {"schema": 3, "nodes": nodes, "links": links}, None

    @staticmethod
    def _show_summary(summary: dict, textbox: QPlainTextEdit):
        textbox.appendPlainText(
            f"{tr('运行完成：')}{summary['ok_count']}/{summary['total']} {tr('节点成功')}"
            + (f"，{len(summary['errors'])} {tr('个失败')}" if summary["errors"] else ""))
        for nid, err in summary["errors"].items():
            textbox.appendPlainText(f"  ✘ {tr('节点')}#{nid}：{err}")
        if summary["logs"]:
            textbox.appendPlainText(summary["logs"])
        textbox.appendPlainText(
            tr("（本次运行日志已存 platform_data/logs/流程运行日志/）"))


if __name__ == "__main__":  # 独立调试入口
    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    w = OperatorWorkbench()
    w.resize(1280, 760)
    w.show()
    sys.exit(app.exec_())

# ---- Cython 编译态槽包装（信号槽参数裁剪修复，见 xccv/qt_slots.py）----
from xccv.qt_slots import apply_auto_slot_module  # noqa: E402
apply_auto_slot_module(globals())
