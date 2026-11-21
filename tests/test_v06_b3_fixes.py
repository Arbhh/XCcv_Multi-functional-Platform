"""回归。

- 标注：L 键智能标注（NameError）/ 批量预标注（auto_cleanup_thread 缺参）/
  画笔局部重绘 / 双击判定 250ms / 切图「保持缩放」开关；
- 训练：小目标锚框全链路（UI→Project→TrainConfig→ckpt）/ 卡片按更新时间倒序；
- 模型工具：缺陷生成固定优先端口（同进程多实例不撞端口）/ 加速尺寸建议 + 进度条；
- YOLO（anchor-free）独立工业损失 NWD-CIoU / 宽松口径逐轮指标（validator 检出收集）。
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import torch
from PyQt5 import QtCore, QtGui, QtWidgets

import xccv.annotation  # noqa: F401 先导入：固定 PyQt5 绑定并在需要时装 osam 桩
from xccv.ui_widgets import _prefs_path

ROOT = Path(__file__).resolve().parents[1]
_APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)


def _pump_until(pred, timeout=30.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        QtWidgets.QApplication.processEvents()
        if pred():
            return True
        time.sleep(0.01)
    return False


class _PrefsGuard(unittest.TestCase):
    """用例会写 ui_prefs.json：整文件备份/还原1092合并写删不掉键）。"""

    def setUp(self):
        p = _prefs_path()
        self._prefs_raw = p.read_bytes() if p.is_file() else None
        self.addCleanup(self._restore_prefs)

    def _restore_prefs(self):
        p = _prefs_path()
        if self._prefs_raw is None:
            if p.is_file():
                p.unlink()
        else:
            p.write_bytes(self._prefs_raw)


# ============================================================ 标注平台
def _main_window(dataset_dir):
    from labelme.app import MainWindow
    from labelme.config import get_default_config

    win = MainWindow(config=get_default_config())
    win.importDirImages(str(dataset_dir), load=True)
    return win


class SmartAnnotateSingleTest(_PrefsGuard):
    """L 键：原实现引用未定义的 labelme.utils / Path，按下必 NameError。"""

    def setUp(self):
        super().setUp()
        from tests.synthetic import make_labelme_dataset

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ds, _ = make_labelme_dataset(Path(self._tmp.name) / "ds",
                                          n_images=2)
        self.weights = Path(self._tmp.name) / "fake.pt"
        self.weights.write_bytes(b"0")

    def _run(self, task_value):
        pred = SimpleNamespace(labels=[0], scores=[0.9],
                               boxes=[[5.0, 5.0, 30.0, 30.0]], masks=None)
        fake = SimpleNamespace(task=SimpleNamespace(value=task_value),
                               classes=["cat"], predict=lambda arr: pred)
        win = _main_window(self.ds)
        self.addCleanup(win.close)
        win._config["ai"] = {"batch_model": str(self.weights)}
        n0 = len(win.canvas.shapes)
        with mock.patch("xccv.engine.models.registry.load_model",
                        return_value=fake):
            win._act_smart.trigger()
            done = _pump_until(lambda: "智能标注" in
                               win.statusBar().currentMessage()
                               and "推理中" not in
                               win.statusBar().currentMessage())
        self.assertTrue(done, "智能标注未结束")
        return win, n0

    def test_detection_model_appends_shapes(self):
        win, n0 = self._run("detection")
        self.assertIn("智能标注完成", win.statusBar().currentMessage())
        self.assertEqual(len(win.canvas.shapes), n0 + 1)

    def test_classification_model_rejected_with_message(self):
        win, n0 = self._run("classification")
        self.assertIn("需要检测/分割模型", win.statusBar().currentMessage())
        self.assertEqual(len(win.canvas.shapes), n0)


class AutoAnnotateDialogTest(_PrefsGuard):
    """批量预标注：auto_cleanup_thread 缺 thread 参数，点「开始」即 TypeError。"""

    def test_start_runs_and_remembers_paths(self):
        from xccv.annotation import auto_annotate
        from xccv.annotation.auto_annotate_window import AutoAnnotateDialog

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        model = Path(tmp.name) / "m.pt"
        model.write_bytes(b"0")

        def fake_runner(progress_cb=None, cancel=None, **kw):
            progress_cb(1, 1, "a.jpg")
            return {"total": 1, "annotated": 1, "skipped": 0,
                    "cancelled": False, "failed": 0, "images": []}

        dlg = AutoAnnotateDialog()
        self.addCleanup(dlg.close)
        dlg.model_edit.setText(str(model))
        dlg.folder_edit.setText(tmp.name)
        with mock.patch.object(auto_annotate, "annotate_folder", fake_runner):
            dlg.btn_start.click()
            ok = _pump_until(lambda: dlg.status.text().startswith("完成"))
        self.assertTrue(ok, dlg.status.text())
        self.assertEqual(dlg.bar.value(), 100)
        dlg2 = AutoAnnotateDialog()                 # 记住上次的模型/文件夹
        self.addCleanup(dlg2.close)
        self.assertEqual(dlg2.model_edit.text(), str(model))
        self.assertEqual(dlg2.folder_edit.text(), tmp.name)


def _canvas(mode):
    from labelme.widgets.canvas import Canvas

    cv = Canvas()
    cv.resize(200, 100)
    cv.pixmap = QtGui.QPixmap(200, 100)
    cv.pixmap.fill(QtGui.QColor(255, 255, 255))
    cv.createMode = mode
    cv.mode = Canvas.CREATE
    return cv


def _move(cv, x, y):
    ev = QtGui.QMouseEvent(QtCore.QEvent.MouseMove, QtCore.QPointF(x, y),
                           QtCore.Qt.NoButton, QtCore.Qt.NoButton,
                           QtCore.Qt.NoModifier)
    cv.mouseMoveEvent(ev)


class BrushRegionRepaintTest(unittest.TestCase):
    """笔刷框延迟：画笔移动不再同步整幅 repaint()，改局部区域 update。"""

    def test_brush_hover_does_not_full_repaint(self):
        cv = _canvas("brush")
        calls = []
        cv.repaint = lambda *a: calls.append(1)
        _move(cv, 40, 30)
        _move(cv, 60, 50)
        self.assertEqual(calls, [])
        want = cv.transformPos(QtCore.QPointF(60, 50))
        got = cv._brush_last_cursor
        self.assertAlmostEqual(got.x(), want.x(), places=3)
        self.assertAlmostEqual(got.y(), want.y(), places=3)

    def test_other_modes_keep_repaint(self):
        cv = _canvas("rectangle")
        calls = []
        cv.repaint = lambda *a: calls.append(1)
        _move(cv, 40, 30)
        self.assertEqual(calls, [1])

    def test_width_change_resets_cursor_cache(self):
        cv = _canvas("brush")
        _move(cv, 40, 30)
        self.assertIsNotNone(cv._brush_last_cursor)
        cv.setBrushWidth(30)
        self.assertIsNone(cv._brush_last_cursor)


class DoubleClick250Test(unittest.TestCase):
    """全平台双击判定 250ms。"""

    def test_ai_pending_click_window_is_250ms(self):
        from labelme.shape import Shape

        cv = _canvas("ai_polygon")
        cv.current = Shape(shape_type="points")
        cv._ai_prompts = [(QtCore.QPointF(5, 5), 1)]
        cv._ai_click(QtCore.QPointF(50, 50), 1, QtCore.QPointF(50, 50),
                     QtCore.Qt.NoModifier)
        try:
            self.assertTrue(cv._ai_click_timer.isActive())
            self.assertEqual(cv._ai_click_timer.interval(), 250)
        finally:
            cv._ai_click_timer.stop()

    def test_qt_global_and_web_tools(self):
        import inspect

        from xccv import platform_entry

        self.assertIn("setDoubleClickInterval(250)",
                      inspect.getsource(platform_entry.run_platform))
        js = (ROOT / "xccv/defectgen/web/js/interact.js").read_text(
            encoding="utf-8")
        self.assertIn("now - lastPolyClick.t < 250", js)
        lg = (ROOT / "xccv/modeltools/web/litegraph.js").read_text(
            encoding="utf-8")
        self.assertIn("now - this.last_mouseclick < 250", lg)


class KeepScaleToggleTest(_PrefsGuard):
    """实测补充：切图保留上一张缩放且无法关闭——按文件缩放记忆须服从开关。"""

    def test_toggle_controls_per_file_zoom_memory(self):
        from tests.synthetic import make_labelme_dataset
        from xccv.ui_widgets import load_prefs

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ds, _ = make_labelme_dataset(Path(tmp.name) / "ds", n_images=2)
        win = _main_window(ds)
        self.addCleanup(win.close)
        a, b = str(ds / "img00.png"), str(ds / "img01.png")

        win.enableKeepPrevScale(False)
        win.loadFile(a)
        win.setZoom(250)
        win.loadFile(b)
        win.loadFile(a)
        self.assertNotAlmostEqual(win.canvas.scale, 2.5, places=2,
                                  msg="关闭时回到 A 应重新适应窗口")
        self.assertIs(load_prefs().get("keep_prev_scale"), False)

        win.enableKeepPrevScale(True)
        win.loadFile(a)
        win.setZoom(250)
        win.loadFile(b)
        win.loadFile(a)
        self.assertAlmostEqual(win.canvas.scale, 2.5, places=2)
        self.assertIs(load_prefs().get("keep_prev_scale"), True)


# ============================================================ 训练平台
class AnchorSmallChainTest(unittest.TestCase):
    """小目标锚框：Project → TrainThread → TrainConfig → 网络 → ckpt 往返。"""

    def test_trainconfig_default_off(self):
        from xccv.engine.models.base import TrainConfig

        cfg = TrainConfig(task="detection", dataset_dir=".", output_dir=".",
                          model_name="x")
        self.assertFalse(cfg.anchor_small)

    def test_trainer_thread_passes_flag(self):
        from xccv.training.dataset_stats import DatasetStats
        from xccv.training.project import Project
        from xccv.training.trainer_thread import TrainThread

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        seen = {}

        class _FakeModel:
            def train(self, cfg, reporter):
                seen["cfg"] = cfg
                return {"best_val_map50": 0.0}

        p = Project(name="t", dataset_dir=tmp.name, model_dir=tmp.name,
                    task="segmentation", model_name="maskrcnn18",
                    anchor_small=True)
        th = TrainThread(p, Path(tmp.name) / "run")
        stats = DatasetStats(classes=["a"], n_images=1)
        with mock.patch("xccv.training.trainer_thread.collect_stats",
                        return_value=stats), \
                mock.patch("xccv.engine.models.create_model",
                           return_value=_FakeModel()):
            th.run()
        self.assertTrue(seen["cfg"].anchor_small)

    def test_anchor_pyramid_and_ckpt_roundtrip(self):
        from xccv.engine.models.rcnn import RCNNModel, _anchor_base

        self.assertEqual(_anchor_base(640), 8)
        self.assertEqual(_anchor_base(1280), 16)
        m = RCNNModel("maskrcnn18", "segmentation", 2, device="cpu",
                      use_pretrained=False)
        self.assertEqual(m.net.rpn.anchor_generator.sizes[0], (32,))
        net = m._build("maskrcnn18", 2, 320, False, anchor_base=16.0)
        want = tuple((16.0 * 2 ** i,) for i in range(5))
        self.assertEqual(tuple(net.rpn.anchor_generator.sizes), want)
        ckpt = {"format": "xccv-rcnn", "arch": "maskrcnn18",
                "task": "segmentation", "classes": ["a", "b"],
                "min_size": 320, "state_dict": net.state_dict(),
                "industrial_opt": False, "mask_head_light": False,
                "slim_head": False, "anchor_base": 16.0}
        m2 = RCNNModel("fasterrcnn", "detection", 1, device="cpu",
                       use_pretrained=False)
        m2.load_weights(ckpt)
        self.assertEqual(tuple(m2.net.rpn.anchor_generator.sizes), want)


class ProjectOrderTest(unittest.TestCase):
    """项目卡片按最近更新时间倒序（project.json 与各次训练 metrics.json）。"""

    def test_order_by_latest_touch(self):
        from tests.synthetic import make_labelme_dataset
        from xccv.training import project as project_mod

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ds, _ = make_labelme_dataset(Path(tmp.name) / "ds", n_images=1)
        with mock.patch.object(project_mod, "PROJECTS_DIR",
                               Path(tmp.name) / "projects"):
            a = project_mod.create_project("甲", str(ds), "", "detection")
            b = project_mod.create_project("乙", str(ds), "", "detection")
            os.utime(a.project_dir / "project.json", (1000, 1000))
            os.utime(b.project_dir / "project.json", (2000, 2000))
            self.assertEqual(project_mod.list_projects(), ["乙", "甲"])
            run = a.project_dir / "models" / "20261107_100000_maskrcnn18"
            run.mkdir(parents=True)
            (run / "metrics.json").write_text("{}", encoding="utf-8")
            os.utime(run / "metrics.json", (3000, 3000))
            self.assertEqual(project_mod.list_projects(), ["甲", "乙"])


from tests.test_ui_upgrade import _TrainEnv  # noqa: E402


class AnchorCheckboxTest(_TrainEnv):
    """训练窗口「小目标锚框自适应」复选框：加载/保存/持久化。"""

    def test_checkbox_persists(self):
        from xccv.training.project import load_project

        pw = self._window()
        self.assertFalse(pw.par_anchor.isChecked())
        pw.par_anchor.setChecked(True)
        pw._save_params()
        self.assertTrue(load_project("ui").anchor_small)
        pw2 = self._window()
        self.assertTrue(pw2.par_anchor.isChecked())


# ============================================================ 模型工具
class DefectgenPortTest(unittest.TestCase):
    """固定优先端口（IndexedDB 按 origin 分区，端口漂移=资产清空）；同进程
    多实例不得撞端口（探测释放后服务线程才监听的竞态）。"""

    def test_claims_distinct_and_release(self):
        code = ("from xccv.defectgen.defectgen_window import _pick_port, "
                "_release_port; a=_pick_port(); b=_pick_port(); "
                "_release_port(a); c=_pick_port(); print(a, b, c)")
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                           env=env, capture_output=True, text=True,
                           timeout=180)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        a, b, c = (int(x) for x in r.stdout.split()[-3:])
        self.assertTrue(39771 <= a < 39771 + 48)
        self.assertNotEqual(a, b)
        self.assertEqual(c, a)


class AccelUiTest(unittest.TestCase):
    """模型加速：源权重自动建议输入尺寸 + 导出进度条（忙碌/百分比）。"""

    def test_suggest_imgsz_variants(self):
        from xccv.modeltools.modeltools_window import _suggest_imgsz

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = Path(tmp.name)
        torch.save({"min_size": 512}, d / "a.pt")
        self.assertEqual(_suggest_imgsz(d / "a.pt"), 512)
        torch.save({"train_input": 0, "image_size": 640}, d / "b.pt")
        self.assertEqual(_suggest_imgsz(d / "b.pt"), 640)
        torch.save({"train_args": {"imgsz": 736}}, d / "c.pt")
        self.assertEqual(_suggest_imgsz(d / "c.pt"), 736)
        (d / "bad.pt").write_bytes(b"junk")
        self.assertIsNone(_suggest_imgsz(d / "bad.pt"))

    def test_progress_bar_states(self):
        from xccv.modeltools.modeltools_window import AccelerateToolWindow

        w = AccelerateToolWindow()
        self.addCleanup(w.close)
        w._on_export_progress(-1, "构建中")
        self.assertEqual((w.accel_bar.minimum(), w.accel_bar.maximum()),
                         (0, 0))
        w._on_export_progress(0.5, "一半")
        self.assertEqual(w.accel_bar.maximum(), 100)
        self.assertEqual(w.accel_bar.value(), 50)
        self.assertEqual(w.accel_result.text(), "一半")


# ============================================================ YOLO（anchor-free）
def _case(cx_off_px, stride=8.0):
    """微小目标 GT 7.5×5px，预测中心偏移 cx_off_px（特征尺度坐标）。"""
    tw, th = 7.5 / stride, 5.0 / stride
    tb = torch.tensor([[100.0 - tw / 2, 50.0 - th / 2,
                        100.0 + tw / 2, 50.0 + th / 2]])
    pb = tb.clone()
    pb[0, 0] += cx_off_px / stride
    pb[0, 2] += cx_off_px / stride
    pb.requires_grad_(True)
    pd = torch.full((1, 64), 8.0, requires_grad=True)
    return (pd, pb, torch.tensor([[100.0, 50.0]]), tb,
            torch.tensor([[[0.9]]]), torch.tensor(0.9),
            torch.tensor([True]), torch.tensor([960.0, 960.0]),
            torch.full((1, 1), stride))


class NWDLossTest(unittest.TestCase):
    """旧 Focaler-WIoU 以 GT 尺寸归一再 exp：微小目标 loss 1.5e6 且不重叠时
    框梯度为 0。新损失有界、恒有梯度。"""

    def setUp(self):
        from xccv.engine.models.modules import NWDCIoUBboxLoss

        self.loss = NWDCIoUBboxLoss(reg_max=16)

    def test_bounded_with_gradient_when_far(self):
        args = _case(60.0)
        li, ld = self.loss(*args)
        (li + ld).backward()
        self.assertLessEqual(float(li), 3.0)
        g = float(args[1].grad.abs().sum())
        self.assertTrue(0.0 < g < 1e3, g)

    def test_perfect_prediction_near_zero(self):
        li, _ = self.loss(*_case(0.0))
        self.assertLess(float(li), 0.05)

    def test_batched_2d_fg_mask(self):
        anchors = torch.tensor([[50.0, 50.0], [150.0, 50.0],
                                [50.0, 150.0], [150.0, 150.0]])
        stride = torch.tensor([[8.0], [8.0], [16.0], [16.0]])
        tb = torch.zeros(2, 4, 4)
        tb[0, 0] = torch.tensor([49.0, 49.4, 49.9, 50.0])
        tb[1, 1] = torch.tensor([149.0, 149.4, 149.9, 150.0])
        pb = tb.clone()
        pb[0, 0, [0, 2]] += 60 / 8
        pb[1, 1, [0, 2]] += 60 / 16
        pb.requires_grad_(True)
        ts = torch.zeros(2, 4, 1)
        ts[0, 0, 0] = ts[1, 1, 0] = 0.9
        fg = torch.tensor([[True, False, False, False],
                           [False, True, False, False]])
        li, ld = self.loss(torch.full((2, 4, 64), 8.0), pb, anchors, tb, ts,
                           torch.tensor(1.8), fg,
                           torch.tensor([960.0, 960.0]), stride)
        li.backward()
        self.assertTrue(torch.isfinite(li))
        self.assertLessEqual(float(li), 3.0)
        self.assertTrue(torch.isfinite(pb.grad).all())

    def test_yolo_industrial_patch_uses_nwd_only(self):
        from xccv.engine.models.modules import NWDCIoUBboxLoss
        from xccv.engine.models.yolo import YoloModel

        class _Net(torch.nn.Linear):
            def init_criterion(self):
                return SimpleNamespace(reg_max=16, bbox_loss="native",
                                       bce="native-bce")

        net = _Net(1, 1)
        me = SimpleNamespace(_loss_patch="off")
        YoloModel._apply_industrial_loss(me, SimpleNamespace(model=net))
        crit = net.init_criterion()
        self.assertIsInstance(crit.bbox_loss, NWDCIoUBboxLoss)
        self.assertEqual(crit.bce, "native-bce")      # cls 走原生
        self.assertTrue(me._loss_patch.startswith("applied(NWD"))


def _val_data():
    shp = SimpleNamespace(label="a",
                          points=[[10, 10], [30, 10], [30, 30], [10, 30]])
    smp = SimpleNamespace(width=100, height=100, image_path="a.jpg",
                          shapes=[shp])
    return SimpleNamespace(val_samples=[smp], classes=["a"])


class YoloLooseMetricTest(unittest.TestCase):
    """宽松口径：YOLO 路径此前不产出 map_loose（训练表恒 0）；逐轮评估复用
    验证期已 NMS 的检出（on_val_start 挂不上 trainer 回调）。"""

    def test_validator_wrap_collects_scaled_dets(self):
        from xccv.engine.models.yolo import YoloModel

        class _V:
            calls = 0

            def _prepare_pred(self, pred):
                return pred

            def _prepare_batch(self, si, batch):
                return {"im_file": batch["im_file"][si]}

            def scale_preds(self, predn, pbatch):
                return predn

            def update_metrics(self, preds, batch):
                _V.calls += 1

        v = _V()
        me = SimpleNamespace(_val_index={"a.jpg": 0, "ok.jpg": 1},
                             _epoch_dets=[], _det_seen=0)
        YoloModel._on_val_start(me, SimpleNamespace(validator=v))
        YoloModel._on_val_start(me, SimpleNamespace(validator=v))  # 不重复包
        preds = [{"bboxes": torch.tensor([[10.0, 10.0, 30.0, 30.0]]),
                  "conf": torch.tensor([0.9]), "cls": torch.tensor([0.0])},
                 # OK 负样本：无 GT 且无检出（此前整批检出被丢）
                 {"bboxes": torch.zeros((0, 4)), "conf": torch.zeros(0),
                  "cls": torch.zeros(0)}]
        batch = {"im_file": ["a.jpg", "ok.jpg"],
                 "img": torch.zeros((2, 3, 100, 100)),
                 "ori_shape": [(100, 100), (100, 100)],
                 "ratio_pad": [((1.0, 1.0), (0.0, 0.0)),
                               ((1.0, 1.0), (0.0, 0.0))]}
        v.update_metrics(preds, batch)
        self.assertEqual(_V.calls, 1)
        self.assertEqual(len(me._epoch_dets), 1)
        det = me._epoch_dets[0]
        self.assertEqual((det["image_id"], det["category_id"]), (0, 1))
        self.assertEqual(det["bbox"], [10.0, 10.0, 20.0, 20.0])
        YoloModel._on_epoch_dets_reset(me, None)
        self.assertEqual(me._epoch_dets, [])

    def test_epoch_loose_values(self):
        from xccv.engine.models.yolo import YoloModel

        hit = {"image_id": 0, "category_id": 1,
               "bbox": [10.0, 10.0, 20.0, 20.0], "score": 0.9}
        me = SimpleNamespace(_val_data=_val_data(), _epoch_dets=[hit])
        out = YoloModel._epoch_loose(me)
        self.assertAlmostEqual(out["map_loose"], 1.0, places=3)
        self.assertAlmostEqual(out["mar_loose"], 1.0, places=3)
        me._epoch_dets = []
        self.assertEqual(YoloModel._epoch_loose(me)["map_loose"], 0.0)

    def test_final_loose_with_predict_fn(self):
        from xccv.engine.models.yolo import YoloModel

        box = SimpleNamespace(xyxy=torch.tensor([[10.0, 10.0, 30.0, 30.0]]),
                              cls=torch.tensor(0.0), conf=torch.tensor(0.9))
        out = YoloModel._loose_metrics(
            SimpleNamespace(), _val_data(),
            predict_fn=lambda p: SimpleNamespace(boxes=[box]))
        self.assertAlmostEqual(out["map_loose"], 1.0, places=3)


if __name__ == "__main__":
    unittest.main()

