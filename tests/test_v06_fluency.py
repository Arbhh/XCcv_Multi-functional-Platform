"""全局流畅度治理（后台化/分桶/防抖/缓存）。"""

import os
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

from PyQt5.QtCore import Qt as _Qt

_USER_ROLE = _Qt.UserRole

from xccv.training.data_health import _dhash, _duplicate_groups

_ROOT = Path(__file__).resolve().parents[1]
_APP = None


def _app():
    global _APP
    if _APP is None:
        from PyQt5.QtWidgets import QApplication

        _APP = QApplication.instance() or QApplication([])
    return _APP


def _pump(ms: int):
    from PyQt5.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec_()


def _flush_deferred():
    """立即执行 deleteLater 队列（对齐 test_ui_upgrade 模式，防 offscreen 退出段错误）。"""
    from PyQt5.QtCore import QEvent

    _app().processEvents()
    _app().sendPostedEvents(None, QEvent.DeferredDelete)
    _app().processEvents()


class DataHealthBucketTest(unittest.TestCase):
    """分桶聚类与全量两两比对结果等价（同一判定谓词）。"""

    @staticmethod
    def _make_hashes(n_unique: int = 60, dup_clusters: int = 4) -> dict:
        rng = np.random.default_rng(7)
        hashes, i = {}, 0
        for c in range(dup_clusters):
            base = int(rng.integers(0, 2**64, dtype=np.uint64))
            for _k in range(3):  # 每簇 3 张近似图（距离 ≤4）
                noisy = base
                for _b in range(2):
                    noisy ^= 1 << int(rng.integers(0, 64))
                hashes[f"dup{c}_{i}.png"] = noisy
                i += 1
        for u in range(n_unique):
            hashes[f"u{u}.png"] = int(rng.integers(0, 2**64, dtype=np.uint64))
        return hashes

    def _groups_as_sets(self, groups):
        return sorted(frozenset(g) for g in groups)

    def test_bucket_equivalent_to_bruteforce(self):
        hashes = self._make_hashes()
        for threshold in (0, 2, 4, 6, 7):   # 分桶路径（<8）
            fast = _duplicate_groups(hashes, threshold)
            # 暴力参考实现（O(n²)，强制走 fallback 分支逻辑）
            paths = list(hashes)
            parent = {p: p for p in paths}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    if bin(hashes[paths[i]] ^ hashes[paths[j]]).count("1") <= threshold:
                        parent[find(paths[i])] = find(paths[j])
            ref = {}
            for p in paths:
                ref.setdefault(find(p), []).append(p)
            ref_groups = [g for g in ref.values() if len(g) > 1]
            self.assertEqual(self._groups_as_sets(fast),
                             self._groups_as_sets(ref_groups),
                             f"threshold={threshold} 分桶与全量结果不一致")

    def test_high_threshold_fallback_bruteforce(self):
        hashes = self._make_hashes()
        fast = _duplicate_groups(hashes, 8)     # ≥8 走回退分支
        brute = _duplicate_groups(hashes, 12)   # 必然全对（更大阈值更多组）
        self.assertIsInstance(fast, list)
        self.assertIsInstance(brute, list)
        self.assertLessEqual(len(fast), len(brute))

    def test_bucket_performance_on_thousand(self):
        """千级哈希分桶耗时显著低于全量两两（量级 sanity，不卡阈值）。"""
        hashes = {f"k{i}.png": (i * 2654435761) % (1 << 64)
                  for i in range(1200)}
        t0 = time.perf_counter()
        _duplicate_groups(hashes, 4)
        fast = time.perf_counter() - t0
        self.assertLess(fast, 20.0)  # 全量两两在千级要 >5s，分桶应远小于此


class WorkbenchAsyncTest(unittest.TestCase):
    """试跑后台化——提交后主线程立即可继续（线程在跑），完成后结果回填。"""

    def test_run_single_async_with_wait(self):
        _app()
        from xccv.workbench import OperatorWorkbench

        w = OperatorWorkbench()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "in.png"
            cv2.imwrite(str(p), np.full((64, 64), 90, np.uint8))
            w.run_image_edit.setText(str(p))
            for i in range(w.op_list.count()):
                if w.op_list.item(i).data(_USER_ROLE) == "xccv2/gaussian_blur":
                    w.op_list.setCurrentRow(i)
                    break
            w._run_single()
            self.assertIsNotNone(w._run_thread)   # 已在后台线程
            self.assertFalse(w.btn_run.isEnabled())  # 忙态禁用
            deadline = time.monotonic() + 30
            while w._run_thread is not None and time.monotonic() < deadline:
                _pump(20)
            _app().processEvents()
            self.assertIsNone(w._run_thread)
            self.assertTrue(w.btn_run.isEnabled())
            self.assertIn("2/2 节点成功", w.single_out.toPlainText())
            w.close()
            _flush_deferred()


class SearchDebounceTest(unittest.TestCase):
    """textChanged 走 300ms 防抖（一次输入风暴只触发一次过滤）。"""

    def test_debounce_batches_typing(self):
        from unittest import mock

        _app()
        from xccv.app import HomeWindow

        win = HomeWindow()
        self.addCleanup(win.close)
        self.addCleanup(_flush_deferred)
        with mock.patch.object(win, "_apply_search") as spied:
            for ch in "训练平台":
                win.search_edit.setText(win.search_edit.text() + ch)
                _pump(30)  # 每次输入间隔 30ms（<300ms 防抖窗口）
            self.assertEqual(spied.call_count, 0)  # 防抖窗口内不触发
            _pump(400)  # 越过防抖窗口
            self.assertEqual(spied.call_count, 1)  # 落定后恰好一次
        win._apply_search("")  # 清理界面态
        _flush_deferred()


class FlowTreeAsyncTest(unittest.TestCase):
    """目录扫描后台化 + 预览缓存命中。"""

    def _make_page(self, images: int = 120):
        _app()
        from xccv.flow_platform import FlowPage

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name) / "ds"
        root.mkdir()
        for i in range(images):
            cv2.imwrite(str(root / f"im{i:04d}.png"),
                        np.full((24, 32, 3), i % 255, np.uint8))
        page = FlowPage()
        self.addCleanup(page.close)
        self.addCleanup(_flush_deferred)
        page.data_edit.setText(str(root))
        page._load_data_root()
        deadline = time.monotonic() + 60
        while page._scan_thread is not None and time.monotonic() < deadline:
            _pump(20)
        _app().processEvents()
        return page, root

    def test_tree_filled_async(self):
        page, _root = self._make_page(120)
        self.assertIn("120/120", page.count_label.text())
        self.assertGreater(page.tree.topLevelItemCount(), 0)

    def test_preview_cache_hit(self):
        from unittest import mock

        page, root = self._make_page(3)
        # 找第一张图片项（可能直接在隐形根下，也可能在目录项下）
        img_item = None
        roots = [page.tree.topLevelItem(i)
                 for i in range(page.tree.topLevelItemCount())]
        for it in roots:
            kind, _val = it.data(0, _USER_ROLE) or ("", "")
            if kind == "img":
                img_item = it
                break
            for i in range(it.childCount()):
                kind, _val = it.child(i).data(0, _USER_ROLE) or ("", "")
                if kind == "img":
                    img_item = it.child(i)
                    break
            if img_item is not None:
                break
        self.assertIsNotNone(img_item)
        page.tree.setCurrentItem(img_item)
        with mock.patch("xccv.flow_platform.imread_unicode", wraps=__import__(
                "xccv.flow_platform", fromlist=["imread_unicode"]).imread_unicode) as spied:
            page._on_tree_click(img_item, 0)
            first_calls = spied.call_count
            page._on_tree_click(img_item, 0)  # 第二次点同一张：缓存命中
            self.assertEqual(spied.call_count, first_calls)  # 不再解码
        _app().processEvents()


if __name__ == "__main__":
    unittest.main()

