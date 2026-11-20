"""槽包装测试的样例模块（test_qt_slots 的 apply_auto_slot_module 扫描对象）。"""

from PyQt5.QtCore import QObject

SAMPLE_LOG = []


class Sample(QObject):
    def do_it(self):
        SAMPLE_LOG.append("done")
