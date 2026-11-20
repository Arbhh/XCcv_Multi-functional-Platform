"""训练平台窗口页级拆分包（0.7.6 P-1）。

- common：模块级共享件（助手/控件类/预检线程/项目卡）
- info_page：页 0「项目信息」页（InfoPageMixin）
- quick_panel：Quick/专家双轨参数面板（QuickPanelMixin）
- train_page：页 1「训练可视化」页与训练运行控制（TrainPageMixin）
- history_view：训练历史视图（HistoryViewMixin）

组合根 ProjectWindow 在 training_window.py 经 mixin 继承装配；
页级结构由 tests/modules/snapshots/training_pages.json 快照锁核对。
"""
