"""tests/modules — 按模块组织的行为锁（0.7.5 P-0 测试矩阵重组）。

与按版本堆积的历史回归锁（tests/test_v04x…test_v07x，保留不动）互补，
本包按"模块 × 行为"组织五大模块的冒烟级行为锁，作为门禁 B 段：

- test_mod_engine_models   engine/models 模型注册元数据 + 权重容器契约
- test_mod_flow_engine     流程引擎（算子注册表 / 执行器 / IR）
- test_mod_service_api     推理服务 HTTP 契约（健康检查 / Key 认证）
- test_mod_testing_review  测试平台复盘（分组口径 / 阈值面板 / 渲染窗）
- test_mod_training_ui     训练平台 UI（Quick 双轨 / 预设 / 数据体检）

口径：行为级断言（用户可观察的行为契约），不复述历史锁的实现细节；
新增行为先落这里，版本锁只留给当轮 bug 修复的精确回归。
"""

