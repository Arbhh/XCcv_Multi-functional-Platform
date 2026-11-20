"""XCcv 推理服务（C# 上位机 HTTP 接口，进程外服务化）。

纯引擎层消费者：所有模型加载走 engine.models.registry.load_model 统一链路，
禁止 import 任何 PyQt/UI 模块——必须无头可运行（NSSM 注册为 Windows 服务）。

安全铁律：
- 权重文件只在服务进程内读取，任何端点不返回权重字节；
- 默认仅监听 127.0.0.1，跨机部署需显式配置 host 并配置路径白名单；
- 所有业务端点要求 API Key（X-XCcv-Key 头），/v1/health 除外（监控用）；
- 响应体 = result_json 统一 JSON（schema_version=1），与平台批量落盘文件同构。
"""
