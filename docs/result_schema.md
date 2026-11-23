# XCcv 统一检出结果 JSON 契约（schema_version = 1）

> 适用范围：测试平台批量、流程图平台批量、部署平台推理服务（HTTP API）——
> 三条链路的 JSON 由 `xccv/engine/result_json.py` 单一实现产出，**格式完全统一**。
> 消费方：C# 上位机（Newtonsoft.Json / System.Web.Extensions 直读）。
> 演进规则：**只增不改**——新字段只允许追加，不修改既有字段语义；v1 字段永久兼容。

## 一、文件形态

- 编码 UTF-8、`ensure_ascii=False`（中文类别名直接可读）、缩进 1；
- 键全 ASCII snake_case；值为纯 JSON 基元（str/int/float/bool/null/array/object）；
- 每图一个同名 `.json`（批量落盘镜像输入相对目录）；HTTP 接口响应体即本文档结构。

## 二、顶层结构

```json
{
 "schema_version": 1,
 "task": "detection",
 "source":  {"file": "斑点.png", "path": "E:\\...\\斑点.png",
             "width": 531, "height": 230},
 "model":   {"file": "best.pt", "name": "斑点分割"},
 "inference": {"conf": 0.25, "iou": 0.45,
               "flow_applied": true, "flow_removed": 1,
               "count_check": "OK", "flow_name": "", "timestamp": "2026-09-19 15:12:03"},
 "detections": [ {...}, ... ],
 "classifications": [ {...}, ... ],
 "geometries":   [ {...}, ... ],
 "measurements": [ {...}, ... ],
 "verdict": {"passed": true, "source": "count_check", "detail": "数量判定 OK"},
 "error": null
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| schema_version | int | 恒为 1（消费端校验用） |
| task | str | `"detection"` / `"segmentation"` / `"classification"` / `"flow"`（传统流程）/ `"anomaly"`（无监督异常检测，V0.7 增）/ `"ocr"`（字符识别，V0.7 增） |
| source.file | str | 图文件名（HTTP 上传名经 RFC2047 解码） |
| source.path | str | 本地路径（服务上传模式为空串；流程模式为服务端临时路径） |
| source.width/height | int/null | 图像像素尺寸 |
| model.file | str | 权重文件名；流程模式恒空串 |
| model.name | str | 业务别名（部署平台「模型加载」中设置）；流程模式恒空串 |
| inference.conf / iou | float/null | 推理参数回显 |
| inference.flow_applied | bool | 是否应用了检出过滤 |
| inference.flow_removed | int | 过滤移除的检出数 |
| inference.count_check | str/null | `"OK"` / `"NG"`（旧链数量判定结果） |
| inference.flow_name | str | 流程文件名（仅 task=flow） |
| inference.timestamp | str | 本地时间 `YYYY-MM-DD HH:MM:SS` |
| error | str/null | 单图失败原因；正常为 null |

## 三、detections 数组（检测 / 分割）

```json
{"label": "缺陷", "label_id": 0, "score": 0.9312,
 "bbox": [101.5, 88.0, 190.25, 176.75], "bbox_area": 7792,
 "mask_area": 5120}
```

| 字段 | 说明 |
|---|---|
| label / label_id | 类别名（类别表序号；无类别表时 label 为序号字符串） |
| score | 置信度，4 位小数 |
| bbox | `[x1, y1, x2, y2]` 像素，2 位小数 |
| bbox_area | 边界框面积（像素²，整数） |
| mask_area | 分割掩码面积；仅分割任务输出 |
| text | 识别文本；**仅 OCR 任务输出**（V0.7 增，`task="ocr"` 的 detections 逐框透出） |

## 四、classifications 数组（分类任务）

`[{"label": "NG", "label_id": 1, "score": 0.9312}]` —— 按得分降序 Top-5；
检测/分割任务该数组为空数组（恒存在，消费端零分支）。

## 五、geometries / measurements（仅 task=flow）

```json
"geometries":   [{"kind": "circle", "data": [99.51, 166.50, 12.95],
                  "subpixel": true}],
"measurements": [{"values": [88.7, 175.4], "unit": "px", "passed": true}]
```

- `geometries.kind`：`point`(N×2) / `line`(2×2) / `circle`(3: cx,cy,r) /
  `polygon`(N×2) / `rotated_rect`(5: cx,cy,w,h,angle)；
- `subpixel` 仅在 true 时输出（默认噪声剔除）；
- `measurements`：value 或 values 二选一，`unit` 仅非 px 时输出，`passed` 可空；
- 空壳条目（无 data / 无 value+values）整条剔除；NaN/None 不输出。

## 六、verdict（v0.5 增补，**key 恒存在**；无判定 = null）

```json
"verdict": {"passed": true, "source": "count_check", "detail": "数量判定 OK"}
```

| 字段 | 说明 |
|---|---|
| passed | bool：整体判定（多判定并存取 AND，任一 NG 即不通过） |
| source | `"count_check"`（数量判定节点/旧链检出流程）或 `"logic"`（数值比较/结果聚合） |
| detail | 人话说明（列出未通过节点） |

判定来源：①AI 检出 + 检出流程 → `inference.count_check` 自动映射；
②传统流程 → 扫描全部成功节点的 bool passed 输出（filter_count_check /
compare_value / result_merge）。**无判定节点时 verdict 为 null**（非错误）。

## 七、C# 消费示例（.NET Framework 4.5+，无第三方依赖）

```csharp
using System.Web.Script.Serialization;   // 引用 System.Web.Extensions.dll
var ser = new JavaScriptSerializer();
dynamic r = ser.Deserialize<Dictionary<string, object>>(File.ReadAllText(jsonPath));
Console.WriteLine(r["schema_version"]);
var verdict = r["verdict"] as Dictionary<string, object>;   // 可能为 null
bool ok = verdict != null && (bool)verdict["passed"];
foreach (Dictionary<string, object> det in (r["detections"] as IEnumerable
                                            ?? new object[0]))
    Console.WriteLine(det["label"] + " " + det["score"] + " " + det["bbox"]);
```

或直接使用部署平台随包的 `XCcv.Vision.Sdk.cs`（XCcvClient 自动解析与异常映射，
Demo 见 platform_data/csharp_demo/XCcvCSharpDemo.exe）。

## 八、HTTP API（部署平台推理服务，127.0.0.1:8600，响应体=本文档结构）

| 端点 | 方法 | 说明 |
|---|---|---|
| /v1/health | GET | 免 Key；探活 |
| /v1/models/load | POST | 加载 .pt/.engine 明文产物（模型常驻内存，永不返回权重文件） |
| /v1/models/unload · /status | POST/GET | 卸载 / 已加载列表 |
| /v1/infer/image · /infer/batch | POST | multipart 图片 → 统一 JSON |
| /v1/flow/run | POST | multipart flow(.json/.xvap)+image → 流程 JSON（含 verdict） |
| /v1/render/overlay | POST | 推理+叠加 → PNG 字节 |
| /docs · /openapi.json | GET | 交互文档 / 契约导出（服务运行时） |

鉴权：除 health 外全部要求请求头 `X-XCcv-Key`（Key 见部署平台页）。
错误统一 `{"code", "message", "detail"}`。

## 九、V0.6 增补（2026-11-03，只增不改——上文全部字段与语义保持不变）

本节只补充 V0.6 流程算子对既有字段的**取值来源**说明，未新增/修改任何 JSON 字段。

1. **`detections[].label` 可能经流程「标签修改」(relabel_map) 改名**：label 为改名后的
   名字；`label_id` 指向扩展后的类别表——原类别序号与名字不变，新名字追加在表尾
   （同一模型的其它分支不受影响）。原类别名只保留在流程内部（不进 JSON）。
2. **级联流程（检测 → 按检出裁图 → 分割，模板 `tmpl_cascade_det_seg`）**：`source.width/height`
   取自接入「保存结果」image 端口的图像（模板中为裁图）；该份 JSON 的 bbox/掩码坐标**相对裁图**。
   裁图在原图中的位置由流程内图像 meta `roi_rect=[x1,y1,x2,y2]` 给出（据此把 bbox/掩码
   坐标回映到原图坐标系）。
3. **`inference.flow_removed`** 口径不变（过滤移除数）。被滤掉的检出在流程内可经各过滤算子新增的
   `removed` 输出端口继续处理（如接另一个「保存结果」单独导出），默认不进 JSON。
4. **区域过滤参数**（屏蔽区域 / 保留区域）新增形状语法 `rect:` / `circle:cx,cy,r` /
   `polygon:x1,y1,…`（无前缀 = 旧矩形写法，零改动兼容），只影响过滤结果，不影响 JSON 结构。
5. **新增落盘形态（非本 JSON）**：
   - 「保存表格」(csv_sink)：CSV，utf-8-sig 编码（Excel 直接打开中文不乱码）；可选列
     `source,time,index,label,label_id,score,x1,y1,x2,y2,bbox_area,mask_area,value,unit,passed`，
     列值口径与本文档 detections/measurements 同源；追加写入时表头不一致自动把旧文件备份为
     `*.bak_时间戳.csv` 后重建。
   - 「保存图像」(image_sink)：`<来源图名><后缀>.png|jpg|bmp`，overlay 模式叠加检出框蒙版。
   - 「按检出裁图」/「网格分块」设置落盘目录时：`<前缀>_<来源图名>_<序号>.png` /
     `<前缀>_<来源图名>_r<行>c<列>.png`。
