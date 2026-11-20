// XCcv.Vision.Sdk.cs — XCcv 推理服务 C# 薄客户端 SDK（单文件，无第三方依赖）
//
// 目标框架：.NET Framework 4.5+（与常见工业上位机 Demo 同基线；C# 5 语法，
// 系统自带 csc.exe 可直接编译）。仅做 HTTP 封装——全部推理在 XCcv 推理服务
// 进程内完成，权重文件永不离开服务端，客户端只见 JSON。
//
// 响应契约：统一检出 JSON（schema_version=1），字段说明见 XCcv 安装目录
// docs/result_schema.md。错误统一 {code, message, detail} → XccvException。
//
// 用法示例：
//   using (var client = new XccvClient("http://127.0.0.1:8600", "你的APIKey")) {
//       ModelHandle m = await client.LoadModelAsync(@"C:\XCcv\models\best.pt");
//       string json = await client.InferImageJsonAsync(m.ModelId, imageBytes, "a.png");
//       Console.WriteLine(json);
//   }
//
// 需引用程序集：System、System.Core、System.Net.Http、System.Web.Extensions。

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;
using System.Web.Script.Serialization;

namespace XCcv.Vision
{
    /// <summary>服务端错误（HTTP 非 2xx）。</summary>
    public class XccvException : Exception
    {
        public string Code { get; private set; }
        public string Detail { get; private set; }
        public int HttpStatus { get; private set; }

        public XccvException(int status, string code, string message, string detail)
            : base(message)
        {
            HttpStatus = status;
            Code = code;
            Detail = detail;
        }
    }

    /// <summary>已加载模型句柄（轻量；ModelId 供推理端点寻址）。</summary>
    public class ModelHandle
    {
        public string ModelId { get; internal set; }
        public string Alias { get; internal set; }
        public string Task { get; internal set; }
        public string Device { get; internal set; }
    }

    /// <summary>XCcv 推理服务客户端（薄封装；线程安全可复用）。</summary>
    public class XccvClient : IDisposable
    {
        // 共享静态实例：每客户端 new HttpClient 会耗尽 socket（TIME_WAIT）
        private static readonly HttpClient _sharedHttp = new HttpClient();
        private readonly HttpClient _http;
        private readonly JavaScriptSerializer _json = new JavaScriptSerializer();

        public string BaseUrl { get; private set; }

        /// <param name="baseUrl">如 http://127.0.0.1:8600</param>
        /// <param name="apiKey">部署平台「推理服务」面板可查看/复制</param>
        public XccvClient(string baseUrl, string apiKey)
            : this(baseUrl, apiKey, TimeSpan.FromSeconds(180))
        {
        }

        public XccvClient(string baseUrl, string apiKey, TimeSpan timeout)
        {
            BaseUrl = baseUrl.TrimEnd('/');
            _http = _sharedHttp;
            _sharedHttp.Timeout = timeout;
            if (!string.IsNullOrEmpty(apiKey))
                _http.DefaultRequestHeaders.Add("X-XCcv-Key", apiKey);
        }

        // ---- 探活 / 状态（免 Key / 需 Key 各一） ----

        public async Task<string> HealthJsonAsync()
        {
            return await GetStringAsync("/v1/health");
        }

        public async Task<string> StatusJsonAsync()
        {
            return await GetStringAsync("/v1/models/status");
        }

        /// <summary>已加载流程列表 JSON 原文（{"flows":[{flow_id, alias, ...}]}）。</summary>
        public async Task<string> FlowsStatusJsonAsync()
        {
            return await GetStringAsync("/v1/flows/status");
        }

        // ---- 模型生命周期 ----

        public async Task<ModelHandle> LoadModelAsync(string modelPath,
            string device = "auto", string alias = "", bool warmup = false)
        {
            // /v1/models/load 为 JSON 体（请求 schema 见 docs/result_schema.md）
            var payload = _json.Serialize(new Dictionary<string, object>
            {
                { "path", modelPath },
                { "device", device ?? "auto" },
                { "alias", alias ?? "" },
                { "warmup", warmup }
            });
            using (var content =
                new StringContent(payload, Encoding.UTF8, "application/json"))
            using (var resp = await _http.PostAsync(BaseUrl + "/v1/models/load", content))
            {
                string body = await resp.Content.ReadAsStringAsync();
                EnsureOk(resp, body);
                var d = ParseDict(body);
                return new ModelHandle
                {
                    ModelId = Str(d, "model_id"),
                    Alias = Str(d, "alias"),
                    Task = Str(d, "task"),
                    Device = Str(d, "device")
                };
            }
        }

        public async Task<bool> UnloadModelAsync(string modelId)
        {
            var payload = _json.Serialize(new Dictionary<string, object>
            {
                { "model_id", modelId }
            });
            using (var content =
                new StringContent(payload, Encoding.UTF8, "application/json"))
            using (var resp = await _http.PostAsync(BaseUrl + "/v1/models/unload", content))
            {
                string body = await resp.Content.ReadAsStringAsync();
                EnsureOk(resp, body);
                return true;
            }
        }

        // ---- 流程常驻 API（加载一次、多次推理；流程内模型首次推理自动加载） ----

        /// <summary>加载流程包（.xvap/.json）为常驻流程 API。</summary>
        public async Task<ModelHandle> LoadFlowAsync(string flowPath, string alias = "")
        {
            var payload = _json.Serialize(new Dictionary<string, object>
            {
                { "path", flowPath },
                { "alias", alias ?? "" }
            });
            using (var content =
                new StringContent(payload, Encoding.UTF8, "application/json"))
            using (var resp = await _http.PostAsync(BaseUrl + "/v1/flows/load", content))
            {
                string body = await resp.Content.ReadAsStringAsync();
                EnsureOk(resp, body);
                var d = ParseDict(body);
                return new ModelHandle
                {
                    ModelId = Str(d, "flow_id"),
                    Alias = Str(d, "alias"),
                    Task = "flow",
                    Device = ""
                };
            }
        }

        /// <summary>运行已加载流程（multipart 图片）→ 统一流程 JSON 原文。</summary>
        public async Task<string> RunFlowByIdJsonAsync(string flowId,
            byte[] image, string fileName)
        {
            var resp = await PostMultipartAsync("/v1/flows/run", flowId, "flow_id",
                null, null,
                new[] { new KeyValuePair<string, byte[]>("image", image) },
                new[] { new KeyValuePair<string, string>("image", fileName ?? "image.png") },
                null, null);
            using (resp)
            {
                string body = await resp.Content.ReadAsStringAsync();
                EnsureOk(resp, body);
                return body;
            }
        }

        /// <summary>运行已加载流程 + 叠加图 → PNG 字节（供界面预览）。</summary>
        public async Task<byte[]> RunFlowByIdOverlayPngAsync(string flowId,
            byte[] image, string fileName)
        {
            var resp = await PostMultipartAsync("/v1/flows/run", flowId, "flow_id",
                null, null,
                new[] { new KeyValuePair<string, byte[]>("image", image) },
                new[] { new KeyValuePair<string, string>("image", fileName ?? "image.png") },
                null, new List<KeyValuePair<string, string>>
                {
                    new KeyValuePair<string, string>("render", "true")
                });
            using (resp)
            {
                EnsureOk(resp, await resp.Content.ReadAsStringAsync());
                return await resp.Content.ReadAsByteArrayAsync();
            }
        }

        /// <summary>卸载已加载流程。</summary>
        public async Task<bool> UnloadFlowAsync(string flowId)
        {
            var payload = _json.Serialize(new Dictionary<string, object>
            {
                { "model_id", flowId }
            });
            using (var content =
                new StringContent(payload, Encoding.UTF8, "application/json"))
            using (var resp = await _http.PostAsync(BaseUrl + "/v1/flows/unload", content))
            {
                string body = await resp.Content.ReadAsStringAsync();
                EnsureOk(resp, body);
                return true;
            }
        }

        // ---- 推理（multipart 原始图片字节；拒绝 base64——性能契约） ----

        /// <summary>单帧推理 → 统一检出 JSON 原文（schema_version=1）。</summary>
        public async Task<string> InferImageJsonAsync(string modelId,
            byte[] image, string fileName, float? conf = null, float? iou = null)
        {
            var resp = await PostMultipartAsync("/v1/infer/image", modelId, conf, iou,
                new[] { new KeyValuePair<string, byte[]>("image", image) },
                new[] { new KeyValuePair<string, string>("image", fileName ?? "image") },
                null, null);
            using (resp)
            {
                string body = await resp.Content.ReadAsStringAsync();
                EnsureOk(resp, body);
                return body;
            }
        }

        /// <summary>批量推理 → JSON 数组原文（顺序与上传一致）。</summary>
        public async Task<string> InferBatchJsonAsync(string modelId,
            IList<byte[]> images, IList<string> fileNames,
            float? conf = null, float? iou = null)
        {
            var resp = await PostMultipartAsync("/v1/infer/batch", modelId, conf, iou,
                images.Select((b, i) => new KeyValuePair<string, byte[]>("images", b)).ToList(),
                fileNames.Select((n, i) => new KeyValuePair<string, string>("images", n)).ToList(),
                null, null);
            using (resp)
            {
                string body = await resp.Content.ReadAsStringAsync();
                EnsureOk(resp, body);
                return body;
            }
        }

        /// <summary>流程推理（.json 流程图 / .xvap 流程包字节 + 图片）→ 统一流程 JSON。</summary>
        public async Task<string> RunFlowJsonAsync(byte[] flow, string flowName,
            byte[] image, string imageName)
        {
            var resp = await PostMultipartAsync("/v1/flow/run", null, null, null,
                new List<KeyValuePair<string, byte[]>>
                {
                    new KeyValuePair<string, byte[]>("flow", flow),
                    new KeyValuePair<string, byte[]>("image", image)
                },
                new List<KeyValuePair<string, string>>
                {
                    new KeyValuePair<string, string>("flow", flowName ?? "flow.json"),
                    new KeyValuePair<string, string>("image", imageName ?? "image.png")
                }, null, null);
            using (resp)
            {
                string body = await resp.Content.ReadAsStringAsync();
                EnsureOk(resp, body);
                return body;
            }
        }

        /// <summary>推理 + 结果叠加图 → PNG 字节（供上位机直接显示）。</summary>
        public async Task<byte[]> RenderOverlayPngAsync(string modelId,
            byte[] image, string fileName, float? conf = null, float? iou = null)
        {
            var resp = await PostMultipartAsync("/v1/render/overlay", modelId, conf, iou,
                new[] { new KeyValuePair<string, byte[]>("image", image) },
                new[] { new KeyValuePair<string, string>("image", fileName ?? "image") },
                null, null);
            using (resp)
            {
                EnsureOk(resp, await resp.Content.ReadAsStringAsync());
                return await resp.Content.ReadAsByteArrayAsync();
            }
        }

        // ---- JSON 工具（契约解析辅助） ----

        /// <summary>JSON 原文 → 字典（结果 JSON 的顶层字段直接取用）。</summary>
        public Dictionary<string, object> Parse(string json)
        {
            return ParseDict(json);
        }

        // ---- 内部 ----

        private async Task<HttpResponseMessage> PostMultipartAsync(string path,
            string modelId, float? conf, float? iou,
            IList<KeyValuePair<string, byte[]>> files,
            IList<KeyValuePair<string, string>> fileNames,
            IList<KeyValuePair<string, string>> extraText,
            IList<KeyValuePair<string, string>> extraFlags)
        {
            return await PostMultipartAsync(path, modelId, "model_id", conf, iou,
                files, fileNames, extraText, extraFlags);
        }

        private async Task<HttpResponseMessage> PostMultipartAsync(string path,
            string modelId, string idFieldName, float? conf, float? iou,
            IList<KeyValuePair<string, byte[]>> files,
            IList<KeyValuePair<string, string>> fileNames,
            IList<KeyValuePair<string, string>> extraText,
            IList<KeyValuePair<string, string>> extraFlags)
        {
            var form = new MultipartFormDataContent();
            if (modelId != null)
                form.Add(new StringContent(modelId), idFieldName);
            if (conf.HasValue)
                form.Add(new StringContent(conf.Value.ToString(
                    System.Globalization.CultureInfo.InvariantCulture)), "conf");
            if (iou.HasValue)
                form.Add(new StringContent(iou.Value.ToString(
                    System.Globalization.CultureInfo.InvariantCulture)), "iou");
            if (extraText != null)
                foreach (var kv in extraText)
                    form.Add(new StringContent(kv.Value), kv.Key);
            if (extraFlags != null)
                foreach (var kv in extraFlags)
                    form.Add(new StringContent(kv.Value), kv.Key);
            for (int i = 0; i < files.Count; i++)
            {
                var bytesContent = new ByteArrayContent(files[i].Value);
                bytesContent.Headers.ContentType =
                    new MediaTypeHeaderValue("application/octet-stream");
                string name = files[i].Key;
                string file = (fileNames != null && i < fileNames.Count)
                    ? fileNames[i].Value : "upload.bin";
                form.Add(bytesContent, name, file);
            }
            return await _http.PostAsync(BaseUrl + path, form);
        }

        private async Task<string> GetStringAsync(string path)
        {
            using (var resp = await _http.GetAsync(BaseUrl + path))
            {
                string body = await resp.Content.ReadAsStringAsync();
                EnsureOk(resp, body);
                return body;
            }
        }

        private void EnsureOk(HttpResponseMessage resp, string body)
        {
            if (resp.IsSuccessStatusCode)
                return;
            string code = "error_" + (int)resp.StatusCode;
            string message = body;
            string detail = "";
            try
            {
                var d = ParseDict(body);
                if (d.ContainsKey("code")) code = Str(d, "code");
                if (d.ContainsKey("message")) message = Str(d, "message");
                if (d.ContainsKey("detail")) detail = Str(d, "detail");
            }
            catch
            {
                // 非 JSON 错误体（网关等）原样抛出
            }
            throw new XccvException((int)resp.StatusCode, code, message, detail);
        }

        private Dictionary<string, object> ParseDict(string json)
        {
            return _json.Deserialize<Dictionary<string, object>>(json);
        }

        private static string Str(Dictionary<string, object> d, string key)
        {
            object v;
            if (d.TryGetValue(key, out v) && v != null)
                return v.ToString();
            return "";
        }

        public void Dispose()
        {
            _http.Dispose();
        }
    }
}
