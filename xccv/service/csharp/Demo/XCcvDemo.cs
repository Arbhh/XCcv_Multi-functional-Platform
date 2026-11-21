// XCcvDemo.cs — XCcv 推理服务 C# 测试程序（WinForms，参考常见工业上位机 Demo 形态）。
// 布局：顶部工具区 + 左侧「运行结果」（摘要 + 逐条检出
// + JSON 原文）+ 右侧图像预览（推理叠加图直接可视化）。
//
// C# 5 语法（系统自带 csc.exe 可编译）；启动参数：--url <base> --key <apiKey>
// --lang zh|en（部署平台「C# 测试程序」卡片自动传入；缺省中文，
// 界面语言与平台设置联动）。

using System;
using System.Collections;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Web.Script.Serialization;
using System.Windows.Forms;
using XCcv.Vision;

namespace XCcvDemo
{
    internal static class Program
    {
        [STAThread]
        private static void Main(string[] args)
        {
            string url = "http://127.0.0.1:8600";
            string key = "";
            string lang = "zh";
            for (int i = 0; i < args.Length - 1; i++)
            {
                if (args[i] == "--url") url = args[i + 1];
                if (args[i] == "--key") key = args[i + 1];
                if (args[i] == "--lang") lang = args[i + 1];
            }
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new MainForm(url, key, lang));
        }
    }

    internal class MainForm : Form
    {
        private readonly TextBox _urlBox = new TextBox();
        private readonly TextBox _keyBox = new TextBox();
        private readonly ComboBox _apiCombo = new ComboBox();      // 模型+流程混合
        private readonly TextBox _confBox = new TextBox();
        private readonly TextBox _iouBox = new TextBox();
        private readonly TextBox _jsonBox = new TextBox();          // JSON 原文
        private readonly TextBox _summaryBox = new TextBox();       // 摘要 + 逐条检出
        private readonly PictureBox _preview = new PictureBox();    // 右侧图像预览
        private Image _currentImage;                                 // 当前预览（负责释放）
        private XccvClient _client;
        private string _clientUrl = "";
        private string _clientKey = "";
        private readonly string _lang;

        /// <summary>双语取词：界面语言 zh（缺省）/ en，随平台设置经 --lang 传入。</summary>
        private string T(string zh, string en)
        {
            return _lang == "en" ? en : zh;
        }

        public MainForm(string url, string key, string lang)
        {
            _lang = lang == "en" ? "en" : "zh";
            Text = T("XCcv C# 测试程序 v1.2（推理服务对接 Demo）",
                     "XCcv C# Test v1.2 (Inference Service Demo)");
            Size = new Size(1360, 880);
            MinimumSize = new Size(1100, 700);
            StartPosition = FormStartPosition.CenterScreen;

            // 窗体/任务栏图标显式加载（exe 内嵌 win32icon 受 Windows 图标
            // 缓存影响，运行期从 exe 同目录的 .ico 直读保证立即生效）
            try
            {
                string icoPath = Path.Combine(Application.StartupPath, "XCcvDemo.ico");
                if (File.Exists(icoPath)) Icon = new Icon(icoPath);
            }
            catch { }   // 图标缺失不阻断启动

            var root = new TableLayoutPanel
            {
                Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 3,
                Padding = new Padding(6)
            };
            root.RowStyles.Add(new RowStyle(SizeType.AutoSize));          // 工具行（紧凑）
            root.RowStyles.Add(new RowStyle(SizeType.AutoSize));          // 操作行
            root.RowStyles.Add(new RowStyle(SizeType.Percent, 100));      // 结果区

            // 三条独立工具行（FlowLayoutPanel AutoSize 串联，行高紧凑不留空）
            var row1 = new FlowLayoutPanel
            { Dock = DockStyle.Top, AutoSize = true, WrapContents = false };
            row1.Controls.Add(Lbl(T("服务地址 ", "Service URL ")));
            _urlBox.Width = 200; _urlBox.Text = url; row1.Controls.Add(_urlBox);
            row1.Controls.Add(Lbl("  API Key "));
            _keyBox.Width = 270; _keyBox.Text = key; row1.Controls.Add(_keyBox);
            row1.Controls.Add(MakeButton(T("健康检查", "Health Check"), btnHealth_Click));
            root.Controls.Add(row1, 0, 0);

            var row2 = new FlowLayoutPanel
            { Dock = DockStyle.Top, AutoSize = true, WrapContents = false };
            row2.Controls.Add(Lbl(T("已加载 API ", "Loaded APIs ")));
            _apiCombo.Width = 300; _apiCombo.DropDownStyle = ComboBoxStyle.DropDownList;
            row2.Controls.Add(_apiCombo);
            row2.Controls.Add(MakeButton(T("刷新", "Refresh"), btnRefresh_Click));
            row2.Controls.Add(MakeButton(T("加载模型…", "Load Model…"), btnLoadModel_Click));
            row2.Controls.Add(MakeButton(
                T("加载流程…(.xvap/.json)", "Load Flow…(.xvap/.json)"), btnLoadFlow_Click));
            row2.Controls.Add(MakeButton(T("卸载所选", "Unload Selected"), btnUnload_Click));
            row2.Controls.Add(Lbl("  conf "));
            _confBox.Width = 46; row2.Controls.Add(_confBox);
            row2.Controls.Add(Lbl(" iou "));
            _iouBox.Width = 46; row2.Controls.Add(_iouBox);
            root.Controls.Add(row2, 0, 1);

            var actRow = new FlowLayoutPanel
            { Dock = DockStyle.Top, AutoSize = true, WrapContents = false };
            var btnInfer = MakeButton(T("单帧推理（模型）", "Infer (Model)"), btnInfer_Click);
            btnInfer.BackColor = Color.FromArgb(210, 228, 252);
            actRow.Controls.Add(btnInfer);
            var btnFlow = MakeButton(T("流程推理（流程）", "Run Flow"), btnFlowRun_Click);
            btnFlow.BackColor = Color.FromArgb(210, 228, 252);
            actRow.Controls.Add(btnFlow);
            actRow.Controls.Add(MakeButton(T("保存预览图…", "Save Preview…"), btnSaveImage_Click));
            actRow.Controls.Add(MakeButton(T("清空输出", "Clear Output"),
                delegate { _summaryBox.Clear(); _jsonBox.Clear(); }));
            root.Controls.Add(actRow, 0, 2);

            // ---- 结果区：左摘要+JSON，右图像预览（占满剩余全部高度） ----
            var split = new SplitContainer
            {
                Dock = DockStyle.Fill, SplitterDistance = 520,
                Orientation = Orientation.Vertical
            };
            var left = new TableLayoutPanel
            {
                Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 4
            };
            left.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            left.RowStyles.Add(new RowStyle(SizeType.Percent, 34));
            left.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            left.RowStyles.Add(new RowStyle(SizeType.Percent, 66));
            left.Controls.Add(Lbl(T("运行结果", "Run Result")), 0, 0);
            _summaryBox.Dock = DockStyle.Fill; _summaryBox.Multiline = true;
            _summaryBox.ReadOnly = true; _summaryBox.ScrollBars = ScrollBars.Vertical;
            _summaryBox.Font = new Font(FontFamily.GenericMonospace, 9.5f);
            _summaryBox.BackColor = Color.White;
            left.Controls.Add(_summaryBox, 0, 1);
            left.Controls.Add(Lbl(T("JSON 原文（响应体）", "JSON (response body)")), 0, 2);
            _jsonBox.Dock = DockStyle.Fill; _jsonBox.Multiline = true;
            _jsonBox.ReadOnly = true; _jsonBox.ScrollBars = ScrollBars.Both;
            _jsonBox.WordWrap = false;
            _jsonBox.Font = new Font(FontFamily.GenericMonospace, 9f);
            left.Controls.Add(_jsonBox, 0, 3);
            split.Panel1.Controls.Add(left);

            _preview.Dock = DockStyle.Fill;
            _preview.SizeMode = PictureBoxSizeMode.Zoom;
            _preview.BackColor = Color.FromArgb(43, 43, 43);
            _preview.AllowDrop = true;
            _preview.DragEnter += PreviewDragEnter;
            _preview.DragDrop += PreviewDragDrop;
            split.Panel2.Controls.Add(_preview);
            root.Controls.Add(split, 0, 2);
            Controls.Add(root);

            Load += async delegate { await RefreshApisAsync(); };
        }

        private Label Lbl(string text)
        {
            return new Label { Text = text, AutoSize = true, Anchor = AnchorStyles.Left,
                               Padding = new Padding(2, 4, 2, 2) };
        }

        private Button MakeButton(string text, EventHandler handler)
        {
            var b = new Button { Text = text, AutoSize = true };
            b.Click += handler;
            return b;
        }

        /// <summary>ToC1 修复：地址或 Key 任一变化即重建客户端（旧实现只比较
        /// 地址，改 Key 不生效）。</summary>
        private XccvClient Client()
        {
            string url = _urlBox.Text.Trim();
            string key = _keyBox.Text.Trim();
            if (_client == null || _clientUrl != url || _clientKey != key)
            {
                if (_client != null) _client.Dispose();
                _client = new XccvClient(url, key);
                _clientUrl = url;
                _clientKey = key;
            }
            return _client;
        }

        private float? OptFloat(TextBox box)
        {
            float v;
            if (float.TryParse(box.Text.Trim(), out v)) return v;
            return null;
        }

        private void Log(string text)
        {
            _summaryBox.AppendText(text + Environment.NewLine);
        }

        private void SetJson(string json)
        {
            _jsonBox.Text = Pretty(json);
        }

        private void ShowPng(byte[] png)
        {
            using (var ms = new MemoryStream(png))
            {
                var img = Image.FromStream(ms);
                if (_currentImage != null) _currentImage.Dispose();
                _currentImage = img;
                _preview.Image = _currentImage;
            }
        }

        // ---- API 列表：模型 [M] + 流程 [F] ----

        private bool SelectedIsFlow()
        {
            var s = SelectedId();
            return s.StartsWith("F|");
        }

        private string SelectedId()
        {
            return _apiCombo.SelectedItem == null
                ? "" : _apiCombo.SelectedItem.ToString();
        }

        private async Task RefreshApisAsync()
        {
            try
            {
                string modelsJson = await Client().StatusJsonAsync();
                var md = Client().Parse(modelsJson);
                var items = new List<string>();
                if (md.ContainsKey("models"))
                    foreach (object m in (md["models"] as IEnumerable ?? new object[0]))
                    {
                        var d = m as Dictionary<string, object>;
                        if (d != null && d.ContainsKey("model_id"))
                            items.Add("M|" + d["model_id"]);
                    }
                try
                {
                    string flowsJson = await Client().FlowsStatusJsonAsync();
                    var fd = Client().Parse(flowsJson);
                    if (fd.ContainsKey("flows"))
                        foreach (object f in (fd["flows"] as IEnumerable ?? new object[0]))
                        {
                            var d = f as Dictionary<string, object>;
                            if (d != null && d.ContainsKey("flow_id"))
                                items.Add("F|" + d["flow_id"]);
                        }
                }
                catch { /* 旧版服务无 flows 端点：忽略 */ }
                _apiCombo.Items.Clear();
                foreach (var it in items) _apiCombo.Items.Add(it);
                if (_apiCombo.Items.Count > 0) _apiCombo.SelectedIndex = 0;
                Log(T("已加载 API：", "Loaded APIs: ") + (_apiCombo.Items.Count == 0
                    ? T("（无）", "(none)") : _apiCombo.Items.Count + T(" 项", " item(s)")));
            }
            catch (Exception ex)
            {
                _apiCombo.Items.Clear();
                Log(T("无法获取 API 列表：", "Failed to list APIs: ") + ex.Message);
            }
        }

        // ---- 动作 ----

        private async void btnHealth_Click(object sender, EventArgs e)
        {
            try
            {
                SetJson(await Client().HealthJsonAsync());
                Log(T("健康检查 OK（详见 JSON 原文）",
                      "Health check OK (see JSON body)"));
            }
            catch (Exception ex) { Log(T("健康检查失败：", "Health check failed: ") + ex.Message); }
        }

        private async void btnLoadModel_Click(object sender, EventArgs e)
        {
            using (var dlg = new OpenFileDialog
            {
                Title = T("选择模型文件（.pt/.engine）",
                          "Select model file (.pt/.engine)"),
                Filter = T("模型权重 (*.pt;*.engine)|*.pt;*.engine|所有文件|*.*",
                           "Model weights (*.pt;*.engine)|*.pt;*.engine|All files|*.*")
            })
            {
                if (dlg.ShowDialog() != DialogResult.OK) return;
                try
                {
                    ModelHandle m = await Client().LoadModelAsync(dlg.FileName);
                    Log(T("模型已加载：", "Model loaded: ") + "id=" + m.ModelId
                        + "  task=" + m.Task + "  device=" + m.Device);
                    await RefreshApisAsync();
                }
                catch (Exception ex) { Log(T("加载失败：", "Load failed: ") + ex.Message); }
            }
        }

        private async void btnLoadFlow_Click(object sender, EventArgs e)
        {
            using (var dlg = new OpenFileDialog
            {
                Title = T("选择流程包（.xvap 流程包 / .json 流程图）",
                          "Select flow package (.xvap package / .json flow)"),
                Filter = T("流程 (*.xvap;*.json)|*.xvap;*.json",
                           "Flows (*.xvap;*.json)|*.xvap;*.json")
            })
            {
                if (dlg.ShowDialog() != DialogResult.OK) return;
                try
                {
                    ModelHandle f = await Client().LoadFlowAsync(dlg.FileName);
                    Log(T("流程已加载：", "Flow loaded: ") + "id=" + f.ModelId
                        + T("（内嵌模型首次推理时自动加载）",
                            " (embedded model auto-loads on first inference)"));
                    await RefreshApisAsync();
                }
                catch (Exception ex) { Log(T("流程加载失败：", "Flow load failed: ") + ex.Message); }
            }
        }

        private async void btnUnload_Click(object sender, EventArgs e)
        {
            if (_apiCombo.SelectedItem == null) return;
            try
            {
                string id = SelectedId();
                if (SelectedIsFlow()) await Client().UnloadFlowAsync(id.Substring(2));
                else await Client().UnloadModelAsync(id.Substring(2));
                Log(T("已卸载：", "Unloaded: ") + id);
                await RefreshApisAsync();
            }
            catch (Exception ex) { Log(T("卸载失败：", "Unload failed: ") + ex.Message); }
        }

        private async void btnRefresh_Click(object sender, EventArgs e)
        {
            await RefreshApisAsync();
        }

        /// <summary>单帧推理（模型）：JSON + 摘要 + 右侧叠加预览。</summary>
        private async void btnInfer_Click(object sender, EventArgs e)
        {
            if (_apiCombo.SelectedItem == null || SelectedIsFlow())
            {
                MessageBox.Show(T(
                    "请先在下拉选择一个模型（[M] 开头）；流程请用「流程推理」",
                    "Pick a model ([M]) in the dropdown first; use Run Flow for flows"));
                return;
            }
            using (var dlg = new OpenFileDialog
            {
                Title = T("选择测试图片", "Select test image"),
                Filter = T("图片 (*.jpg;*.jpeg;*.png;*.bmp)|*.jpg;*.jpeg;*.png;*.bmp",
                           "Images (*.jpg;*.jpeg;*.png;*.bmp)|*.jpg;*.jpeg;*.png;*.bmp")
            })
            {
                if (dlg.ShowDialog() != DialogResult.OK) return;
                string mid = SelectedId().Substring(2);
                byte[] imgBytes = File.ReadAllBytes(dlg.FileName);
                string name = Path.GetFileName(dlg.FileName);
                try
                {
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    string json = await Client().InferImageJsonAsync(mid, imgBytes,
                        name, OptFloat(_confBox), OptFloat(_iouBox));
                    long ms = sw.ElapsedMilliseconds;
                    SetJson(json);
                    Summarize(name, ms, json);
                    try
                    {
                        byte[] png = await Client().RenderOverlayPngAsync(mid,
                            imgBytes, name, OptFloat(_confBox), OptFloat(_iouBox));
                        ShowPng(png);
                    }
                    catch (Exception ex) { Log(T("叠加预览失败：", "Overlay preview failed: ") + ex.Message); }
                }
                catch (Exception ex) { Log(T("推理失败：", "Inference failed: ") + ex.Message); }
            }
        }

        /// <summary>流程推理（已加载流程）：JSON + 摘要 + 叠加预览。</summary>
        private async void btnFlowRun_Click(object sender, EventArgs e)
        {
            if (_apiCombo.SelectedItem == null || !SelectedIsFlow())
            {
                MessageBox.Show(T(
                    "请先在下拉选择一个流程（[F] 开头）；单模型请用「单帧推理」",
                    "Pick a flow ([F]) in the dropdown first; use Infer (Model) for models"));
                return;
            }
            using (var dlg = new OpenFileDialog
            {
                Title = T("选择测试图片", "Select test image"),
                Filter = T("图片 (*.jpg;*.jpeg;*.png;*.bmp)|*.jpg;*.jpeg;*.png;*.bmp",
                           "Images (*.jpg;*.jpeg;*.png;*.bmp)|*.jpg;*.jpeg;*.png;*.bmp")
            })
            {
                if (dlg.ShowDialog() != DialogResult.OK) return;
                string fid = SelectedId().Substring(2);
                byte[] imgBytes = File.ReadAllBytes(dlg.FileName);
                string name = Path.GetFileName(dlg.FileName);
                try
                {
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    string json = await Client().RunFlowByIdJsonAsync(fid, imgBytes, name);
                    long ms = sw.ElapsedMilliseconds;
                    SetJson(json);
                    Summarize(name, ms, json);
                    byte[] png = await Client().RunFlowByIdOverlayPngAsync(fid,
                        imgBytes, name);
                    ShowPng(png);
                }
                catch (Exception ex) { Log(T("流程推理失败：", "Flow inference failed: ") + ex.Message); }
            }
        }

        private void btnSaveImage_Click(object sender, EventArgs e)
        {
            if (_preview.Image == null)
            {
                MessageBox.Show(T("当前没有预览图（先推理一次）",
                                  "No preview image yet (run an inference first)"));
                return;
            }
            using (var dlg = new SaveFileDialog
            {
                Title = T("保存预览图", "Save preview image"),
                Filter = T("PNG 图片|*.png", "PNG image|*.png"),
                FileName = "overlay_" + DateTime.Now.ToString("yyyyMMdd_HHmmss") + ".png"
            })
            {
                if (dlg.ShowDialog() != DialogResult.OK) return;
                _preview.Image.Save(dlg.FileName,
                    System.Drawing.Imaging.ImageFormat.Png);
                Log(T("预览图已保存：", "Preview image saved: ") + dlg.FileName);
            }
        }

        /// <summary>把统一 JSON 提炼成人话摘要（模型/图片/耗时 + 逐条检出）。</summary>
        private void Summarize(string image, long ms, string json)
        {
            try
            {
                var d = Client().Parse(json);
                _summaryBox.AppendText(T("图片：", "Image: ") + image
                    + T("（", " (") + ms + " ms" + T("）", ")")
                    + Environment.NewLine);
                object v;
                if (d.TryGetValue("verdict", out v) && v != null)
                {
                    var vd = v as Dictionary<string, object>;
                    if (vd != null)
                        _summaryBox.AppendText(T("判定：", "Verdict: ")
                            + (true.Equals(vd.ContainsKey("passed") ? vd["passed"] : null)
                            ? T("通过", "PASS") : T("不通过", "FAIL"))
                            + T("（", " (") + Str(vd, "detail") + T("）", ")")
                            + Environment.NewLine);
                }
                int n = 0;
                if (d.TryGetValue("detections", out v))
                    foreach (object det in (v as IEnumerable ?? new object[0]))
                    {
                        var dd = det as Dictionary<string, object>;
                        if (dd == null) continue;
                        n++;
                        _summaryBox.AppendText(string.Format("[{0}] {1}  score={2}  bbox={3}",
                            n, Str(dd, "label"),
                            dd.ContainsKey("score") ? dd["score"] : "",
                            dd.ContainsKey("bbox") ? FormatList(dd["bbox"]) : "")
                            + Environment.NewLine);
                    }
                if (d.TryGetValue("classifications", out v))
                    foreach (object c in (v as IEnumerable ?? new object[0]))
                    {
                        var cd = c as Dictionary<string, object>;
                        if (cd == null) continue;
                        n++;
                        _summaryBox.AppendText(string.Format("[Top] {0}  score={1}",
                            Str(cd, "label"), cd.ContainsKey("score") ? cd["score"] : "")
                            + Environment.NewLine);
                    }
                if (n == 0) _summaryBox.AppendText(T("检出：0 个", "Detections: 0")
                    + Environment.NewLine);
                _summaryBox.AppendText(Environment.NewLine);
            }
            catch { _summaryBox.AppendText(json + Environment.NewLine); }
        }

        private static string Str(Dictionary<string, object> d, string key)
        {
            object v;
            if (d.TryGetValue(key, out v) && v != null) return v.ToString();
            return "";
        }

        private static string FormatList(object arr)
        {
            var parts = new List<string>();
            foreach (object x in (arr as IEnumerable ?? new object[0]))
                parts.Add(x.ToString());
            return "(" + string.Join(", ", parts) + ")";
        }

        // ---- 拖拽图片到预览区直接推理 ----
        private void PreviewDragEnter(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
                e.Effect = DragDropEffects.Copy;
        }

        private void PreviewDragDrop(object sender, DragEventArgs e)
        {
            var files = e.Data.GetData(DataFormats.FileDrop) as string[];
            if (files == null || files.Length == 0) return;
            if (_apiCombo.SelectedItem == null)
            {
                MessageBox.Show(T("请先加载并选择模型/流程",
                                  "Load and pick a model/flow first"));
                return;
            }
            string id = SelectedId().Substring(2);
            byte[] imgBytes = File.ReadAllBytes(files[0]);
            string name = Path.GetFileName(files[0]);
            Task.Run(async delegate
            {
                if (SelectedIsFlow()) await Client().RunFlowByIdJsonAsync(id, imgBytes, name);
                else await Client().InferImageJsonAsync(id, imgBytes, name,
                    OptFloat(_confBox), OptFloat(_iouBox));
            }).ContinueWith(t =>
            {
                BeginInvoke(new Action(() =>
                    Log(T("拖拽推理完成：", "Drag-drop inference done: ") + name
                        + T("（详情见 JSON）", " (see JSON)"))));
            });
        }

        // ---- JSON 美化 ----
        private static string Pretty(string json)
        {
            try
            {
                var ser = new JavaScriptSerializer();
                ser.MaxJsonLength = int.MaxValue;
                object obj = ser.Deserialize<object>(json);
                var sb = new StringBuilder();
                Write(obj, 0, sb);
                return sb.ToString();
            }
            catch { return json; }
        }

        private static void Write(object obj, int indent, StringBuilder sb)
        {
            // WinForms TextBox 只认 \r\n：纯 \n 不换行（会显示成一长行）
            string nl = "\r\n";
            string pad = new string(' ', indent * 2);
            string pad2 = new string(' ', (indent + 1) * 2);
            if (obj is Dictionary<string, object>)
            {
                var d = (Dictionary<string, object>)obj;
                if (d.Count == 0) { sb.Append("{}"); return; }
                sb.Append("{").Append(nl);
                int i = 0;
                foreach (var kv in d)
                {
                    sb.Append(pad2).Append('"').Append(kv.Key).Append("\": ");
                    Write(kv.Value, indent + 1, sb);
                    if (++i < d.Count) sb.Append(',');
                    sb.Append(nl);
                }
                sb.Append(pad).Append('}');
            }
            else if (obj is IEnumerable && !(obj is string))
            {
                var list = ((IEnumerable)obj).Cast<object>().ToList();
                if (list.Count == 0) { sb.Append("[]"); return; }
                sb.Append("[").Append(nl);
                for (int i = 0; i < list.Count; i++)
                {
                    sb.Append(pad2);
                    Write(list[i], indent + 1, sb);
                    if (i < list.Count - 1) sb.Append(',');
                    sb.Append(nl);
                }
                sb.Append(pad).Append(']');
            }
            else if (obj is string) sb.Append('"').Append(obj).Append('"');
            else sb.Append(obj == null ? "null" : obj.ToString());
        }
    }
}

