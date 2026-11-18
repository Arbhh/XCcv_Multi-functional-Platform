// Defect Augment Studio — i18n runtime (V0.7.3): DOM text-node translation
// driven by js/i18n_en.js dictionary. Only activates when the Qt shell sets
// window.__DG_LANG='en' (zh mode never applies → zero cost). Covers static
// HTML, JS-rendered DOM (MutationObserver), placeholder/title attributes and
// exposes __dgT() for canvas fillText labels. Backend notes/errors reaching
// the DOM (toasts/reports) translate through the same dictionary.
(function () {
  "use strict";
  if (window.__dgI18nLoaded) return;
  window.__dgI18nLoaded = true;

  var exact = {};
  var patterns = [];   // [RegExp, template] for keys with ${...}/%s/%d
  var substrKeys = []; // substring fallback keys, longest first

  function rxQuote(ch) {
    return /[.*+?^${}()|[\]\\]/.test(ch) ? "\\" + ch : ch;
  }

  function build() {
    if (window.__dgBuilt) return;
    window.__dgBuilt = true;
    var D = window.__DG_I18N || {};
    for (var zh in D) {
      if (!Object.prototype.hasOwnProperty.call(D, zh)) continue;
      var en = D[zh];
      if (zh === en || !zh) continue;
      exact[zh] = en;
      if (/(\$\{[^}]*\}|%s|%d)/.test(zh)) {
        var src = zh
          .replace(/\$\{[^}]*\}/g, "\u0001")
          .replace(/%s/g, "\u0001")
          .replace(/%d/g, "\u0002");
        var rx = "";
        for (var i = 0; i < src.length; i++) {
          var ch = src[i];
          if (ch === "\u0001") rx += "(.+?)";
          else if (ch === "\u0002") rx += "(\\d+)";
          else rx += rxQuote(ch);
        }
        patterns.push([new RegExp("^" + rx + "$"), en]);
      }
    }
    substrKeys = Object.keys(exact).filter(function (k) {
      return k.length >= 2 && !/(\$\{[^}]*\}|%s|%d)/.test(k);
    }).sort(function (a, b) { return b.length - a.length; });
    patterns.sort(function (a, b) { return b[0].source.length - a[0].source.length; });
  }

  function trText(s) {
    if (!s || !/[\u4e00-\u9fff]/.test(s)) return s;
    var t = s.trim();
    // 1) exact full match
    if (Object.prototype.hasOwnProperty.call(exact, t)) {
      return s.replace(t, exact[t]);
    }
    // 2) anchored patterns (${...}/%s/%d templates)
    for (var i = 0; i < patterns.length; i++) {
      var m = t.match(patterns[i][0]);
      if (m) {
        var out = patterns[i][1];
        for (var g = 1; g < m.length; g++) {
          out = out.split("$" + g).join(m[g] != null ? m[g] : "");
        }
        return s.replace(t, out);
      }
    }
    // 3) substring pass（混合文案：长键优先，避免短键截断）
    var res = s;
    for (var j = 0; j < substrKeys.length; j++) {
      var k = substrKeys[j];
      if (res.indexOf(k) >= 0) res = res.split(k).join(exact[k]);
    }
    return res;
  }

  function walkNode(node) {
    if (node.nodeType === 3) {           // text node
      var v = node.nodeValue;
      if (v && /[\u4e00-\u9fff]/.test(v)) {
        var nv = trText(v);
        if (nv !== v) node.nodeValue = nv;
      }
      return;
    }
    if (node.nodeType !== 1) return;
    if (node.tagName === "SCRIPT" || node.tagName === "STYLE") return;
    if (node.tagName === "TEXTAREA" || node.tagName === "INPUT") {
      var ph = node.getAttribute && node.getAttribute("placeholder");
      if (ph) {
        var np = trText(ph);
        if (np !== ph) node.setAttribute("placeholder", np);
      }
      if (node.tagName === "INPUT") {
        var vl = node.getAttribute("value");
        if (vl && /[\u4e00-\u9fff]/.test(vl)) {
          var nv2 = trText(vl);
          if (nv2 !== vl) node.setAttribute("value", nv2);
        }
      }
    }
    var ti = node.getAttribute && node.getAttribute("title");
    if (ti && /[\u4e00-\u9fff]/.test(ti)) {
      var nt = trText(ti);
      if (nt !== ti) node.setAttribute("title", nt);
    }
    if (node.childNodes) {
      for (var c = 0; c < node.childNodes.length; c++) walkNode(node.childNodes[c]);
    }
  }

  function applyAll() {
    build();
    if (window.__DG_LANG !== "en") return;
    walkNode(document.body || document.documentElement);
  }

  // canvas fillText 用：window.__dgT("原始")
  window.__dgT = function (s) {
    build();
    if (window.__DG_LANG !== "en") return s;
    var r = trText(s);
    return r || s;
  };

  window.__dgApplyI18N = applyAll;

  function startObserver() {
    if (!window.MutationObserver || window.__dgObs) return;
    var pending = false;
    var obs = new MutationObserver(function () {
      if (pending) return;
      pending = true;
      setTimeout(function () {
        pending = false;
        applyAll();
      }, 120);
    });
    obs.observe(document.body || document.documentElement,
      { childList: true, subtree: true, characterData: true });
    window.__dgObs = obs;
  }

  function boot() {
    // 语言来自 Qt 壳 URL 参数（?lang=en）——脚本加载期即定，早于 ES 模块渲染
    if (window.__DG_LANG == null) {
      window.__DG_LANG = /(?:^|[?&])lang=en(?:&|$)/.test(location.search) ? "en" : "zh";
    }
    if (window.__DG_LANG !== "en") return;  // zh：不装观察器，零开销
    applyAll();
    startObserver();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
