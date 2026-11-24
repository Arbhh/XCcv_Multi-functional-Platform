# -*- coding: utf-8 -*-
"""预训练权重缺项补齐（训练前可见下载，替代 ultralytics 的静默下载）。

- 落点：WEIGHTS_DIR/ultralytics/<名称>.pt（开发态=引擎包 weights/，
  安装态/冻结态=运行根 weights/，与随包官方预训练权重同位）；
- 进度经 notify(text, frac) 上报（训练链路接 TrainReporter.on_message；
  首次通知 frac=None 表示不确定进度，之后按字节比例上报）；
- 多源回退：官方 GitHub 源失败后依次尝试镜像，全部失败抛 RuntimeError，
  由训练线程把失败原因带回界面并结束任务——不再回退 ultralytics 自带
  下载（其 curl/urllib 无超时且可能弹出终端窗口，网络异常时会静默卡死）；
- 下载使用带连接/读取超时的流式实现，并校验文件大小与 .pt 文件头；
- fetcher 可注入（测试以本地写入替代网络下载）。
"""

import hashlib
import logging
import os
import time
import urllib.request
from pathlib import Path

from ..offline import WEIGHTS_DIR

log = logging.getLogger(__name__)

_OFFICIAL_URL = ("https://github.com/ultralytics/assets/releases/"
                 "download/v8.3.0/{name}")
# 镜像按顺序尝试：GitHub 官方 → 两个 GitHub 加速前缀 → HuggingFace 镜像
_MIRROR_URLS = (
    "https://ghfast.top/" + _OFFICIAL_URL,
    "https://ghproxy.net/" + _OFFICIAL_URL,
    "https://hf-mirror.com/Ultralytics/YOLO11/resolve/main/{name}",
)
_MIN_BYTES = 100 * 1024          # 小于此值视为下载不完整（YOLO 权重均数 MB）
# 官方 v8.3.0 资产校验（镜像回退时防止第三方源内容被替换；未知文件名跳过）
_KNOWN_SHA256 = {
    "yolo11n.pt": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    "yolo11s.pt": "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5",
    "yolo11n-seg.pt": "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152",
    "yolo11s-seg.pt": "1caa81c0195412efa411b632bcfb8c184939dddb6ae41f6a80c41b211ff257c3",
}
_CONNECT_TIMEOUT = 8             # 连接超时（秒）
_READ_TIMEOUT = 30               # 单次读取超时（秒）
_DEADLINE_S = 120.0              # 单源总耗时上限（秒），防低速网络无限拖住
_TOTAL_DEADLINE_S = 360.0        # 全部下载源合计上限（秒）
_CHUNK = 256 * 1024


def _url_candidates(name: str) -> list:
    """下载源列表；XCCV_WEIGHT_URLS 可覆盖（逗号分隔，支持 {name} 占位）。"""
    env = (os.environ.get("XCCV_WEIGHT_URLS") or "").strip()
    if env:
        templates = [u.strip() for u in env.replace(";", ",").split(",")
                     if u.strip()]
    else:
        templates = [_OFFICIAL_URL, *_MIRROR_URLS]
    urls = []
    for tpl in templates:
        urls.append(tpl.format(name=name) if "{name}" in tpl
                    else tpl.rstrip("/") + "/" + name)
    return urls


def _check_file(path: Path, name: str = "") -> bool:
    """已存在文件是否可用：大小达标、具备 .pt 容器头；已知权重再校验 SHA-256。"""
    try:
        if not path.is_file() or path.stat().st_size < _MIN_BYTES:
            return False
        with open(path, "rb") as fh:
            head = fh.read(4)
        expected = _KNOWN_SHA256.get(name)
        if expected:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected:
                log.warning("预训练权重校验失败（%s）：%s ≠ %s",
                            name, digest[:12], expected[:12])
                return False
    except OSError:
        return False
    return head[:2] == b"PK" or head[:1] == b"\x80"


def _progress_cb(progress, done: int, total: int):
    if progress is not None and total > 0:
        progress(min(1.0, done / total))


def _requests_fetcher(url: str, dst: Path, progress=None) -> None:
    """requests 流式下载（连接/读取超时 + 总耗时上限）。"""
    import requests

    started = time.monotonic()
    with requests.get(url, stream=True, timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
                      allow_redirects=True) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        with open(dst, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=_CHUNK):
                if not chunk:
                    continue
                fh.write(chunk)
                done += len(chunk)
                _progress_cb(progress, done, total)
                if time.monotonic() - started > _DEADLINE_S:
                    raise TimeoutError("下载超时")


def _urllib_fetcher(url: str, dst: Path, progress=None) -> None:
    """urllib 回退（requests 缺失时）：流式读写 + socket 超时。"""
    req = urllib.request.Request(url, headers={"User-Agent": "XCcv"})
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=_READ_TIMEOUT) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        with open(dst, "wb") as fh:
            while True:
                chunk = resp.read(_CHUNK)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                _progress_cb(progress, done, total)
                if time.monotonic() - started > _DEADLINE_S:
                    raise TimeoutError("下载超时")


def _default_fetcher(url: str, dst: Path, progress=None) -> None:
    """默认下载器：优先 requests（超时/代理支持好），缺失时用 urllib。"""
    try:
        _requests_fetcher(url, dst, progress)
    except ImportError:
        _urllib_fetcher(url, dst, progress)


def ensure_pretrained_weight(init_weight: str, notify=None,
                             fetcher=None) -> str:
    """确保预训练权重就绪，返回应传给 YOLO() 的本地权重路径。

    仅处理"裸文件名"（如 yolo11n.pt）：显式路径与 .yaml 结构配置原样返回。
    下载失败不再交给 ultralytics 自行处理，而是抛出 RuntimeError，让界面
    显示失败原因并结束本次训练（避免无提示卡死）。
    """
    name = Path(init_weight).name
    if str(init_weight) != name or name.endswith(".yaml"):
        return str(init_weight)
    dest = WEIGHTS_DIR / "ultralytics" / name
    if _check_file(dest, name):
        return str(dest)

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"无法创建预训练权重目录：{dest.parent}（{exc}）。"
            "请检查安装目录写权限后重试。") from exc

    if fetcher is None:
        fetcher = _default_fetcher
    urls = _url_candidates(name)
    errors = []
    total_deadline = time.monotonic() + _TOTAL_DEADLINE_S
    for idx, url in enumerate(urls, start=1):
        if time.monotonic() >= total_deadline:
            errors.append(f"源{idx} 未尝试：全部下载源总耗时超过 "
                          f"{int(_TOTAL_DEADLINE_S)} 秒")
            break
        tmp = dest.with_name(dest.name + f".part{idx}")
        if notify is not None:
            notify(f"正在下载预训练权重 {name}（下载源 {idx}/{len(urls)}）…", None)

        def _progress(frac, _idx=idx):
            if notify is not None:
                notify(f"正在下载预训练权重 {name}（下载源 {_idx}/{len(urls)}）…",
                       frac)

        try:
            fetcher(url, tmp, _progress)
            if not _check_file(tmp, name):
                raise ValueError("下载内容不完整、文件头无效或 SHA-256 校验失败")
            tmp.replace(dest)
            if notify is not None:
                notify(f"预训练权重就绪：{name}", 1.0)
            return str(dest)
        except Exception as exc:  # noqa: BLE001（逐源记录，最后统一上报）
            errors.append(f"源{idx} {type(exc).__name__}: {str(exc)[:160]}")
            log.warning("预训练权重下载失败（%s/%s，%s）：%s",
                        idx, len(urls), url, exc)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            if notify is not None and idx < len(urls):
                notify(f"下载源 {idx} 失败（{type(exc).__name__}），"
                       f"正在切换下一个下载源…", None)

    detail = ("；".join(errors[-2:])) if errors else "未知错误"
    message = (f"预训练权重 {name} 下载失败（已尝试 {len(urls)} 个下载源）。"
               f"请检查网络或代理后重试，也可手动下载该文件放入：{dest}。"
               f"详情：{detail}")
    if notify is not None:
        notify(message, None)
    raise RuntimeError(message)
