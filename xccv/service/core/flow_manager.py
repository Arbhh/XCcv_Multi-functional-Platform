"""已加载流程池：.xvap/.json 流程包常驻为可寻址 API（flow_id）。

与 ModelManager 对称：加载（解包/改写模型引用到持久目录）→ 运行（逐流程
串行，FlowSession 常驻使流程内 AI 模型仅加载一次）→ 卸载清理。LRU 驻留
上限同模型池。安全：加载路径走白名单；流程内嵌模型引用运行前同样校验。
"""

import itertools
import json
import logging
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

_FLOW_DATA_DIR = None   # 惰性初始化（app_root 依赖，测试可重定向）


def _data_dir() -> Path:
    global _FLOW_DATA_DIR
    if _FLOW_DATA_DIR is None:
        from ...app_paths import app_root

        _FLOW_DATA_DIR = app_root() / "platform_data" / "service" / "flow_pkgs"
    return _FLOW_DATA_DIR


class FlowNotFound(Exception):
    """flow_id 不存在（→ 404）。"""


@dataclass
class FlowEntry:
    flow_id: str
    alias: str
    path: str                 # 持久化后的 .xvap/.json（服务生命周期内有效）
    flow: dict                # 模型引用已改写为持久绝对路径
    session: object
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    last_used: float = 0.0
    loaded_at: float = 0.0


class FlowManager:
    def __init__(self, max_loaded: int = 3):
        self._max = max(1, int(max_loaded))
        self._lock = threading.RLock()
        self._entries: dict[str, FlowEntry] = {}
        self._seq = itertools.count(1)
        self._loading: dict[str, threading.Lock] = {}  # per-path 单飞锁（与模型池同语义）

    def list(self) -> list[dict]:
        with self._lock:
            items = sorted(self._entries.values(), key=lambda e: e.last_used)
            return [{"flow_id": e.flow_id, "alias": e.alias, "path": e.path,
                     "loaded_at": e.loaded_at, "last_used": e.last_used}
                    for e in items]

    def get(self, flow_id: str) -> FlowEntry:
        with self._lock:
            entry = self._entries.get(flow_id)
        if entry is None:
            raise FlowNotFound(f"流程未加载或已卸载: {flow_id}")
        return entry

    def load(self, path: Path, alias: str, cfg) -> FlowEntry:
        """加载 .xvap/.json 为常驻流程（同路径幂等；LRU 淘汰最久未用）。

        cfg 用于 .json 流程内模型引用的白名单校验（.xvap 内嵌模型解包到
        服务数据目录，天然在受控范围内，无需再校验）。
        """
        from ...engine import flow_package as fp
        from ...modeltools.flow_session import FlowSession

        path = Path(path)
        key = str(path.resolve())
        with self._lock:
            for e in self._entries.values():
                if e.path == key:
                    e.last_used = time.time()
                    return e
            path_gate = self._loading.setdefault(key, threading.Lock())
        with path_gate:
            with self._lock:
                # 等锁期间他人可能已完成加载——单飞复用出口
                for e in self._entries.values():
                    if e.path == key:
                        e.last_used = time.time()
                        return e
            safe = (alias or path.stem)
            flow_id = f"flow-{next(self._seq)}-{safe}"[:64]
            # 持久化到服务数据目录（解包出的模型在服务生命周期内保持有效）
            pkg_dir = _data_dir() / flow_id
            pkg_dir.mkdir(parents=True, exist_ok=True)
            if path.suffix.lower() == ".xvap":
                local_pkg = pkg_dir / path.name
                shutil.copyfile(path, local_pkg)
                flow, _model_dir = fp.unpack_to_dir(local_pkg)
            elif path.suffix.lower() == ".json":
                flow = json.loads(path.read_text(encoding="utf-8"))
                _check_flow_model_paths(flow, cfg)
            else:
                raise ValueError("流程文件仅支持 .json / .xvap")
            with self._lock:
                while len(self._entries) >= self._max:
                    oldest = min(self._entries.values(), key=lambda e: e.last_used)
                    log.info("LRU 卸载流程 %s：超过驻留上限 %d",
                             oldest.flow_id, self._max)
                    self._drop(oldest.flow_id)
                entry = FlowEntry(flow_id=flow_id, alias=alias or path.stem,
                                  path=key, flow=flow,
                                  session=FlowSession(), loaded_at=time.time())
                self._entries[flow_id] = entry
            log.info("流程已加载: %s（%s，%d 节点）", flow_id, path.name,
                     len(flow.get("nodes") or []))
            return entry

    def run(self, flow_id: str, image_path: Path, source_name: str,
            flow_name: str = ""):
        """运行已加载流程（逐流程串行；返回 (payload, session, base_image)）。"""
        from ..api.routes_flow import run_flow_payload

        entry = self.get(flow_id)
        with entry.lock:
            payload, session, base = run_flow_payload(
                entry.flow, flow_name or entry.alias, image_path, source_name,
                session=entry.session)
            entry.last_used = time.time()   # 锁内更新（与 LRU 淘汰互斥）
        return payload, session, base

    def unload(self, flow_id: str) -> bool:
        with self._lock:
            if flow_id not in self._entries:
                return False
            self._drop(flow_id)
        return True

    def _drop(self, flow_id: str) -> None:
        entry = self._entries.pop(flow_id, None)
        if entry is None:
            return
        # 与运行互斥：entry.lock 被 in-flight 运行持有时延迟到其结束再清理
        with entry.lock:
            pkg_dir = _data_dir() / flow_id
            shutil.rmtree(pkg_dir, ignore_errors=True)
        log.info("流程已卸载: %s", flow_id)


def _check_flow_model_paths(flow: dict, cfg) -> None:
    """.json 流程直接引用的本机模型路径白名单校验（防绕过模型白名单）。"""
    from ..security import ensure_path_allowed

    for n in flow.get("nodes") or []:
        props = n.get("properties") or n.get("params") or {}
        mp = str(props.get("model_path") or "").strip()
        if mp and Path(mp).suffix.lower() in (".pt", ".engine"):
            ensure_path_allowed(mp, cfg)

