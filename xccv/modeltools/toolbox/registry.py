# -*- coding: utf-8 -*-
"""百宝箱子功能注册表：注册即上架（新工具不改宿主代码）。

每个子功能一条 ToolSpec（id / 名称 / 描述 / 主题色 / 图标 / 搜索关键词 /
入口工厂）；百宝箱窗口的卡片墙与搜索框均由注册表驱动。
"""

import logging
from dataclasses import dataclass, field
from typing import Callable

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolSpec:
    """一个百宝箱子功能的登记项。factory 为入口窗口类（惰性实例化）。"""

    key: str
    title: str
    desc: str
    color: str
    icon: str
    keywords: str = ""       # 搜索关键词（空格分隔，含中文/英文别名）
    tooltip: str = ""        # 卡片悬停说明（详细口径，替代描述里的括号注释）
    factory: Callable | None = None


_REGISTERED: list = []
_BY_KEY: dict = {}


def register(spec: ToolSpec) -> None:
    """登记子功能；重复 key 覆盖（幂等，支持模块重导入）。"""
    if spec.key in _BY_KEY:
        _REGISTERED[_REGISTERED.index(_BY_KEY[spec.key])] = spec
    else:
        _REGISTERED.append(spec)
    _BY_KEY[spec.key] = spec


def all_tools() -> list:
    """按登记顺序返回全部子功能。"""
    return list(_REGISTERED)


def search(query: str) -> list:
    """按名称/描述/关键词过滤子功能（不区分大小写，空查询返回全部）。"""
    q = (query or "").strip().lower()
    if not q:
        return all_tools()
    out = []
    for spec in _REGISTERED:
        hay = " ".join([spec.title, spec.desc, spec.keywords]).lower()
        if all(tok in hay for tok in q.split()):
            out.append(spec)
    return out


def get(key: str) -> ToolSpec | None:
    return _BY_KEY.get(key)
