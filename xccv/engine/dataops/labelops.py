# -*- coding: utf-8 -*-
"""labelme 数据集标签批量操作纯逻辑（替换 / 删除 / 迁移）。

约定：
- 数据集目录（含各级子目录）按"图片 + 同名 .json"成对组织；无 json 的
  图片视为未标注样本；
- 所有操作均分两步：plan_* 只读扫描生成预览条目，apply_* 按预览执行；
- JSON 改写保留其余全部字段原样（shapes/flags/imagePath/imageData 等），
  以 utf-8 ensure_ascii=False 落盘；
- 匹配口径：plain 模式为标签全等；regex 模式为 re.fullmatch（整词正则）。
"""

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class SamplePair:
    """一个样本对：image 为主键；json 为 None 表示未标注。"""

    image: Path
    json: Path | None = None

    @property
    def labeled(self) -> bool:
        return self.json is not None


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def scan_pairs(root: Path) -> list:
    """递归扫描目录（含各级子目录）生成样本对列表。

    以图片为锚匹配同目录同名 .json（扩展名与主干比较不区分大小写）；
    另纳入"有 json 无图片"的对（图片缺失，image 字段指向不存在的文件，
    迁移/删除时仍可成对处理）。图片对按源图完整路径排序，孤儿 json 随后
    按目录/主干排序，结果可复现。
    """
    root = Path(root)
    images: list = []
    json_index: dict = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix == ".json":
            json_index.setdefault((p.parent, p.stem.lower()), p)
        elif suffix in IMAGE_EXTS:
            images.append(p)
    pairs: list = []
    seen_json = set()
    for p in images:
        jp = json_index.get((p.parent, p.stem.lower()))
        if jp is not None:
            seen_json.add(jp)
            pairs.append(SamplePair(image=p, json=jp))
        else:
            pairs.append(SamplePair(image=p, json=None))
    for key in sorted(json_index):
        jp = json_index[key]
        if jp not in seen_json:
            pairs.append(SamplePair(image=jp.with_suffix(""), json=jp))
    return pairs


def iter_labels(pair: SamplePair) -> set:
    """一个样本对的标签集合（形状 label + 图片级 flags 键）。"""
    if not pair.labeled:
        return set()
    try:
        data = _load_json(pair.json)
    except Exception:
        return set()
    labels = set()
    for s in data.get("shapes", []):
        lab = s.get("label")
        if lab:
            labels.add(lab)
    for k in (data.get("flags") or {}):
        labels.add(k)
    return labels


def _match(label: str, old: str, regex: bool) -> bool:
    if regex:
        try:
            return re.fullmatch(old, label) is not None
        except re.error:
            return False
    return label == old


# ---------------------------------------------------------------- 替换


@dataclass
class ReplaceEntry:
    """一条替换预览：目标 json + 命中的形状序号与新旧标签。"""

    json_path: Path
    hits: list = field(default_factory=list)  # [(shape_idx, old_label, new_label)]
    flag_key_hits: list = field(default_factory=list)  # [(scope, old_key, new_key)]


def plan_replace(pairs: list, old: str, new: str, regex: bool = False) -> list:
    """生成标签替换预览（不改盘）。flags 键同步替换（图片级与形状级）。"""
    entries = []
    for pair in pairs:
        if not pair.labeled:
            continue
        try:
            data = _load_json(pair.json)
        except Exception:
            continue
        entry = ReplaceEntry(json_path=pair.json)
        for idx, s in enumerate(data.get("shapes", [])):
            lab = s.get("label") or ""
            if _match(lab, old, regex):
                entry.hits.append((idx, lab, new))
        for scope in ("image", "shape"):
            if scope == "image":
                flag_keys = list((data.get("flags") or {}).keys())
            else:
                flag_keys = []
                for s in data.get("shapes", []):
                    flag_keys.extend((s.get("flags") or {}).keys())
            for k in flag_keys:
                if _match(k, old, regex):
                    entry.flag_key_hits.append((scope, k, new))
        if entry.hits or entry.flag_key_hits:
            entries.append(entry)
    return entries


def apply_replace(entries: list) -> dict:
    """执行替换，返回统计（改动文件数 / 形状数 / flags 键数）。"""
    n_files = n_shapes = n_flags = 0
    for entry in entries:
        data = _load_json(entry.json_path)
        for idx, _old, new in entry.hits:
            data["shapes"][idx]["label"] = new
            n_shapes += 1
        for scope, old_key, new_key in entry.flag_key_hits:
            n_flags += _rename_flag_key(entry, data, scope, old_key, new_key)
        _dump_json(entry.json_path, data)
        n_files += 1
    return {"files": n_files, "shapes": n_shapes, "flag_keys": n_flags}


def _rename_flag_key(entry: ReplaceEntry, data, scope: str, old_key: str, new_key: str) -> int:
    """把 flags 字典中 old_key 改名为 new_key（保序），返回改动数。"""
    containers = []
    if scope == "image":
        containers.append(data.get("flags") or {})
    else:
        containers.extend((s.get("flags") or {}) for s in data.get("shapes", []))
    changed = 0
    for flags in containers:
        if old_key in flags:
            value = flags.pop(old_key)
            flags[new_key] = value
            changed += 1
    return changed


# ---------------------------------------------------------------- 删除


@dataclass
class DeleteEntry:
    """一条删除预览。

    mode="shapes"：删除指定 json 内的匹配形状（matches 为形状序号列表；
    labels 为命中的目标标签，执行时同步清除同名图片级 flags 键）；
    mode="images"：整图过滤，删除图片与 json（has_image 表示图片文件是否存在）。
    """

    mode: str
    json_path: Path | None
    image_path: Path | None
    matches: list = field(default_factory=list)
    labels_seen: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    has_image: bool = True


def plan_delete(pairs: list, labels: list, mode: str = "shapes") -> list:
    """生成删除预览。labels 为精确标签集合；images 模式按"含任一标签"整图命中。"""
    want = set(labels)
    entries = []
    for pair in pairs:
        if not pair.labeled:
            continue
        try:
            data = _load_json(pair.json)
        except Exception:
            continue
        shape_labels = [s.get("label") or "" for s in data.get("shapes", [])]
        if mode == "shapes":
            matches = [i for i, lab in enumerate(shape_labels) if lab in want]
            if matches:
                entries.append(DeleteEntry(
                    mode=mode, json_path=pair.json, image_path=pair.image,
                    matches=matches, labels_seen=sorted(set(shape_labels)),
                    labels=sorted(want & set(shape_labels))))
        else:
            if any(lab in want for lab in shape_labels):
                entries.append(DeleteEntry(
                    mode=mode, json_path=pair.json, image_path=pair.image,
                    labels_seen=sorted(set(shape_labels)), has_image=pair.image.exists()))
    return entries


def apply_delete(entries: list, backup_dir: Path | None = None) -> dict:
    """执行删除。backup_dir 给定时先复制待删文件再删（可选，默认直接删）。

    shapes 模式删空后（无形状且图片级 flags 全 False）按平台语义删除空 json。
    """
    n_shapes = n_json = n_images = 0

    def _backup(path: Path) -> None:
        if backup_dir is None:
            return
        bdir = Path(backup_dir)
        bdir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, bdir / path.name)

    for entry in entries:
        if entry.mode == "shapes":
            data = _load_json(entry.json_path)
            keep = [s for i, s in enumerate(data.get("shapes", []))
                    if i not in set(entry.matches)]
            n_shapes += len(entry.matches)
            data["shapes"] = keep
            flags = data.get("flags") or {}
            for lab in entry.labels:      # 同名图片级 flags 键一并清除
                flags.pop(lab, None)
            if not keep and not any(flags.values()):
                _backup(entry.json_path)
                entry.json_path.unlink()
                n_json += 1
            else:
                _backup(entry.json_path)
                _dump_json(entry.json_path, data)
                n_json += 1
        else:
            if entry.has_image and entry.image_path and entry.image_path.exists():
                _backup(entry.image_path)
                entry.image_path.unlink()
                n_images += 1
            if entry.json_path and entry.json_path.exists():
                _backup(entry.json_path)
                entry.json_path.unlink()
                n_json += 1
    return {"shapes": n_shapes, "jsons": n_json, "images": n_images}


# ---------------------------------------------------------------- 迁移


@dataclass
class MoveEntry:
    """一条迁移预览：成对搬运的文件与命中标签（未标注对 labels 为空）。"""

    image_path: Path | None
    json_path: Path | None
    labels: list = field(default_factory=list)
    unlabeled: bool = False


def plan_move(pairs: list, labels: list | None = None, keyword: str = "",
              include_unlabeled: bool = True) -> list:
    """生成迁移预览。

    命中规则：任一形状标签在 labels 集合内，或标签含 keyword（不区分大小写）；
    未标注图（无 json）按 include_unlabeled 纳入；有 json 无图片的对同样可命中。
    """
    want = set(labels or [])
    kw = keyword.strip().lower()
    entries = []
    for pair in pairs:
        pair_labels = iter_labels(pair)
        if not pair.labeled:
            if include_unlabeled:
                entries.append(MoveEntry(image_path=pair.image, json_path=None,
                                         unlabeled=True))
            continue
        hit = bool(want & pair_labels)
        if not hit and kw:
            hit = any(kw in lab.lower() for lab in pair_labels)
        if hit:
            entries.append(MoveEntry(image_path=pair.image, json_path=pair.json,
                                     labels=sorted(pair_labels)))
    return entries


def apply_move(entries: list, dst: Path) -> dict:
    """执行迁移。同名冲突跳过并计数；返回统计供移动报告。"""
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    n_pairs = n_files = skipped = 0
    for entry in entries:
        sources = [p for p in (entry.image_path, entry.json_path)
                   if p is not None and p.exists()]
        if not sources:
            continue
        occupied = any((dst / p.name).exists() for p in sources)
        if occupied:
            skipped += 1
            continue
        for p in sources:
            shutil.move(str(p), str(dst / p.name))
            n_files += 1
        n_pairs += 1
    return {"pairs": n_pairs, "files": n_files, "skipped": skipped}
