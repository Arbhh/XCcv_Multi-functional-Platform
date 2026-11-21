# -*- coding: utf-8 -*-
"""mmdet r18-FPN → xccv slim 权重转换器。

用法：
  python tools/convert_mmdet_r18.py <权重文件> [--out weights/maskrcnn18_slim_coco.pt]
  python tools/convert_mmdet_r18.py --selftest     # 映射表合成往返自证（无需真权重）

源支持：明文 torch 文件（导出的 .pth，含 "state_dict" 或直接是 state_dict）：
mmdet 2.x/3.x 键名 → torchvision slim 结构 strict 加载。

映射表依据结构取证：
R18 + FPN(lateral 64/128/256/512→64) + RPN(64) + Shared2FC(256) + mask 4×64
——与 xccv rcnn.py maskrcnn18+slim_head 结构一一对应（键名见 _MAP）。

产物为明文 .pt（平台自建 init 权重优先；本转换产物仅作兜底）。
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# mmdet 键名 → xccv slim 键名（{i} 为层序号占位）。
# ResNet 骨干仅加 "body." 前缀（BN 四件套同名，FrozenBatchNorm2d 兼容）。
_MAP = [
    (re.compile(r"^backbone\."), "backbone.body."),
    (re.compile(r"^neck\.lateral_convs\.(\d+)\.conv\."),
     r"backbone.fpn.inner_blocks.\1.0."),
    (re.compile(r"^neck\.fpn_convs\.(\d+)\.conv\."),
     r"backbone.fpn.layer_blocks.\1.0."),
    # torchvision 0.22 RPNHead.conv=Conv2dNormActivation → 键为 conv.0.0
    (re.compile(r"^rpn_head\.rpn_conv\."), "rpn.head.conv.0.0."),
    (re.compile(r"^rpn_head\.rpn_cls\."), "rpn.head.cls_logits."),
    (re.compile(r"^rpn_head\.rpn_reg\."), "rpn.head.bbox_pred."),
    # Shared2FC 头为 shared_fcs.0/1
    (re.compile(r"^roi_head\.bbox_head\.shared_fcs\.0\."),
     "roi_heads.box_head.fc6."),
    (re.compile(r"^roi_head\.bbox_head\.shared_fcs\.1\."),
     "roi_heads.box_head.fc7."),
    (re.compile(r"^roi_head\.bbox_head\.fc_cls\."),
     "roi_heads.box_predictor.cls_score."),
    (re.compile(r"^roi_head\.bbox_head\.fc_reg\."),
     "roi_heads.box_predictor.bbox_pred."),
    (re.compile(r"^roi_head\.mask_head\.convs\.(\d+)\.conv\."),
     r"roi_heads.mask_head.\1.0."),
    (re.compile(r"^roi_head\.mask_head\.upsample\."),
     "roi_heads.mask_predictor.conv5_mask."),
    (re.compile(r"^roi_head\.mask_head\.conv_logits\."),
     "roi_heads.mask_predictor.mask_fcn_logits."),
]


def map_key(k: str) -> str:
    for pat, repl in _MAP:
        if pat.search(k):
            return pat.sub(repl, k)
    return ""


def load_plain(path: Path) -> dict:
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return dict(ckpt["state_dict"])
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("非 state_dict 权重文件")


def num_classes_of(sd: dict) -> int:
    for k, v in sd.items():
        # mmdet 3.x=fc_cls；torchvision=cls_score
        if k.endswith(("cls_score.weight", "fc_cls.weight")):
            return int(v.shape[0]) - 1
    raise ValueError("权重缺 fc_cls/cls_score，无法确定类别数")


def convert(src: Path, out: Path) -> dict:
    import torch

    sd = load_plain(src)
    from xccv.engine.dataset.base import TaskType
    from xccv.engine.models.rcnn import RCNNModel

    nc = num_classes_of(sd)
    net = RCNNModel("maskrcnn18", TaskType.SEGMENTATION, nc, device="cpu",
                    use_pretrained=False)._build(
        "maskrcnn18", nc, 640, use_pretrained=False, slim_head=True)
    model_sd = net.state_dict()
    mapped, unmatched = {}, []
    for k, v in sd.items():
        nk = map_key(k)
        if not nk:
            unmatched.append(k)
            continue
        if nk in model_sd and model_sd[nk].shape != v.shape:
            if nk.startswith(("roi_heads.box_predictor.bbox_pred",
                              "roi_heads.mask_predictor.mask_fcn_logits")):
                pass    # 背景行语义差异，稍后零填充适配
            else:
                raise SystemExit(f"形状不匹配：{k} {tuple(v.shape)} → "
                                 f"{nk} {tuple(model_sd[nk].shape)}")
        mapped[nk] = v
    # 背景行语义适配：mmdet 的 fc_reg 不含背景（类别×4），torchvision
    # bbox_pred 含背景（(类+1)×4）；mask logits 同理（mmdet 无背景行）。
    # 差行以零填充插在最高位（背景），其余行原值迁移。
    import torch as _t

    for nk in ("roi_heads.box_predictor.bbox_pred",
               "roi_heads.mask_predictor.mask_fcn_logits"):
        for suffix in ("weight", "bias"):
            k = nk + "." + suffix
            if k not in mapped or mapped[k].shape == model_sd[k].shape:
                continue
            tv, sv = model_sd[k], mapped[k]
            if tv.shape[1:] != sv.shape[1:] or tv.shape[0] < sv.shape[0]:
                raise SystemExit(f"背景行适配失败：{k} "
                                 f"{tuple(sv.shape)} → {tuple(tv.shape)}")
            pad = tv.shape[0] - sv.shape[0]
            mapped[k] = _t.cat([_t.zeros(pad, *sv.shape[1:]), sv], 0)
            print(f"背景行零填充：{k} {tuple(sv.shape)} → {tuple(tv.shape)}")
    missing = [k for k in model_sd if k not in mapped]
    if unmatched:
        print("警告：未映射的源键（跳过）：", unmatched[:8])
    if missing:
        raise SystemExit(f"映射后仍缺目标键（strict 失败）：{missing[:8]}")
    net.load_state_dict(mapped, strict=True)
    ckpt = {"format": "xccv-rcnn", "arch": "maskrcnn18",
            "task": TaskType.SEGMENTATION.value,
            "classes": [f"class_{i}" for i in range(nc)],
            "min_size": 640, "slim_head": True,
            "industrial_opt": False, "mask_head_light": False,
            "state_dict": {k: v.cpu() for k, v in net.state_dict().items()}}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out) + ".part")
    try:
        torch.save(ckpt, tmp)
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    n = sum(p.numel() for p in net.parameters())
    return {"src": str(src), "out": str(out), "classes": nc,
            "params": n, "bytes": out.stat().st_size}


def to_mmdet_for_test(k: str) -> str:
    """slim 键名 → mmdet 键名（selftest/单测共用的逆映射，与 _MAP 一一对应）。"""
    s = k
    s = re.sub(r"^backbone\.body\.", "backbone.", s)
    s = re.sub(r"^backbone\.fpn\.inner_blocks\.(\d+)\.0\.",
               r"neck.lateral_convs.\1.conv.", s)
    s = re.sub(r"^backbone\.fpn\.layer_blocks\.(\d+)\.0\.",
               r"neck.fpn_convs.\1.conv.", s)
    s = re.sub(r"^rpn\.head\.conv\.0\.0\.", "rpn_head.rpn_conv.", s)
    s = re.sub(r"^rpn\.head\.cls_logits\.", "rpn_head.rpn_cls.", s)
    s = re.sub(r"^rpn\.head\.bbox_pred\.", "rpn_head.rpn_reg.", s)
    s = re.sub(r"^roi_heads\.box_head\.fc6\.",
               "roi_head.bbox_head.shared_fcs.0.", s)
    s = re.sub(r"^roi_heads\.box_head\.fc7\.",
               "roi_head.bbox_head.shared_fcs.1.", s)
    s = re.sub(r"^roi_heads\.box_predictor\.cls_score\.",
               "roi_head.bbox_head.fc_cls.", s)
    s = re.sub(r"^roi_heads\.box_predictor\.bbox_pred\.",
               "roi_head.bbox_head.fc_reg.", s)
    s = re.sub(r"^roi_heads\.mask_head\.(\d+)\.0\.",
               r"roi_head.mask_head.convs.\1.conv.", s)
    s = re.sub(r"^roi_heads\.mask_predictor\.conv5_mask\.",
               "roi_head.mask_head.upsample.", s)
    s = re.sub(r"^roi_heads\.mask_predictor\.mask_fcn_logits\.",
               "roi_head.mask_head.conv_logits.", s)
    return s


def selftest() -> None:
    """映射表自证：slim 权重按 _MAP 逆映射成 mmdet 键名，再正向转换回。

    逆映射构造源（全部键可逆）→ convert 语义（map→strict load）必须零缺失
    零未匹配——证明映射表覆盖且无歧义。
    """
    import torch

    from xccv.engine.dataset.base import TaskType
    from xccv.engine.models.rcnn import RCNNModel

    nc = 3
    net = RCNNModel("maskrcnn18", TaskType.SEGMENTATION, nc, device="cpu",
                    use_pretrained=False)._build(
        "maskrcnn18", nc, 640, use_pretrained=False, slim_head=True)

    to_mmdet = to_mmdet_for_test

    fake = {to_mmdet(k): v for k, v in net.state_dict().items()}
    bad = [k for k in fake if not map_key(k)]
    assert not bad, f"逆映射键无正向规则: {bad[:5]}"
    roundtrip = {map_key(k): v for k, v in fake.items()}
    model_sd = net.state_dict()
    assert set(roundtrip) == set(model_sd), (
        f"覆盖差 missing={sorted(set(model_sd) - set(roundtrip))[:5]} "
        f"extra={sorted(set(roundtrip) - set(model_sd))[:5]}")
    for k, v in roundtrip.items():
        assert v.shape == model_sd[k].shape, k
    net.load_state_dict(roundtrip, strict=True)
    n = sum(p.numel() for p in net.parameters())
    assert 11.2e6 < n < 13.8e6, n
    print(f"SELFTEST OK：映射表 {len(_MAP)} 条规则往返一致，参数量 {n/1e6:.2f}M")


def main() -> int:
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--selftest" in sys.argv or not argv:
        selftest()
        return 0
    from xccv.engine.offline import WEIGHTS_DIR

    src, out = Path(argv[0]), Path(
        argv[1] if len(argv) > 1 else
        WEIGHTS_DIR / "maskrcnn18_slim_coco.pt")
    info = convert(src, out)
    print(json.dumps(info, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

