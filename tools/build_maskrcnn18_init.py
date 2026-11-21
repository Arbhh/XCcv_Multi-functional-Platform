# -*- coding: utf-8 -*-
"""自建 maskrcnn18 slim 初始权重（纯 torch 初始化配方）。

结构 = xccv rcnn.py maskrcnn18 + slim_head（与 mmdetection 的 r18-FPN 同构，
约 12.45M）：ResNet-18(ImageNet, FrozenBN) + FPN64 + RPN(64) + TwoMLPHead
(3136→256→256) + FastRCNNPredictor(256) + MaskRCNNHeads 4×64 + MaskRCNNPredictor。

初始化配方：
- FPN inner/layer 卷积：kaiming_normal_(fan_out, relu)，bias=0；
- RPN 全部卷积：N(0, 0.01)，bias=0；
- box 头 fc6/fc7：torch 默认；
- box_predictor：cls_score N(0, 0.01) / bbox_pred N(0, 0.001)，bias=0；
- mask 卷积与转置卷积：kaiming；mask_fcn_logits N(0, 0.001)，bias=0。

产出：weights/maskrcnn18_slim_init.pt（明文 .pt，xccv-rcnn/slim_head=True）。
训练面板勾选「slim 头 FPN64」+ use_pretrained 时由 _load_slim_pretrained
自动注入（本脚本产物优先；slim_coco 转换真权重仅最后兜底）。

用法：
  python tools/build_maskrcnn18_init.py            # 构建并落盘（含自检）
  python tools/build_maskrcnn18_init.py --out X    # 指定输出
  python tools/build_maskrcnn18_init.py --no-imagenet   # 不下载 ImageNet（测试用）
"""

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NC = 80          # COCO 类数口径（注入时类别头按形状匹配，不匹配自动跳过）
MIN_SIZE = 640   # 训练输入下限口径


def build_init_net(num_classes: int = NC, imagenet: bool = True,
                   v2: bool = True):
    """构建 slim 网络并套用初始化配方。返回 (net, notes dict)。"""
    import torch

    from xccv.engine.dataset.base import TaskType
    from xccv.engine.models.rcnn import RCNNModel

    torch.manual_seed(42)
    m = RCNNModel("maskrcnn18", TaskType.SEGMENTATION, num_classes,
                  device="cpu", use_pretrained=False)
    # _build 内部已 num_classes+1（加背景类）——传原始类别数，勿手动 +1
    net = m._build("maskrcnn18", num_classes, MIN_SIZE,
                   use_pretrained=imagenet, slim_head=True)
    notes = {"backbone": "random"}
    if imagenet:
        # _build 已载入 IMAGENET1K_V1；尝试升级 V2（top-1 73.3 vs 69.8）
        if v2:
            try:
                from torchvision.models import resnet18, ResNet18_Weights
                from torchvision.ops.misc import FrozenBatchNorm2d

                last = None
                for _ in range(2):   # 下载抖动重试一次
                    try:
                        body2 = resnet18(
                            weights=ResNet18_Weights.IMAGENET1K_V2,
                            norm_layer=FrozenBatchNorm2d)
                        sd = {k: v for k, v in body2.state_dict().items()
                              if not k.startswith("fc.")}
                        missing = net.backbone.body.load_state_dict(
                            sd, strict=False)
                        assert not missing.missing_keys, missing
                        notes = {"backbone": "imagenet_v2"}
                        last = None
                        break
                    except Exception as e:
                        last = e
                if last is not None:  # 无网/缓存失败 → 保持 V1
                    notes = {"backbone": "imagenet_v1",
                             "v2_skip": str(last)[:80]}
            except Exception as e:  # 兜底：升级路径任何异常都保持 V1
                notes = {"backbone": "imagenet_v1", "v2_skip": str(e)[:80]}
        else:
            notes = {"backbone": "imagenet_v1"}
    # 配方域重置种子：V2 升级会多构造一次 resnet18（body2）消耗 RNG，导致
    # 配方初始化序列随骨干路径漂移——重置后配方结果只由种子决定（确定性）。
    torch.manual_seed(42)
    _apply_recipe(net)
    return net, notes


def _apply_recipe(net) -> dict:
    """初始化配方；返回各规则覆盖的参数键前缀组（审计用）。"""
    import torch
    import torch.nn as nn

    audit = {"fpn": [], "rpn": [], "box_pred": [], "mask_head": [],
             "mask_pred": []}

    def kaiming(w, b=None):
        nn.init.kaiming_normal_(w, mode="fan_out", nonlinearity="relu")
        if b is not None:
            nn.init.zeros_(b)

    def normal01(w, b=None, std=0.01):
        nn.init.normal_(w, 0.0, std)
        if b is not None:
            nn.init.zeros_(b)

    # FPN（inner/layer blocks：Conv2dNormActivation → conv 是 .0 子模块）
    for name, mod in net.backbone.fpn.named_modules():
        if hasattr(mod, "weight") and mod.weight.dim() == 4:
            kaiming(mod.weight, mod.bias)
            audit["fpn"].append(name)
    # RPN 头（conv + cls_logits + bbox_pred：全部 N(0, 0.01)）
    for name, p in net.rpn.head.named_parameters():
        if name.endswith("weight"):
            normal01(p)
            audit["rpn"].append(name)
        elif name.endswith("bias"):
            nn.init.zeros_(p)
            audit["rpn"].append(name)
    # box 头 fc6/fc7：torch 默认（不触）
    # box 预测头
    normal01(net.roi_heads.box_predictor.cls_score.weight,
             net.roi_heads.box_predictor.cls_score.bias, 0.01)
    normal01(net.roi_heads.box_predictor.bbox_pred.weight,
             net.roi_heads.box_predictor.bbox_pred.bias, 0.001)
    audit["box_pred"] += ["cls_score", "bbox_pred"]
    # mask 头卷积 kaiming（键形态 0.0.weight / 0.0.bias）
    for name, p in net.roi_heads.mask_head.named_parameters():
        if name.endswith("weight"):
            kaiming(p)
            audit["mask_head"].append(name)
        elif name.endswith("bias"):
            nn.init.zeros_(p)
            audit["mask_head"].append(name)
    # mask 预测头：conv5_mask（转置卷积）kaiming；mask_fcn_logits N(0, 0.001)
    mp = net.roi_heads.mask_predictor
    kaiming(mp.conv5_mask.weight, mp.conv5_mask.bias)
    normal01(mp.mask_fcn_logits.weight, mp.mask_fcn_logits.bias, 0.001)
    audit["mask_pred"] += ["conv5_mask", "mask_fcn_logits"]
    return audit


def _sd_sha(sd: dict) -> str:
    """按 键名+张量字节 哈希（torch.save 容器含时间戳等非确定字节，不可用）。"""
    h = hashlib.sha256()
    for k in sorted(sd):
        v = sd[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()


def run_checks(net, notes: dict, imagenet: bool = True, v2: bool = True) -> dict:
    """六检（无数据部分）：形状/stride、参数量、特征统计、参数审计、确定性。"""
    import torch

    report = {"notes": notes}
    # 1) 前向形状/通道/stride
    x = torch.randn(1, 3, 352, 480)
    with torch.no_grad():
        feats = net.backbone(x)
    expect = {"0": (64, 4), "1": (64, 8), "2": (64, 16), "3": (64, 32)}
    for k, (c, s) in expect.items():
        assert k in feats and feats[k].shape[1] == c, (k, feats.get(k))
        assert feats[k].shape[-1] == x.shape[-1] // s, (k, feats[k].shape)
    assert "pool" in feats
    # 2) 参数量预算
    n = sum(p.numel() for p in net.parameters())
    assert 11.2e6 < n < 13.8e6, f"参数量 {n/1e6:.2f}M 超出 12~13M±10%"
    report["params_m"] = round(n / 1e6, 2)
    # 3) 特征统计（无 NaN、量级 sane）
    stds = [float(f.std()) for f in feats.values()]
    assert all(s == s for s in stds), "特征含 NaN"
    report["feat_stds"] = [round(s, 3) for s in stds]
    # 4) 初始化配方落地断言
    mp = net.roi_heads.mask_predictor
    assert float(mp.mask_fcn_logits.weight.std()) < 0.002, "mask logits 初始化未生效"
    assert float(net.roi_heads.box_predictor.cls_score.weight.std()) < 0.012
    assert float(net.rpn.head.cls_logits.weight.std()) < 0.012
    assert float(net.roi_heads.box_predictor.cls_score.bias.abs().max()) == 0.0
    # 5) 确定性（同种子重建：配方域=骨干以外哈希一致。骨干是固定下载权重，
    #    V2 下载存在网络抖动——首次落 V1 复建落 V2 会造成假不一致，故排除）
    net2, _ = build_init_net(num_classes=_nc_of(net), imagenet=False)
    keys = sorted(k for k in net.state_dict()
                  if not k.startswith("backbone.body."))
    h1 = _sd_sha({k: net.state_dict()[k] for k in keys})
    h2 = _sd_sha({k: net2.state_dict()[k] for k in keys})
    if h1 != h2:
        import torch as _t

        bad = [k for k in keys
               if not _t.equal(net.state_dict()[k], net2.state_dict()[k])]
        raise AssertionError(f"配方域确定性破坏: {bad[:8]} "
                             f"(h1={h1[:8]} h2={h2[:8]})")
    report["sha256"] = h1[:12]
    # 6) 纯前向一遍 MaskRCNN 训练态（小尺寸，CPU 数秒）确认无异常
    with torch.no_grad():
        net.eval()
        images = [torch.randn(3, 320, 320)]
        net(images)  # eval 推理冒烟
    return report


def _nc_of(net) -> int:
    return int(net.roi_heads.box_predictor.cls_score.out_features) - 1


def save_artifact(net, out: Path) -> dict:
    """权重容器 → 明文 .pt 落盘（原子写）。"""
    import os

    import torch

    from xccv.engine.dataset.base import TaskType

    nc = _nc_of(net)
    ckpt = {"format": "xccv-rcnn", "arch": "maskrcnn18",
            "task": TaskType.SEGMENTATION.value,
            "classes": [f"class_{i}" for i in range(nc)],
            "min_size": MIN_SIZE, "slim_head": True,
            "industrial_opt": False, "mask_head_light": False,
            "init_recipe": "chatglm_build_maskrcnn18_v1",
            "state_dict": {k: v.detach().cpu()
                           for k, v in net.state_dict().items()}}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "best.pt"
        torch.save(ckpt, tmp)
        data = tmp.read_bytes()
    tmp_out = Path(str(out) + ".part")
    with open(tmp_out, "wb") as f:
        f.write(data)
    os.replace(tmp_out, out)
    return {"bytes": len(data), "dst": str(out)}


def main() -> int:
    argv = [a for a in sys.argv[1:]]
    from xccv.engine.offline import WEIGHTS_DIR

    out = WEIGHTS_DIR / "maskrcnn18_slim_init.pt"
    if "--out" in argv:
        out = Path(argv[argv.index("--out") + 1])
    imagenet = "--no-imagenet" not in argv
    net, notes = build_init_net(imagenet=imagenet)
    report = run_checks(net, notes, imagenet=imagenet, v2=("--no-v2" not in argv)) if imagenet else {"notes": notes}
    info = save_artifact(net, out)
    print(json.dumps({"out": str(out), **report, "bytes": info["bytes"]},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

