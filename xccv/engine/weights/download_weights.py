"""一次性脚本：下载全部基准模型权重到本目录（随程序打包分发，离线运行）。

用法：python xccv/engine/weights/download_weights.py
产出：
  torch/hub/checkpoints/*.pth   resnet18 / resnet50 / mobilenet_v3_small（ImageNet 预训练）
  ultralytics/*.pt              yolo11n / yolo11s / yolo11n-seg / yolo11s-seg（COCO 预训练）
"""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ["TORCH_HOME"] = str(HERE / "torch")
os.environ.setdefault("YOLO_CONFIG_DIR", str(HERE / "_ultralytics_cfg"))

YOLO_WEIGHTS = ["yolo11n.pt", "yolo11s.pt", "yolo11n-seg.pt", "yolo11s-seg.pt"]


def main():
    ckpt_dir = HERE / "torch" / "hub" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    import torchvision.models as tvm

    print("== torchvision 预训练权重 ==")
    tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
    tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    for f in sorted(ckpt_dir.glob("*.pth")):
        print(f"  就绪 {f.name}  {f.stat().st_size / 1e6:.0f} MB")

    print("== ultralytics 预训练权重 ==")
    (HERE / "ultralytics").mkdir(exist_ok=True)
    os.chdir(HERE / "ultralytics")
    from ultralytics import YOLO

    for f in YOLO_WEIGHTS:
        YOLO(f)  # 文件不存在时自动从官方源下载到当前目录
        p = HERE / "ultralytics" / f
        print(f"  就绪 {f}  {p.stat().st_size / 1e6:.0f} MB")

    print("全部基准权重下载完成。")


if __name__ == "__main__":
    main()
