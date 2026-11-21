"""构造合成数据集供测试使用。"""

import json
from pathlib import Path

import cv2
import numpy as np

from xccv.engine.io_utils import imwrite_unicode


def make_labelme_dataset(root: Path, n_images: int = 6, size=(96, 96),
                         classes=("cat", "dog"), shape_type: str = "rectangle", seed: int = 0):
    """生成 labelme 结构数据集：图片 + 同名 json + classes.txt。"""
    rng = np.random.default_rng(seed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    w, h = size
    for i in range(n_images):
        img = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
        x1, y1 = int(rng.integers(5, w // 3)), int(rng.integers(5, h // 3))
        x2 = min(x1 + int(rng.integers(w // 4, w // 2)), w - 2)
        y2 = min(y1 + int(rng.integers(h // 4, h // 2)), h - 2)
        label = classes[i % len(classes)]
        if shape_type == "rectangle":
            points = [[float(x1), float(y1)], [float(x2), float(y2)]]
        else:
            points = [[float(x1), float(y1)], [float(x2), float(y1)],
                      [float(x2), float(y2)], [float(x1), float(y2)]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), -1)
        name = f"img{i:02d}"
        imwrite_unicode(root / f"{name}.png", img)
        (root / f"{name}.json").write_text(json.dumps({
            "version": "5.8.1", "imagePath": f"{name}.png", "imageData": None,
            "imageWidth": w, "imageHeight": h,
            "shapes": [{"label": label, "shape_type": shape_type, "points": points, "flags": {}}],
        }, ensure_ascii=False), encoding="utf-8")
    (root / "classes.txt").write_text("\n".join(classes), encoding="utf-8")
    return root, list(classes)


def make_classification_dataset(root: Path, classes=("cat", "dog"), per_class: int = 6,
                                size=(48, 48), seed: int = 0):
    """生成分类文件夹结构数据集：{类别}/{图片}。"""
    rng = np.random.default_rng(seed)
    root = Path(root)
    for c in classes:
        d = root / c
        d.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            img = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
            imwrite_unicode(d / f"{c}_{i}.png", img)
    return root, list(classes)

