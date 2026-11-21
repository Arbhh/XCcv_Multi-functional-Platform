"""轮次权重存档：每次训练共用的权重文件夹，逐轮保存、新训练覆盖。

- 目录：<模型储存路径>/epoch_weights，同一项目所有训练共用一个文件夹；
- 训练启动时清空旧文件（新训练直接覆盖旧权重，磁盘占用只保留最近一次训练）；
- 每轮结束把权重序列化为明文 .pt（原子写：先写 .part 再替换，进程中断不留
  半截文件）。
"""

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

# 共用文件夹名（模型储存路径下）；scan_runs 等目录扫描需跳过它
EPOCH_DIR_NAME = "epoch_weights"


class EpochWeightStore:
    """轮次权重写入器：reset() 清空共用目录，save() 明文落盘。"""

    def __init__(self, model_dir):
        self.dir = Path(model_dir) / EPOCH_DIR_NAME

    def reset(self) -> None:
        """训练启动时调用：清空上一训练留下的轮次权重（覆盖式）。"""
        self.dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in self.dir.iterdir():
            if f.is_file():
                f.unlink()
                n += 1
        if n:
            log.info("轮次权重文件夹已清空旧内容: %s（%s 个文件）", self.dir, n)

    def save(self, epoch: int, payload) -> Path:
        """保存第 epoch 轮权重为明文 .pt（原子替换）。

        目录可能被外部清理（用户删除、同步盘冲突）：自动重建，保证训练
        不因轮次存档中断。
        """
        import torch

        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"epoch_{epoch:03d}.pt"
        tmp = path.with_name(path.name + ".part")
        try:
            with open(tmp, "wb") as f:
                torch.save(payload, f)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        log.debug("轮次权重已保存: %s (%.1f MB)",
                  path.name, path.stat().st_size / 1e6)
        return path

