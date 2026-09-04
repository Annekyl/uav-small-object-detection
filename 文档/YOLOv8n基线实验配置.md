# YOLOv8n Baseline 实验配置

## 实验身份

- 消融编号：A0
- 模型：YOLOv8n 预训练权重
- 输入尺寸：640×640
- 数据集：VisDrone2019-DET 官方 train/val/test-dev 划分
- 主随机种子：0
- 训练轮数：200

配置文件为 `configs/train/yolov8n_baseline_640.yaml`。后续结构消融必须继承同一训练配方，仅修改被研究变量。

## 关键决定

- 使用显式 SGD，避免 `optimizer=auto` 随数据和版本改变选择结果；
- 开启 `deterministic=true`，固定 `seed=0`；
- 使用默认风格的 Mosaic、HSV、平移、缩放和水平翻转；
- 最后 10 个 epoch 关闭 Mosaic；
- 不在 Baseline 中加入垂直翻转、MixUp 或自定义小目标增强；
- `max_det=1000`，避免 VisDrone 密集场景被默认 300 个结果上限截断；
- 主实验使用 640，960 作为后续独立分辨率实验。

## 当前电脑配置检查

当前无 GPU 电脑不需要安装 PyTorch，可运行：

```powershell
uv run python scripts/train_baseline.py --dry-run
```

该命令检查：

- train/val/test 的 images 与 labels 是否存在；
- 图像和标签数量是否一致；
- 数据集绝对路径是否正确生成；
- 最终训练参数和 Git commit 是否被记录。

## RTX 4060 默认启动方式

安装并核验 GPU 版 PyTorch 后执行：

```powershell
uv run python scripts/train_baseline.py
```

若显存不足，仅覆盖机器相关参数，不修改受版本管理的主配置：

```powershell
uv run python scripts/train_baseline.py `
  --override batch=8 `
  --override workers=4
```

第一次只运行小规模链路测试：

```powershell
uv run python scripts/train_baseline.py `
  --override epochs=1 `
  --override batch=4 `
  --override workers=2 `
  --override name=A0-smoke-test
```

链路测试不能作为论文实验结果。正式实验开始后不得覆盖已有 run 目录。

## 最终复验

模型结构和训练配方冻结后，A0 与最终模型分别使用 seed 0、1、2 训练，报告均值和标准差。开发阶段的中间消融先使用 seed 0。
