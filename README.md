# Tian/Damseh Modified V-Net for 2PM Vessel Segmentation

本仓库是 Lei Tian、Rafat Damseh 等人双光子脑血管分割 Modified V-Net 的独立
PyTorch comparator。它与宿主医学分割项目通过命令行、JSON 和预测文件交互，不允许
宿主项目直接导入本仓库的 Python 模块。

## 来源与许可证

- 上游仓库：https://github.com/bu-cisl/2PM_Vascular_Segmentation_DNN
- 固定上游提交：`e56562d303aedfc90124d5a25fba96f325562a82`
- 本端口许可证：`GPL-3.0-only`
- 许可证全文：`LICENSE`
- 来源和修改边界：`NOTICE.md`

本仓库包含从上游 TensorFlow 实现移植的网络、损失和训练采样协议。重新分发本仓库、
修改版本、容器或可执行训练包时必须遵守 GPL-3.0。

## 冻结协议

`configs/frozen_protocol.json` 固定以下研究协议：

- 199 optimizer epochs；
- Adam，学习率 `1e-4`，`beta=(0.9, 0.999)`；
- 仅卷积权重使用 `L2=0.01`；
- `128 x 128 x 128` patch，训练 stride 为 `64`；
- batch size `4`，FP32，无数据增强；
- 每个 epoch 对 14 个 inner-validation 体积进行完整推理；
- 以 Dice3D 与 clDice3D 的调和均值选择 checkpoint；
- 省略宿主旧管线中不参与权重更新或选模的随机 validation-patch loss 日志；
- 固定阈值 `0.5`；
- 禁止访问 outer-test。

## 运行

```bash
python -m modified_vnet_2pm.cli \
  --protocol configs/frozen_protocol.json \
  --train-root /path/to/fold_1/train \
  --validation-root /path/to/fold_1/val \
  --assignment-manifest /path/to/split_manifest.json \
  --expected-assignment-sha256 86db22755ded10c71486ea16c243ad0d1ab66e35ef0c0726036cbfc84f856fea \
  --fold 1 \
  --output-root /path/to/results/fold_1 \
  --expected-commit $(git rev-parse HEAD)
```

训练器只接收并读取两个显式目录 `fold_N/train` 和 `fold_N/val`，不会接收或枚举
完整 fold 根目录。完成后输出：

- `training_result.json`
- `training_history.jsonl`
- `checkpoints/best_model.pt`
- `checkpoints/latest_model.pt`
- `predictions/<case>.npy`

最终论文统一指标由宿主项目从 `.npy` 预测文件独立计算。本仓库不读取 outer-test，
也不产生模型优越性结论。

进程边界是本项目的分发约束：宿主不得将 `modified_vnet_2pm` 作为 Python 依赖导入，
也不得把本仓库源码复制回宿主代码树。若未来改成同进程插件或复制实现，需要重新进行
GPL 合规审查。

## 验证

```bash
python -m pytest -q
```

如已取得固定上游仓库和 TensorFlow checkpoint，可运行：

```bash
MODIFIED_VNET_UPSTREAM=/path/to/2PM_Vascular_Segmentation_DNN \
python tools/tensorflow_checkpoint_parity.py
```

既有 parity 审计在 TensorFlow 2.15.1 CPU 与 PyTorch 2.9.1 CPU 间得到
`max_abs_diff=3.4332275390625e-05`、二值 logit 一致率 `1.0`。
