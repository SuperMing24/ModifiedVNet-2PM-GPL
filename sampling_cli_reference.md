# Modified V-Net 随机采样 CLI 与恢复契约

日期：2026-09-11。工作流：独立 GPL 仓库 `auto/random-sampling`。
起点：`de7a2e9528c2f7d7da362de22ede1b84a87813b6`。
本文件描述工程接口，不冻结采样科学协议，不授权 GPU 或测试集访问。

## 保持不变

网络、损失、优化器、数据归一化和推理实现沿用已审计版本。
[冻结配方](configs/frozen_protocol.json)仍为 199 epochs、FP32、batch 4、
128³ patch、stride 64、Adam、原权重衰减和 BCE/TV 参数，逐轮内部验证选模，阈值 0.5。
没有新增提前停止、学习率调度、增广或预训练权重。
历史配置的 seed 42 保留；采样模式使用冻结成员清单中的 `training_seed`，并绑定恢复身份。
宿主仅通过 CLI、JSON 和 NumPy 预测文件交互，不复制或导入本仓库 GPL 实现。

## 命令接口

沿用原训练 CLI，采样模式新增三个参数，必须同时提供：

```bash
python -m modified_vnet_2pm.cli \
  --protocol /checkout/configs/frozen_protocol.json \
  --train-root /unit/fold_0/train \
  --validation-root /unit/fold_0/val \
  --assignment-manifest /dataset/split_manifest.json \
  --expected-assignment-sha256 ASSIGNMENT_SHA256 \
  --fold 0 \
  --output-root /runs/unique_unit \
  --expected-commit GPL_SAMPLING_COMMIT \
  --sampling-manifest /frozen/sample_usage.json \
  --expected-sampling-manifest-sha256 LEDGER_FILE_SHA256 \
  --subset-key f0_p80_r0001
```

示例路径和大写值是调用方必须替换的占位符，不能直接提交。
`--expected-commit` 是本采样分支最终完整提交号，不是旧 `main` 的起点。
不传三个采样参数时保留旧全数据入口的 42/14/2058 检查；采样参数不完整时拒绝。
未新增采样测试集 CLI；原 `predict_cli` 未修改，不能假设它适用于新的统计控制器协议。

## 冻结成员清单

使用宿主 `sample_usage.json` 的以下字段，不读取任何性能观测：

| 字段 | 约束 |
| --- | --- |
| `schema_version`、`frozen` | 必须为 1、true；preview 明确拒绝 |
| `protocol_version` | 非空版本字符串 |
| `source_sha256` | 必须等于原 assignment 文件原始字节 SHA256 |
| `assignment_sha256`、`outer_fold` | 与命令行和父划分一致 |
| `training_seed` | 整数，从 0 至 4294967295；不接受布尔值或小数 |
| `fixed_splits.train_pool` | 42 个唯一父训练 ID |
| `fixed_splits.validation` | 14 个固定验证 ID |
| `fixed_splits.outer_test` | 14 个排除用 ID，仅作元数据核对，不打开其数据 |
| `subsets[].subset_key` | 命令指定键必须恰好对应一条记录 |
| `fraction_percent` | 80、60、40、20，分别对应 34、25、17、8 个训练成员 |
| `outer_fold`、`repeat` | fold 一致，repeat 为正整数 |
| `train_ids`、`val_ids`、`test_ids` | 训练必须是父训练池子集；验证和排除测试成员保持固定 |

三个父集合必须两两不相交。每个训练/验证目录均须包含 `images` 和 `masks`，
NIfTI 文件名必须与对应成员集合精确一致；缺失、多余或同 ID 同时存在 `.nii`/`.nii.gz` 均拒绝。
每个采样单元使用独立目录和输出位置。解析到 `test` 或 `outer-test` 目录的输入文件拒绝。
数据布局和文件内容的不可变性由宿主部署审计保证；本接口不重新证明动物独立性或原始内容无重复。

清单的完整文件 SHA256 绑定来源；本仓库另计算排序后 `fold/train_ids/val_ids` 的规范 JSON
成员哈希，记录为 `execution_identity.sampling.membership_sha256`。
这个内部哈希不冒充宿主清单中可能采用其他序列化规则的同名哈希。
运行身份还绑定清单版本与哈希、父清单哈希、子集键、比例、repeat、seed、配方哈希及 GPL 提交。
`frozen=true` 不是预算或作业授权；全局登记、当前阶段许可和预算必须由宿主提交器独立检查。

## 动态训练计数

仍使用原确定性网格及每轮 shuffle，不改为随机 patch 或前景过采样。
按实际加载的训练体积计算 patch 数和 batch 数，不再对采样模式硬断言 2058 patches。
`training_result.json` 记录实际训练/验证体积数、patch 数、每轮 batch 数、有效 seed 和身份。
训练数量变化不会改变 199 轮和每轮固定验证成本，不能直接按病例比例线性估算整体时长。

## 中断恢复

采样 checkpoint 使用 schema 2，并保持推理所需 `model_state_dict`、`epoch` 等字段。
`checkpoints/latest_model.pt` 是唯一的已提交 epoch 状态，保存模型、Adam、Python/NumPy/
Torch CPU/CUDA RNG、完整执行身份、实际 patch 数、最佳权重和从第 1 轮起的完整历史。

每轮先原子替换 latest，再重建 `best_model.pt` 和 `training_history.jsonl`。
中断发生在 latest 写入前时重做尚未提交的整轮；写入后则从下一轮继续并修复衍生文件。
不会在轮中恢复某一个 batch，不会将未完成轮计为完成轮。
最终预测导出中断时保留已完成训练，并从固定最佳权重重新导出验证预测，不再训练。
重放的计算仍属于实际预算成本，不能因为没有形成新 epoch 而从耗时中扣除。

恢复先在 CPU 加载 checkpoint，再将参数和优化器状态交给目标模型。
CPU/CUDA RNG 状态张量显式回到 CPU；CUDA 状态缺失或可见设备数不同会拒绝。
身份、实际 patch 数、历史连续性和最佳 checkpoint 不一致也会拒绝。
latest 缺失但存在 best、历史或已完成进度时，不静默回退到 best，也不自动从零重训。
已完成结果重复调用时核对身份及 checkpoint/预测文件哈希，然后返回原结果。

每个输出目录使用操作系统锁，进程被终止后锁由内核释放；不会删除另一个进程的锁。
Windows 文件替换遇短暂共享占用时最多尝试五次，等待总计不超过 0.75 秒；
这只重试原子文件替换，不自动重提作业或增加实验次数。
原子重命名与文件锁依赖部署文件系统语义；本轮未验证集群共享文件系统或断电持久性。

## 验证与限制

测试只使用合成 ID、临时空文件和小型 CPU 网络，没有加载真实体积或评分，没有 GPU 作业。
CPU 对照覆盖五种中断位置，比较恢复后的模型、优化器、随机状态、最佳权重和历史；
测试中的短训练长度仅通过测试替身设置，正式 CLI 的冻结配方仍为 199 轮。
GPU RNG 设备处理通过 CPU mock 检查，不声称已完成真实 CUDA 恢复等价测试。
正式执行必须继续使用已资格硬件和固定环境；不同设备/框架版本间不承诺位级等价。

本地测试使用既有 vesseg 的 Python 3.9 源码入口，不安装或改动环境；
包元数据仍声明 Python >=3.10，本轮没有资格验证包安装过程。
完整测试结果、首轮失败及修复、代码边界审计见[本分支状态清单](sampling_branch_state_20260911.json)。
Markdown 内容与源码检查单独记录；未实测 VS Code/Codex 原生预览，不将打开文件算作验收。
