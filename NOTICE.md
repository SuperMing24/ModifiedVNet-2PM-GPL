# 来源与修改说明

本项目基于以下 GPL-3.0 上游实现进行 PyTorch 移植：

- Repository: https://github.com/bu-cisl/2PM_Vascular_Segmentation_DNN
- Revision: `e56562d303aedfc90124d5a25fba96f325562a82`
- Upstream title: `2PM Vascular Segmentation DNN`
- Upstream purpose: large-scale cerebral two-photon microscopy angiogram segmentation
- Modification date: 2026-09-05

主要修改：

1. 将 TensorFlow 1.x NDHWC 网络移植为 PyTorch NCDHW；
2. 保留 ReLU/BatchNorm 顺序、卷积偏置、Xavier 初始化和三级 V-Net 结构；
3. 将 balanced BCE 与三维 Sobel total variation 损失移植为 PyTorch；
4. 将 128 立方 patch、stride 64 的 released-code 网格采样适配到 NIfTI；
5. 增加稳定的 CLI、checkpoint 恢复、JSON 运行证据和 NumPy 预测导出；
6. 使用宿主项目预先冻结的 nested train/validation split，不访问 outer-test；
7. 以固定阈值下 Dice3D 与 clDice3D 的调和均值选择 checkpoint。

宿主项目与本仓库仅通过命令行、JSON 和 NumPy 文件交互。宿主不链接或导入本仓库
的 Python 模块；本仓库及其可执行训练产物始终按 GPL-3.0-only 单独分发。

该实现是上游代码的修改和跨语言端口，不声明为独立宽松许可证实现。
