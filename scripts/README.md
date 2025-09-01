# Difix3D 相关脚本

### 训练

- [train_singlegpu.sh](./train_singlegpu.sh)：单卡训练模型
- [train_multigpu.sh](./train_multigpu.sh)：集群上多卡训练模型
- [train_multigpu_finetune.sh](./train_multigpu_finetune.sh)：集群上多卡微调模型(pretrained difix3d_ref model)

### 调试

- [train_debug.sh](./train_debug.sh)：开发机上debug调试

### 评估

- [eval.sh](./eval.sh)：评估训练集/测试集的metrics
- [inference.sh](./inference.sh)：模型推理,用于输出Fix后的图片,作为GT给4DGS训练用
