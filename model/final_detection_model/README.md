# 当前单三分类模型

当前模型：**single_three_class_distilled_finetune_h64_seed123**。一个共享 GNN 编码器 → 一个 192 维隐藏层的三分类 MLP 头 → softmax → argmax → 原几何后处理。
参数量 318,217，网络只有一个三路输出层；推理不加载旧双头，没有二分类阈值合并。

固定 17 架训练（48,095 面）、9 架测试（34,162 面）。旧双头只在训练数据上提供蒸馏辅助监督，测试飞机用于教师监督的数量为零。
选中实验 distilled_finetune，固定 200 轮。训练参数和冻结策略见 configuration.json，原始训练日志、完成记录、范围审计和来源说明见 training_record/。

后处理后铆钉 F1=0.9297、饰面 F1=0.7776、均值=0.8537、背景误报 209 面。
旧双头基准分别为 0.9281、0.7414、均值 0.8348、背景误报 291 面。这是已知开发测试集结果，不作为独立泛化验证。

```powershell
D:\Anaconda\envs\cad_graph_env\python.exe python/main.py predict --verify-only
```

重新推理：同一命令去掉 --verify-only，增加 --output-dir work/新的输出目录，目录必须不存在。
重新训练运行 retrain.ps1，会在 work 中的新目录完成选中三分类方案的 200 轮训练与评估，不覆盖正式模型。训练读取 training_assets/ 中的共享编码器初值、归一化和训练集三路辅助分数，不读取旧双头权重。
权重、stats、configuration、manifest 和 evaluation 为同一配套实验；真值始终来自 test_9_models_frozen_truth.csv。
历史候选、旧消融实验和重复预测已按用户要求清理；本轮记录见 work/current_three_class_cleanup_20261019/。此前清理记录见 work/dual_head_cleanup_20261019/REPORT.md。本轮清单取代此前保留历史候选的范围，不删除当前模型依赖。
本轮修改记录见 CHANGELOG_20261019.md。论文、PDF 与 CAD 删除产物本轮未改动。

## 完整配套与依赖

- model.pth、stats.npz：唯一正式分类权重和对应归一化。
- train_17_models.csv、test_9_models_frozen_truth.csv：完整训练数据和冻结测试真值。
- training_assets/：重新训练所需的编码器初值、训练集辅助分数、归一化及身份索引。initial_encoder.pth 仅含编码器初值，不是另一套完整分类模型。
- training_record/：正式模型原始 200 轮训练日志及来源审计。原始比较参数中的旧路径仅用于来源说明，不是运行依赖。
- evaluation/：与正式权重对应的原始预测、后处理预测、逐机输入和汇总指标。
- dc10_window_recovery.json：当前固定几何恢复配置。

项目 python/ 中的训练、图构建、推理、可视化和共用后处理代码，以及 data/plane_model/after_two/ 中的 STEP 文件仍须保留；完整运行代码和九份几何文件的哈希见 manifest.json。本目录与这些项目依赖共同组成可运行系统，不能单独移走本目录后省略项目依赖。

```powershell
# 在项目根目录重新推理，输出目录必须不存在
D:\Anaconda\envs\cad_graph_env\python.exe python/main.py predict --output-dir work/new_three_class_inference
# 使用缓存训练资产重新训练 200 轮并评估
& model/final_detection_model/retrain.ps1
```

独立重训可能因数值差异略有不同，正式指标始终绑定本目录 model.pth。原始标注审计、论文证据摘要和 CAD/网格/图件工作资料继续保留，不作为其他可选模型入口。
