# gjy_byproject

本项目用于飞机 STEP 面级小特征识别、预测结果审查和经过人工确认后的 CAD 面删除。

当前正式主线为：

```text
STEP → 几何/拓扑特征 → 三分类预测 → surface 后处理
     → 铆钉/表面特征删除 → 模型专用几何修复 → ending 最终结果
```

## 分类约定

- `0 = background`
- `1 = rivet`
- `2 = surface_feature`：原生贴花、原生窗户和后续注入的表面特征

大面积机身、机翼和发动机壳体不能直接成为删除候选。最终删除前必须先完成预测值/真值可视化检查。

## 核心文件

训练与推理模型：

- 训练集：`model/final_detection_model/train_17_models.csv`
- 测试集：`model/final_detection_model/test_9_models_frozen_truth.csv`
- 权重：`model/final_detection_model/model.pth`
- stats：`model/final_detection_model/stats.npz`
- 配套清单：`model/final_detection_model/manifest.json`，记录数据、几何和模型哈希
- 当前识别结果：`model/final_detection_model/evaluation/`（9 架飞机，34,162 面）
- 当前选择与训练记录：同目录的 `selection.json`、`configuration.json` 和 `CHANGELOG_20261019.md`

当前选定 `single_three_class_distilled_finetune_h64_seed123`：一个共享图编码器、一个 192 维隐藏层的三分类 MLP 头，softmax 后取最大概率类别，再执行原几何后处理。后处理后铆钉 F1 为 0.9297、饰面 F1 为 0.7776、两目标平均 F1 为 0.8537，背景误报 209 面。旧双头基准分别为 0.9281、0.7414、平均 0.8348、背景误报 291 面。本版两类 F1 均超过该基准；饰面误报减少，但比旧双头多漏 4 面。固定测试集已用于开发，这些结果不作为独立泛化验证，也不能将提升单独归因于分类头结构。

检查唯一正式数据和结果：

```powershell
D:\Anaconda\envs\cad_graph_env\python.exe python/main.py predict --verify-only
```

重新推理并从同一测试 CSV 的真值计算指标：

```powershell
D:\Anaconda\envs\cad_graph_env\python.exe python/main.py predict --output-dir work/recognition_replay
```

输出目录必须不存在，避免混用已有结果。历史候选模型、旧消融实验、重复预测及专用实验脚本已按用户要求清理，只保留当前正式模型与完整训练、推理依赖。正式模型的原始训练记录位于 `model/final_detection_model/training_record/`，本轮删除与保留清单位于 `work/current_three_class_cleanup_20261019/`。`work/recognition_archive_20261019/` 仅保留原始标签审计和论文证据摘要，其旧移动清单不是现存备份。论文、CAD 端点和网格/QEM结果已统一为当前模型的 2026-10-20 重跑证据；论文交付位于 `paper/manuscript/`，重跑证据位于 `results/current_three_class_complete_20261020/`。

飞机数据目录：

- `data/plane_model/source`：原始 STEP 输入
- `data/plane_model/new_data`：中间几何处理和特征注入结果
- `data/plane_model/after_two`：预测值/真值对比使用的最终验证 STEP
- `data/plane_model/removed`：铆钉删除后的中间 STEP
- `data/plane_model/removed_surface`：表面特征处理候选和历史验证结果
- `data/plane_model/ending`：最终删除后的可视化 STEP
- `data/plane_model/label`：与 STEP 对齐的标签和标注文件

当前已完成验证的模型包括：109、DC-10、747-400、Air Plane Idea A、Airbus、Airplane body、AULIRA 2、Cessna Citation 2 和 Gulfstream G280 v17。

## 训练

训练按完整飞机划分，避免同一飞机的相邻面同时出现在训练集和测试集。历史训练使用双头迁移的共享编码器与训练集教师分数；目前已将相同编码器初值和训练集三路辅助分数独立保存到 `training_assets/`，重训不再读取旧双头模型。仍使用 17 架真实训练标签；前 40 轮冻结编码器，再以低学习率联合微调 160 轮。有效批量 8、图微批量 1、共享宽度 64、头隐藏层 192、种子 123。实际训练参数保存在 `configuration.json`。完整重训命令为：

```powershell
& .\model\final_detection_model\retrain.ps1
```

重训输出写入带时间戳的 `work/three_class_refinement_*`，完成选中微调方案的训练与评估，不覆盖正式权重。原始训练日志、参数和来源审计见 `model/final_detection_model/training_record/`，本轮清理记录见 `work/current_three_class_cleanup_20261019/`。共用三分类代码位于 `python/recognition/model.py`。推理不使用二分类阈值。微批量按监督面数加权累积梯度，每个有效批量更新一次参数。

## 当前 CAD、网格与论文

唯一当前入口为 `python/main.py cad --stage cad`，使用正式三分类权重、同一后处理和重新导出的中间面图，不复用旧 Airbus 中间 STEP。

```powershell
D:\Anaconda\envs\cad_graph_env\python.exe python/main.py cad --stage cad
D:\Anaconda\envs\cad_graph_env\python.exe python/main.py cad --stage mesh
D:\Anaconda\envs\cad_graph_env\python.exe python/main.py qem
```

本轮唯一几何/网格证据在 `results/current_three_class_complete_20261020/`。每架的 `cad/<模型标识>/record.json` 关联原模型预测、中间特征/预测、各阶段 STEP 哈希和操作返回值。最终 STEP 为该目录的 `final.step`。操作失败保留上一端点；零候选不执行饰面操作，不能据此声称完全删除。

原模型的逐面真值不直接适用于改变拓扑的中间模型。DC-10 的原几何补全受文件哈希限制，不能按旧面编号套到中间模型。完整运行协议、计时范围和已知限制见 `doc/当前架构说明.md` 及本轮 CAD/网格/QEM 记录。

论文干净版为 `paper/manuscript/main.pdf`，批注版为同目录的 `main_annotated_three_class_20261019.pdf`；正文源为 `paper/manuscript/main.tex`。本轮完整修改记录与审阅报告在 `work/tog_manuscript_reconstruction_20261020/`，旧稿与重复预览已清理。

## 预测值与真值对比可视化

逐机输入和真值位于 `model/final_detection_model/evaluation/inputs`，对应预测位于同一实验的 `evaluation/predictions`。这些输入直接由唯一测试 CSV 分组生成。STEP、真值 CSV 和预测 CSV 的面编号必须完全一致。

示例：

```powershell
D:\Anaconda\envs\cad_graph_env\python.exe .\python\main.py view-predictions `
  ".\data\plane_model\after_two\Airplane body_wing_rivets.step" `
  --pred-in ".\model\final_detection_model\evaluation\predictions\Airplane body_wing_rivets.pred.csv" `
  --truth-csv ".\model\final_detection_model\evaluation\inputs\Airplane body_wing_rivets.features.csv" `
  --truth-model-name "Airplane body_wing_rivets.step" `
  --context-transparency 0
```

颜色约定：

- 绿色：正确 rivet
- 蓝色：正确 surface feature
- 紫色：rivet 误检或类别错误
- 红色：surface feature 误检
- 黄色：surface feature 漏检或类别错误
- 不透明灰色：模型上下文（context-transparency=0）

不要把原始 `after_two` 的预测面编号套到删除后的 STEP。当前端点可视化应直接打开 `results/current_three_class_complete_20261020/cad/<模型标识>/final.step`。

## 删除后线框可视化

删除后的交互式线框窗口使用：

```powershell
D:\Anaconda\envs\cad_graph_env\python.exe .\python\main.py view-edges `
  ".\results\current_three_class_complete_20261020\cad\Airbus\final.step"
```

窗口读取指定 STEP 的真实拓扑边，不读取预测或真值 CSV；对比图的模型背景保持不透明。

## 重要约束

- `data/current_inference.csv` 是共享临时文件，每次切换 STEP 都必须重新导出。
- 权重和 stats 必须成对使用，不能只替换 `.pth`。
- 真值 CSV、预测 CSV 和 STEP 必须面数一致，且 `face_id` 必须对齐。
- 当前 CAD 端点以本轮 `record.json` 为准，不能用旧 `ending` 或旧中间文件替换。
- 旧预测快照、旧候选 STEP 和诊断日志可以在确认不再需要回溯后清理，但不能删除代码、最终 STEP、真值 CSV 或当前后处理结果。


## Python脚本入口与分组

统一通过 `python/main.py <命令>` 使用当前流程；各功能选项用 `<命令> --help` 查看。源代码按 `recognition/`（模型、训练、预测和筛选）、`cad/`（几何处理和候选）、`geometry/`（几何/网格/QEM测量）、`tools/`（标注与选面）、`visualization/`（CAD与预测审查）分组。

完整命令表见 [python/README.md](python/README.md)。2026-10-21整理记录及新旧输出对照见 `work/python_refactor_20261021/整理报告.md`。旧平铺入口已移除，不再维护两套脚本。

## Git 备份与恢复

当前模型权重、归一化与训练资产、九份评估原始 STEP、CAD 中间及最终 STEP、QEM 网格缓存和两份论文 PDF 使用 Git LFS 保存。克隆后先安装 Git LFS，再在项目根目录执行：

```powershell
git lfs install --local
git lfs pull
git lfs fsck
D:\Anaconda\envs\cad_graph_env\python.exe python/main.py predict --verify-only
```

运行仍需按本文前述说明配置 OpenCASCADE、PyTorch 等依赖并构建 Detector。当前实验的正式日志随结果保存；`work/` 中的诊断、临时构建和备份不上传。模型清单中的哈希对应下载后的实际文件，不对应 Git LFS 指针文本。代码、模型配套记录和当前实验结果按原始字节保存，避免 Git 换行转换改变校验哈希。
