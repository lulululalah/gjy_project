# 当前识别实验入口

唯一正式数据、模型和预测位于 `model/final_detection_model/`。

使用一个共享编码器、宽度 64、一个三分类 MLP 头，softmax 后按最大概率输出背景、铆钉、饰面，再执行几何后处理。
当前选定 `single_three_class_distilled_finetune_h64_seed123`，后处理后铆钉 F1 为 0.9297、饰面 F1 为 0.7776。
选择和修改记录见正式目录的 `selection.json`、`configuration.json`、`CHANGELOG_20261019.md`。
本版平均 F1 为 0.8537，旧双头为 0.8348。数据、真值和后处理相同；训练阶段使用旧双头作为教师，推理仅使用单个三分类头。该固定测试集已用于开发，不作为独立泛化验证。

在项目根目录运行：

```powershell
D:\Anaconda\envs\cad_graph_env\python.exe python/run_recognition_experiment.py --verify-only
```

历史候选权重、旧消融实验、重复预测及专用实验脚本已按用户要求清理，记录见 `work/current_three_class_cleanup_20261019/`。
当前三分类推理和重训不依赖历史模型权重。训练日志与来源记录保留在正式目录 `training_record/`，所需初始化和辅助分数保留在 `training_assets/`。
`work/recognition_archive_20261019/` 仅保留原始标签审计、论文证据摘要及历史移动清单；移动清单中的旧模型文件已经清理，不能按该清单恢复。
论文与 CAD/网格产物本轮未改写。
