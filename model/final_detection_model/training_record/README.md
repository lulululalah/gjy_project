# 当前模型训练记录

training_history.csv 和 training_complete.json 对应正式 model.pth 的原始 200 轮训练。original_training_stats.npz 是训练时的原始归一化文件；正式 stats.npz 另含部署元数据，数值归一化应相同。

original_protocol.json、original_comparison_metrics.csv 记录当时的候选比较与历史输入哈希，其中旧模型路径仅用于来源说明，旧候选已清理，不是当前运行依赖。训练辅助分数只来自 17 架训练飞机。推理不使用教师。

当前重训入口为项目 python/run_three_class_refinement.py 和本包 retrain.ps1；读取 training_assets/ 中的初始编码器、三路辅助分数和归一化。cached_retraining_complete.json 记录移除旧双头权重后完成的独立 200 轮重训验证；该验证权重未替换正式模型，已作为临时实验清理。
