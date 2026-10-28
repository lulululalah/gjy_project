# 当前 Python 脚本

项目根目录下统一运行：

```powershell
D:\Anaconda\envs\cad_graph_env\python.exe python/main.py --help
D:\Anaconda\envs\cad_graph_env\python.exe python/main.py predict --verify-only
```

每个命令的选项用 `python/main.py <命令> --help` 查看。模块文件用于导入，日常操作统一走 `main.py`。

## 目录职责

| 目录/文件 | 职责 |
|---|---|
| `main.py` | 唯一命令入口；按需加载相应功能 |
| `recognition/graph.py` | 面/边属性、归一化、图构建与编码器 |
| `recognition/model.py` | 三分类网络、监督损失与模型评估 |
| `recognition/train.py` | 当前训练缓存、蒸馏目标和重训流程 |
| `recognition/predict.py` | 正式包预测、指标和数据/结果核验 |
| `recognition/postprocess.py` | 尺寸、拓扑、可见性和候选密度筛选 |
| `recognition/recovery.py` | 正式配置中的几何窗口补全 |
| `cad/pipeline.py` | 当前 CAD 顺序处理与成对网格实验 |
| `cad/candidates.py` | 操作配置、几何对应和饰面候选 |
| `cad/windows.py` | 有几何支持的窗片与接缝处理 |
| `geometry/measure.py` | 共用 STEP 读取、几何差异和表面网格测量 |
| `tools/labels.py` | 标签初始化、人工确认的原生特征分组 |
| `tools/faces.py` | 交互选面与布尔宿主面选择 |
| `visualization/cad.py` | CAD/网格论文图及最终边界线框查看 |
| `visualization/predictions.py` | 三分类预测与真值对比查看 |
| `tests/` | 模型、训练、候选、几何路由和出图的回归测试 |

## 常用命令

```powershell
# 核验当前模型包与保存结果
python python/main.py predict --verify-only

# 在新目录重新推理；不覆盖正式模型或保存结果
python python/main.py predict --output-dir work/new_recognition

# 完整重训，输出到新的目录
python python/main.py train --output-dir work/new_training

# 当前九机 CAD 处理及成对表面网格测量
python python/main.py cad --stage cad
python python/main.py cad --stage mesh

# 论文图出图示例
python python/main.py view-cad --export-figure11 paper/manuscript/figures
```

以上 `python` 应指向已安装 PyTorch、PyG 和 pythonOCC 的 `cad_graph_env`；也可直接使用开头的完整解释器路径。`train --smoke`只运行两轮训练检查，不进行测试集评估。

其他入口：`mesh`（单个 STEP 网格测量）、`compare-geometry`、`complete-windows`、`labels-init`、`labels-apply`、`pick-face`、`select-host`、`view-predictions`和`view-edges`。

## 保留约定

当前唯一部署模型仍为 `model/final_detection_model/`。输入特征、网络维度、类别、损失、冻结训练流程和候选处理顺序沿用当前正式配置。旧双头网络、窗口采样训练、旧架构比较入口和未采用的便携恢复分类器已移除。

新运行目录必须与既有证据分开；当前完整 CAD/网格结果仍在 `results/current_three_class_complete_20261020/`。原始训练记录保持原貌，新模块路径及哈希见正式包的 `manifest.json`。本轮整理记录在 `work/python_refactor_20261021/整理报告.md`。
