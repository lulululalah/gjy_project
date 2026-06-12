# gjy_byproject

当前项目采用纯图方法。

主链路是：

`STEP -> FaceFeature -> Face Graph -> GNN -> Face Label / Visualization`

## 当前推荐流程

在项目根目录执行：

```powershell
cd D:\Projects\gjy_byproject
```

1. 导出训练数据

```powershell
.\build\Debug\Detector.exe --train
```

2. 训练 GNN

```powershell
python .\python\train_rivet_gcn.py
```

3. 导出单模型推理数据

```powershell
.\build\Debug\Detector.exe --predict .\data\dirty_training_set\44.step
```

4. 可视化推理结果

```powershell
python .\python\visualize_rivets.py .\data\dirty_training_set\44.step --detector .\build\Debug\Detector.exe
```

## 关键文件

- `src/Workflow.cpp`：训练/推理 CSV 导出工作流
- `python/train_rivet_gcn.py`：图神经网络训练
- `python/visualize_rivets.py`：推理与可视化
- `data/full_training_set.csv`：训练数据
- `data/current_inference.csv`：单模型推理数据

## 相关文档

- [命令行使用说明](D:/Projects/gjy_byproject/doc/命令行使用说明.md)
- [当前架构说明](D:/Projects/gjy_byproject/doc/当前架构说明.md)
- [思路设计](D:/Projects/gjy_byproject/doc/思路设计.txt)
