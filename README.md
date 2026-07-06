# gjy_byproject

当前项目主线已经统一为：
`STEP -> 注入 labels.json -> Face Graph -> GNN -> 推理/可视化`

## 当前推荐流程

在项目根目录执行：

```powershell
cd D:\projects\gjy_byproject
```

1. 导出训练数据

```powershell
.\build\Debug\Detector.exe --train
```

输出：
- `data/wing_rivet_training_set.csv`

2. 按显式训练/验证集训练 GNN

```powershell
D:\Anaconda\envs\cad_graph_env\python.exe .\python\train_rivet_gcn.py --csv data\plane_model\plane_model_face_train.csv --val-csv data\plane_model\plane_model_face_val.csv --training-mode window --window-hop 2 --epochs 150 --model-out rivet_gnn_no_centerz_split20_5.pth --stats-out rivet_gnn_no_centerz_split20_5_stats.npz --eval-out rivet_gnn_no_centerz_split20_5_eval.csv
```

当前仓库只保留这一对 20/5 分割文件：
- `data/plane_model/plane_model_face_train.csv`
- `data/plane_model/plane_model_face_val.csv`

3. 导出单模型推理数据

```powershell
.\build\Debug\Detector.exe --predict .\data\plane_model\step\109-ww1-standart-e-1-aircraft CATIA STP_standart-e1_wing_rivets.step
```

输出：
- `data/current_inference.csv`

4. 可视化推理结果

```powershell
D:\Anaconda\envs\cad_graph_env\python.exe .\python\visualize_rivets.py .\data\plane_model\step\109-ww1-standart-e-1-aircraft CATIA STP_standart-e1_wing_rivets.step --detector .\build\Debug\Detector.exe --compare-labels .\data\plane_model\label\109-ww1-standart-e-1-aircraft CATIA STP_standart-e1_wing_rivets.labels.json --model .\rivet_gnn_no_centerz_split20_5.pth --stats .\rivet_gnn_no_centerz_split20_5_stats.npz --inference-mode window --window-hop 2
```

## 关键文件

- `src/Workflow.cpp`：训练/推理 CSV 导出
- `src/FeatureInjector.cpp`：翼面铆钉注入与真值生成
- `python/train_rivet_gcn.py`：GNN 训练
- `python/visualize_rivets.py`：推理与可视化
- `data/wing_rivet_training_set.csv`：总训练导出
- `data/current_inference.csv`：单模型推理导出

## 相关文档

- [命令行使用说明](D:/projects/gjy_byproject/doc/命令行使用说明.md)
- [当前架构说明](D:/projects/gjy_byproject/doc/当前架构说明.md)
- [算法设计-特征识别与去除](D:/projects/gjy_byproject/doc/算法设计-特征识别与去除.md)
