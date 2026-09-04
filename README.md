# gjy_byproject

项目当前只保留一条识别主线：

`STEP -> 面特征/邻接图 -> 独立 rivet 与 surface specialist -> 壳体保护 -> 预测/真值可视化`

类别约定：

- `0 = background`
- `1 = rivet`
- `2 = surface_feature`（原生贴花、原生窗户和后加贴花）

当前目标是优先控制误检：铆钉允许少量漏检但不应误检；surface 允许少量漏检和误检，但大面积机身或发动机壳体不能成为删除候选。

## 当前数据与模型

- 训练集：`work/uv_train17_xian20_simpletest_train.csv`
- 测试集：`work/uv_test4_xian20_simpletest.csv`
- 模型：`work/rivet_gnn_xian20_train_simpletest_50ep.pth`
- 归一化与推理契约：`work/rivet_gnn_xian20_train_simpletest_50ep_stats.npz`
- 测试结果：`work/rivet_gnn_xian20_train_simpletest_50ep_eval.csv`

训练集和测试集按完整飞机划分，不允许同一飞机同时出现在两边。当前测试机型为 109、DC-10、747-400 和 Gulfstream G280。

## 训练

项目不再使用交叉验证、OOF 或 hard-negative 回灌。训练固定使用一个训练集，完成全部 epoch 后只在测试集评估一次。

```powershell
D:\Anaconda\envs\cad_graph_env\python.exe .\python\train_rivet_gcn.py `
  --csv .\work\uv_train17_xian20_simpletest_train.csv `
  --test-csv .\work\uv_test4_xian20_simpletest.csv `
  --epochs 50 --batch-size 1 --hidden-dim 64 --num-layers 4 `
  --lr 0.005 --dropout 0.2 --weight-decay 0.0001 --seed 123 `
  --model-out .\work\rivet_gnn_xian20_train_simpletest_50ep.pth `
  --stats-out .\work\rivet_gnn_xian20_train_simpletest_50ep_stats.npz `
  --eval-out .\work\rivet_gnn_xian20_train_simpletest_50ep_eval.csv
```

模型使用两个完全独立的 GNN 编码器：一个识别 rivet，一个识别 surface。两个输出分别使用阈值，合并时 rivet 优先。

## 壳体误检保护

推理会在 surface 输出之后检查光滑同曲面连通组件。以下两类面会从 surface 回退为 background：

- 在大型光滑组件中占主导面积的面；
- 在至少 100 个面的密集壳体组件中，占组件面积至少 3% 的异常大面。

该规则结合连通拓扑、组件规模和组件内面积比例，不是全局面积过滤。当前训练集和测试集内满足保护条件的已标注面全部是 background。

## 推理与可视化

Detector 将单个 STEP 导出到共享临时文件 `data/current_inference.csv`。每次切换飞机都必须重新导出，不能复用上一架飞机的该文件。

```powershell
.\build\Release\Detector.exe --predict ".\data\plane_model\after_two\87- 747-400 stp_beoing 747-400 v6_wing_rivets.stp"

D:\Anaconda\envs\cad_graph_env\python.exe .\python\visualize_rivets.py `
  ".\data\plane_model\after_two\87- 747-400 stp_beoing 747-400 v6_wing_rivets.stp" `
  --skip-export `
  --model .\work\rivet_gnn_xian20_train_simpletest_50ep.pth `
  --stats .\work\rivet_gnn_xian20_train_simpletest_50ep_stats.npz `
  --truth-csv .\work\uv_test4_xian20_simpletest.csv `
  --truth-model-name "87- 747-400 stp_beoing 747-400 v6_wing_rivets.stp"
```

对比颜色：绿色为正确铆钉，蓝色为正确 surface，紫色为铆钉错误，红色为 surface 误检，黄色为 surface 漏检，透明灰色为背景。

## 保留的核心代码

- `src/FeatureExtractor.cpp`：几何与 UV 特征提取
- `src/Workflow.cpp`：训练和推理 CSV 导出
- `python/train_rivet_gcn.py`：双 specialist 训练和测试
- `python/visualize_rivets.py`：推理、壳体保护和 OCC 可视化
- `python/swap_models_between_splits.py`：整机级训练/测试互换
- `python/initialize_step_labels.py`、`python/label_native_faces.py`：标签初始化和原生面标注
- `python/pick_step_face.py`、`python/select_boolean_host_face.py`：人工几何检查

模型预测后删除 CAD 面的功能尚未实现；当前阶段只负责稳定识别和输出可靠的删除候选。
