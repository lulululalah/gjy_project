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

- 训练集：`work/uv_train17_xian20_simpletest_train.csv`
- 测试集：`work/uv_test5_xian20_simpletest_cessna.csv`
- 权重：`work/rivet_gnn_xian20_train_simpletest_50ep.pth`
- stats：`work/rivet_gnn_xian20_train_simpletest_50ep_stats.npz`
- 评估结果：`work/rivet_gnn_xian20_train_simpletest_50ep_eval.csv`

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

训练按完整飞机划分，避免同一飞机的相邻面同时出现在训练集和测试集。正式模型采用固定训练集、固定测试集和三分类 dual-head GNN：

```powershell
D:\Anaconda\envs\cad_graph_env\python.exe .\python\train_rivet_gcn.py `
  --csv .\work\uv_train17_xian20_simpletest_train.csv `
  --test-csv .\work\uv_test5_xian20_simpletest_cessna.csv `
  --epochs 50 --batch-size 1 --hidden-dim 64 --num-layers 4 `
  --lr 0.005 --dropout 0.2 --weight-decay 0.0001 --seed 123 `
  --enable-smooth-shell-surface-guard `
  --model-out .\work\rivet_gnn_xian20_train_simpletest_50ep.pth `
  --stats-out .\work\rivet_gnn_xian20_train_simpletest_50ep_stats.npz `
  --eval-out .\work\rivet_gnn_xian20_train_simpletest_50ep_eval.csv
```

两个 specialist 使用同一套几何/拓扑输入，但编码器、分类头和阈值独立；融合时保留 rivet 优先级，输出最终三分类标签。

## 批量删除流程

批量流程入口为：

```powershell
D:\Anaconda\envs\cad_graph_env\python.exe .\python\run_test_removal_pipeline.py
```

默认读取 `data/plane_model/after_two`，以测试清单中的完整飞机为单位处理，并将最终结果写入 `data/plane_model/ending`。中间预测、日志和候选 STEP 写入 `work/test_removal_pipeline`。

每个模型按以下顺序处理：

1. 对原始验证 STEP 提取特征并预测铆钉。
2. 删除预测铆钉，生成 `*_removed.step`。
3. 对铆钉删除后的 STEP 重新提取特征并预测 `surface_feature`。
4. 执行 surface 后处理，抑制大型光滑壳体误报。
5. 根据模型拓扑选择对应的表面特征删除和宿主面重建策略。
6. 将通过检查的结果复制到 `data/plane_model/ending`。

模型专用策略如下：

| 模型 | 删除/修复策略 |
|---|---|
| 109、747-400、Cessna Citation 2 | 通用 surface 特征删除 |
| DC-10 | split window skin 重建，再执行 bridge 修复 |
| AULIRA 2、Airplane body、Gulfstream G280 v17 | embedded window host 重建 |
| Airbus | 使用验证过的铆钉删除拓扑；补全漏检窗面后执行 embedded window host 重建 |
| Air Plane Idea A | 使用 invalid surface host rebuild；允许保留原始非法 BRep 诊断结果 |

Airbus 是特殊情况：重新导出铆钉删除 STEP 会改变拓扑和面映射，因此流程固定复用已经验证过的 `data/plane_model/removed/Airbus_removed.step`，再使用 `surface_visibility_guard.py` 补全 5 组漏检窗面，最后执行窄窗删除策略。

批量流程会检查关键结果日志。Airbus 必须满足 108 个窗环、删除 1037 个残余窗面、1723 面变为 686 面且 BRep 合法，才允许写入 `ending`。

## 预测值与真值对比可视化

真值 CSV 位于 `work/test_eval_models`，当前后处理预测位于 `work/postprocess_current_20260909`。STEP、真值 CSV 和预测 CSV 的面数必须完全一致。

示例：

```powershell
D:\Anaconda\envs\cad_graph_env\python.exe .\python\visualize_rivets.py `
  ".\data\plane_model\after_two\Airplane body_wing_rivets.step" `
  --pred-in ".\work\postprocess_current_20260909\Airplane body_wing_rivets.pred.csv" `
  --truth-csv ".\work\Airplane body_wing_rivets.truth.csv" `
  --truth-model-name "Airplane body_wing_rivets.step" `
  --context-transparency 0.82
```

颜色约定：

- 绿色：正确 rivet
- 蓝色：正确 surface feature
- 紫色：rivet 误检或类别错误
- 红色：surface feature 误检
- 黄色：surface feature 漏检或类别错误
- 透明灰色：模型上下文

不要把 `after_two` 的预测对比模型和 `ending` 的删除后模型混用。删除后可视化应直接打开 `data/plane_model/ending` 中对应的 STEP。

## 删除后线框可视化

删除后的交互式线框窗口使用：

```powershell
D:\Anaconda\envs\cad_graph_env\python.exe .\work\open_ending_step_viewer.py `
  ".\data\plane_model\ending\Airbus_wing_rivets.step"
```

窗口显示透明灰色面和黑色拓扑边，不读取预测或真值 CSV。

## 重要约束

- `data/current_inference.csv` 是共享临时文件，每次切换 STEP 都必须重新导出。
- 权重和 stats 必须成对使用，不能只替换 `.pth`。
- 真值 CSV、预测 CSV 和 STEP 必须面数一致，且 `face_id` 必须对齐。
- 已通过的 `ending` 模型和当前代码属于最终工作结果，不要用旧中间文件覆盖它们。
- 旧预测快照、旧候选 STEP 和诊断日志可以在确认不再需要回溯后清理，但不能删除代码、最终 STEP、真值 CSV 或当前后处理结果。
