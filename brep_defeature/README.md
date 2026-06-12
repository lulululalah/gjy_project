# brep_defeature

BRep 微小特征（铆钉 / 孔 / 窗口 / 贴花 / 倒角）的**识别**与**去除/补全**，
用于在仿真前简化模型、减少网格聚集、加速求解。

完整算法设计见 [`doc/算法设计-特征识别与去除.md`](../doc/算法设计-特征识别与去除.md)。
本目录是该设计的从零实现（重构分支 `feature/defeature-rebuild`）。

## 三项核心设计

1. **训练监督 = 合成注入的真值**，而非手工启发式规则。在干净模型上程序化注入特征，
   注入瞬间即知每个面的真值标签（语义 / 实例 / 操作）。
2. **局部 window** = k-hop 子图采样（训练）+ 浅层感受野（推理）。微小特征是局部的。
3. **先铆钉+小孔跑通端到端**，再扩展更多特征类型与几何去除。

## 架构（两层解耦）

```
            ┌─────────── schema.py（数据契约：FaceGraph + 三层标签）───────────┐
            │                                                                  │
  ML 管线（纯 torch，跨平台）                         几何后端（OCC，需 conda）
  ─────────────────────────────                      ────────────────────────────
  synth/      合成面图 + 真值（无需 OCC）             occ/extract    STEP -> FaceGraph
  graph/      构图 + k-hop 窗口                       occ/inject     注入特征 + 真值
  models/     多任务 GNN（语义/实例/操作）            occ/defeature  删除 + 补平 + 校验
  instancing  逐边亲和度 -> 连通分量 -> 实例
  metrics     mIoU / 操作准确率 / 实例检测 F1
  train,infer 训练与推理
```

`schema.FaceGraph` 是唯一接口：无论图来自 `synth`（无 OCC）还是 `occ.extract`
（真实 STEP），下游训练/推理代码完全一致。因此 **ML 管线可在没有 OpenCASCADE
的机器上独立开发与测试**，几何后端在 Windows/conda 环境接入。

## 安装

ML 管线（任意平台）：

```bash
pip install -r brep_defeature/requirements.txt
```

几何后端（需要 OpenCASCADE）：

```bash
conda install -c conda-forge pythonocc-core
```

## 快速开始

在合成数据上端到端训练并评估（无需 OCC）：

```bash
python -m brep_defeature.train --models 60 --epochs 60
```

跑测试：

```bash
python -m pytest tests/ -q
```

真实数据流程（需 OCC 环境）：

```python
import random
from brep_defeature.occ.extract import read_step
from brep_defeature.occ.inject import make_training_sample

base = read_step("data/clean_part.step")
graph, shaped, records = make_training_sample(base, "part_aug0", random.Random(0),
                                              n_rivets=5, n_holes=4)
graph.to_json("part_aug0.json")   # -> 喂给 train
```

推理 + 去除：

```python
from brep_defeature.occ.extract import step_to_graph
from brep_defeature.occ.defeature import remove_instances
from brep_defeature.infer import load_model, infer
from brep_defeature.schema import NormStats

g = step_to_graph("data/dirty_part.step")
model = load_model("defeature_gnn.pth")
res = infer(model, g, NormStats.load("defeature_stats.npz"))
shape = read_step("data/dirty_part.step")
clean, report = remove_instances(shape, res.instance_face_ids())
print(report)
```

## 现状与边界

- ✅ ML 管线全部可运行、有测试（schema / synth / window / GNN / instancing / metrics / 训练）。
- ✅ 合成数据上端到端通过；当前合成图**刻意可分**，目的是验证管线正确性，
  指标接近满分并不代表真实任务难度——真实 STEP 才是真正考验。
- ⚠️ `occ/` 子包已按 pythonocc-core API 实现，但**未在 OCC 环境实跑验证**；
  其中边的凸/凹符号约定、BOP 历史面映射、`BRepAlgoAPI_Defeaturing` 对样条面的效果
  需在目标 OCC 版本上标定（见设计文档 §9 待讨论项）。

## 路线图（对应设计文档里程碑）

- M0 面标识稳定性（注入直接对内存 shape 构图，规避 STEP 往返）
- M1 铆钉+孔注入器 + 真值导出 ← `occ/inject.py`
- M2 语义分割单任务 ← 已含在多任务中
- M3 局部 window ← `graph/window.py`
- M4 实例头 + 操作头 ← `models/gnn.py` + `instancing.py`
- M5 几何去除 ← `occ/defeature.py`
- M6 扩展特征类型
- M7 端到端业务验证（网格/仿真加速）
```
