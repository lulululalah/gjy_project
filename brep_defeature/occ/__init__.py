"""OpenCASCADE 几何后端（pythonocc-core）。

⚠️ 运行环境：本子包依赖 pythonocc-core，需要在装有 OpenCASCADE 的环境
（推荐 conda: `conda install -c conda-forge pythonocc-core`）下运行。
ML 管线（schema/synth/graph/models/train/infer/instancing/metrics）不依赖本子包，
可在仅有 torch + torch_geometric 的机器上独立开发与测试。

三个模块：
- extract:    TopoDS_Shape / STEP -> FaceGraph（节点/边特征，与 synth 契约一致）。
- inject:     在干净模型上注入铆钉/孔，并依据 BOP 历史给出「注入即真值」标签。
- defeature:  按实例与操作执行删除/补全，并做有效性与恢复误差校验。

导入时若缺少 OCC 依赖会抛 ImportError，属预期行为。
"""
