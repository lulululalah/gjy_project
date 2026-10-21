# 当前论文 CAD／网格出图配置（2026-10-20）

使用 `python/main.py view-cad` 和 OpenCASCADE 真实几何渲染，背景模型透明度为0；原模型蓝色/橙色来自原始注释，用来定位区域，处理后均匀着色不代表新真值。所有配对共享正交相机、中心和尺度，不对处理后模型单独FitAll。

来源清单：`paper/manuscript/current_render_sources.json`（图6、7），`figure_comparison_sources.json`（图8–10），`figure11_mesh_sources.json`（图11）。端点只来自 `results/current_three_class_complete_20261020/cad/`；图11逐项核对该轮 `mesh/` 中的面数、节点数和三角形数。

```powershell
python python/main.py view-cad --export-figure6 paper/manuscript/figures
python python/main.py view-cad --export-figure6-horizontal paper/manuscript/figures
python python/main.py view-cad --export-figure7 paper/manuscript/figures --figure7-model airbus
python python/main.py view-cad --export-figure7 paper/manuscript/figures --figure7-model cessna
python python/main.py view-cad --export-paper-comparisons paper/manuscript/figures
python python/main.py view-cad --export-figure11 paper/manuscript/figures
```

用 `D:\Anaconda\envs\cad_graph_env\python.exe` 执行，各次出图使用独立进程。相机、源 STEP/标签哈希和实际计数保存在 `figures/*manifest.json`。网格采用 `AIS_Triangulation` 模式0，原始三角形直接显示，不降采样、不改坐标。PNG先导出并校验完整性，成功后替换目标，处理Windows偶发导出失败。

在 `paper/manuscript/` 编译 `main.tex`，中间文件放在项目根目录 `work/` 的独立编译目录。交付为同目录的 `main.pdf` 和 `main_annotated_three_class_20261019.pdf`。最新审阅与修改记录在 `work/tog_manuscript_reconstruction_20261020/`；出图只取上述完整运行的端点。
