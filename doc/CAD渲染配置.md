# 当前论文渲染说明（2026-10-14）

## 第 19–21 页最新分页

取消 Figure 7 后的强制分页，Figure 7–10 允许进入正常浮动页；Figure 10 后改用 `FloatBarrier`，使后续网格比较正文利用页内余量。当前 Table 7 在第 19 页，Figure 7/8/9 连续排在第 20 页，Figure 10 和网格比较正文在第 21 页，Figure 11 在第 22 页，Table 8/9 在第 23 页。图片尺寸、模型顺序、图注和科学内容均不变。下文原第 23 页 Figure 11 的记录为此前分页快照。

## Figure 10 新增 Airplane body 行（2026-10-14）

第 21 页 Figure 10 上行为 Airplane body，下行为 109 standard E-1（按作者后续要求调换）；每行左侧为原始模型、右侧为处理后模型。新增两张 2400×1000 CAD 图沿用其他 CAD 对照图的银灰机身、白底、真实面边界和不透明显示（transparency=0），原始窗面标橙；同组投影方向 `[1,0,0]`、向上方向 `[0,0,1]`、中心与尺度一致。

Airplane body 输入为 `data/plane_model/after_two/Airplane body_wing_rivets.step` 与 `data/plane_model/ending/Airplane body_wing_rivets.step`，与 Figure 11 / Table 8 的正式端点一致。原始标签来自 `data/plane_model/label/Airplane body_wing_rivets.labels.json`；处理后统一灰色仅用于显示，不构成新的语义真值。末端 B-Rep 无效，图注明确仅作描述性展示。未重新执行删除、重建或网格实验。

```powershell
& 'D:\Anaconda\envs\cad_graph_env\python.exe' python/visualize_cad_targets.py --export-paper-comparisons paper/manuscript/figures --comparison-model airplane_body
```

配置位于 `figure_comparison_sources.json`；图片与来源/相机清单为 `cad_airplane_body_{before,after}.png`、`comparison_airplane_body_manifest.json`。Figure 10 排版仍由历史命名的 `fig11_109_decal_comparison.tex` 管理。

当前图号、五组 CAD/网格输入、固定相机、清理目录和重建命令，以 [论文图表调整与清理-2026-10-14.md](论文图表调整与清理-2026-10-14.md) 为准。原 Figure 7/8 已合并为 Figure 7；原 9–12 顺延为 8–11。下面保留历次渲染与阶段记录；未采用的图片已经归档，不再全部位于 figures 目录。

## Figure 11 当前三列版本（2026-10-14）

输入配置为 `paper/manuscript/figure11_mesh_sources.json`。五排依次为 AULIRA 2、DC-10、Airbus、Gulfstream G280 v17、Airplane body；每排三列为原始网格、最终处理后的 CAD、最终处理后的网格。整机与同位置局部图均为 2400×800，正文内每幅宽 48 mm。所有模型均为严格侧视，同排对应图使用相同中心、方向、向上方向和尺度。Airplane body 的机长沿 Y 轴，采用投影 `[1,0,0]`、向上 `[0,0,1]`；局部中心由观察侧窗口标签选取。

原始/最终 STEP 直接读取 `results/e3_mesh/per_model/*_{original,simplified}.json`，与正式 Table 8 完全对应。Airbus 使用最终 surface-feature 阶段的 686 面模型，不再使用旧版 Figure 11 的 1,723 面铆钉阶段模型。三角形数为 24,519→12,589、41,914→35,496、95,610→63,505、17,941→14,378、51,605→42,401。全部采用正式 OCC 网格参数，未降采样、未改几何、未做像素编辑。Airplane body 最终 B-Rep 有效性为 false；正文和图注明确该模型只作描述性表面网格比较。

`FIGURE11_STYLE` 仅由此导出入口传入：机体 RGB `(0.94,0.96,0.98)`、plaster 非金属材质、完全不透明；网格边 RGB `(0.015,0.02,0.025)`、宽度 `0.85`。`AIS_Triangulation` 必须使用显示模式 0，且显式设置填充色，防止 aspect 材质保留默认色。其他图和交互查看器沿用既有样式。

上一版第五排为 109：原始贴花面 79 的 14 个实际节点着橙色，其 12 个三角形及坐标均未改动。该说明仅属于历史版本；作者要求排除 Cessna Citation 并替换 109 后，当前第五排采用 Airplane body，全部网格均不作语义着色。替换前文件及停用的六张 Figure 11 图片、清单归档至 `paper/manuscript/work/figure11_airplane_body_backup_20261014`；Figure 10 的 109 CAD 图保留。

```powershell
& 'D:\Anaconda\envs\cad_graph_env\python.exe' python/visualize_cad_targets.py --export-figure11 paper/manuscript/figures
node work/build_figure11_three_columns_20261014.cjs
```

五份 `fig11_*_manifest.json` 保存正式结果来源、源 STEP/标签 SHA-256、实际网格数、渲染样式和配对相机。构图脚本按相机投影定位选框，不凭目测移动选区。上一版样式图保存在 `paper/manuscript/work/figure11_style_backup_20261014`；旧五对网格图片归档清单见图表修订说明。

G280 新增尾部贴花局部行：在 `extra_details` 中明确选择原始标签面 294、327、361、378，检查四面语义均为 decal 后，以完整组的包围盒生成严格侧视相机。三个阶段共用中心 `(55.41114993175907,19.991664623977208,-0.8924576675076533)`、投影 `[0,0,1]`、向上 `[0,1,0]`、尺度 `10.496441811185877`，导出三张 `fig11_gulfstream_{before_mesh,after_cad,after_mesh}_tail_decal.png`，尺寸均为 2400×800。只使用正式原始/最终 STEP，不沿用最终模型的面编号，不改几何和网格参数。窗口细节位于上行、尾贴花细节位于下行；窗口使用实线选框，尾部使用虚线选框。每幅仍宽 48 mm，后续模型整体下移 18 mm，维持标题分组间距。Figure 11 图注局部采用 10 pt 单倍行距，使 33 幅图与图注完整落在一页。30 项渲染器测试通过；本次构建目录为 `paper/manuscript/work/figure11_g280_tail_build`。

作者追加要求将第二排 747-400 替换为 DC-10。此前包含 747-400 的源稿、PDF、配置和图源保存在 `paper/manuscript/work/figure11_dc10_backup_20261014`；停用的六张 747 图片和相机清单亦归入该目录。正文和图注不再使用“前四例为最大降幅四例”的选择描述；Table 8 的九模型正式数据不变。

标题分组调整：行距从 39 mm 增至 42 mm，模型名称与右侧计数的中心从各排基准上方 3 mm 下移至 1.5 mm；模型名称加粗。标题与上一排局部图之间保留更明显的空白，使其更接近本排整机图。单幅仍宽 48 mm，G280 增加尾部细节后保留这些分组间距。此前标题调整备份仍位于 `work/figure11_row_spacing_backup_20261014`（相对 manuscript）。

# 统一 CAD 渲染入口

保留脚本：`python/visualize_cad_targets.py`。当前输入为 Gulfstream G280 的 STEP 和对应面标签。

后续模型沿用此脚本的渲染设置，不重新创建其他风格的查看器：

- 机体：原版银灰亮面，完全不透明。
- 铆钉：蓝色；窗户和贴花：同一种橙色。
- 面边缘：接近黑色，RGB `(0.003, 0.003, 0.003)`，线宽 `1.3`。
- 背景：白色；使用正交投影。
- 菜单支持整体视角、侧视角、俯视角，以及显示/隐藏机体面边缘。

更换模型时，使用 `--step`、`--labels` 和 `--title` 指定输入与窗口标题；不传参数时仍打开原始 G280。标签必须与 STEP 的全部面编号对应，脚本会检查编号集合。标签格式为 `faces` 数组，每行包含 `face_id`、`semantic`；语义支持 `background`、`rivet`、`window`、`decal`。脚本不读取预测 CSV。

`--view side|top|overall` 指定启动视角，默认仍为 `side`。俯视角适用于 Y 为竖直轴的模型（当前 G280、DC-10）；投影方向为正 Y，画面向上方向为负 Z，不修改模型坐标或几何。

`--center X Y Z --detail-scale S` 用已核对的模型坐标设置局部观察中心及视图尺度（S 必须大于零），用于截取真实近景。它只移动相机、放大观察，不修改铆钉尺寸或补画模型。仍可通过 View 菜单恢复整机视角。

启动已有渲染器：

```powershell
& 'D:\Anaconda\envs\cad_graph_env\python.exe' 'D:\projects\gjy_byproject\python\visualize_cad_targets.py'
```

删除铆钉后的 G280 使用 `work/test_removal_pipeline/Gulfstream_G280_v17/Gulfstream_G280_v17_removed.step`，对应上色标签为同目录的 `after_rivet_render.labels.json`。窗户和贴花根据存档的几何特征匹配到新面编号；不是使用预测类别上色。

允许启动已有渲染器，不意味着授权使用 Python 进行图片编辑或其他处理。

已废弃的旧脚本：`work/render_fig4_rivet_detail.py`、`work/render_figure5_real_bspline.py`。
`work/render_figure5_curved_sheet.py`仍用于当前论文 Figure 5，不属于整机查看器，不在此次清理范围内。

## Figure 6 固定相机导出

当前论文 Figure 6 使用 `fig_g280_horizontal_processing.tex`：三个真实 G280 阶段横向排列，采用“铆钉局部图在上、俯视整机在中、窗户/贴花局部图在下”的三层阅读顺序。前两阶段保留三个不同铆钉位置；三个阶段均展示同一乘客窗和尾翼贴花区域的侧视近景。原三排布局和原始图像保留为可恢复素材。

2026-10-14 选区关联调整：取消作者认为脱离主流程的独立铆钉横带；六张近景归入 A、B 列，各宽 14 mm，连线和编号对应整机上的三个真实位置。整机改用已有的配对俯视截图，各宽 42 mm，避免侧视时两侧机翼投影重叠；窗户、贴花近景各宽 20 mm，连接线利用框间与外侧空隙，第三阶段直接连接选区。整机标记按存档相机和世界坐标投影，不凭目测放置。为使三个铆钉连线不交叉，展示编号 1、2、3 对应原近景文件编号 2、1、3（原始面 573、506、578）；A、B 两列映射完全一致，源相机与图片不变。保持普通衬线字体、细灰框线；只更新排版和图注，不修改截图像素、几何、渲染脚本或 Figure 7/8。独立横带前版保留在 `paper/manuscript/work/figure6_callout_integration_backup_20261014`。

2026-10-14 上下分层调整：作者认为铆钉与 surface-feature 近景同处上方会使阅读混乱。将 A/B 的铆钉近景移动至整机行上方，将全部窗户与贴花近景移动至整机行下方；中间整机行保留阶段箭头和操作文字。后续按作者反馈删除所有选区至近景的长引线：铆钉改由整机选框与同编号近景对应，窗户和贴花由同列标题、位置与颜色对应。截图、固定相机、编号映射、面选区和科学表述均不变。

横版素材使用 `--export-figure6-horizontal DIRECTORY` 导出，文件前缀为 `fig6_g280_horizontal`，不覆盖原固定相机素材。整机仍为 2400×1100，同组局部图均为 900×900；铆钉使用原三处位置但相机尺度为原版的 25%，窗户和贴花使用严格侧视。局部相机跨阶段配对；窗户选取原始面 142，贴花选取靠近观察侧的原始面 294、327、361、378。这些编号仅用于记录原模型选区，不沿用到后续模型。源几何、标签哈希和相机写入 `fig6_g280_horizontal_fixed_camera_manifest.json`。只读取存档几何，不重新执行删除或推理。末阶段的灰色显示不是新的真值；所选局部变化不证明全模型完全删除。新增选择与 CLI 测试已写入现有测试文件；遵守仅允许运行现有 CAD 渲染器的约束，不使用 Python 执行测试或图片编辑。

```powershell
& 'D:\Anaconda\envs\cad_graph_env\python.exe' python/visualize_cad_targets.py --export-figure6-horizontal paper/manuscript/figures
```

现有查看器增加 `--export-figure6 DIRECTORY`，直接读取原始 G280、存档的 `_removed.step` 与 `_embedded.step` 以及各自配对的标签。使用同一套不透明机体、上色、白色背景和面边缘设置；不执行删除或重建操作，不裁切、拉伸或补画图片。

```powershell
& 'D:\Anaconda\envs\cad_graph_env\python.exe' python/visualize_cad_targets.py --export-figure6 paper/manuscript/figures
```

三个阶段分别导出严格侧视和俯视。每列跨阶段共用观察中心、投影方向、向上方向和尺度；两列共用水平世界跨度。侧视为 2400×1100，俯视为 2400×2100。前两阶段分别在三个不同铆钉位置导出 900×900 近景，共六张；每个位置的前后相机严格配对，三个位置编号从俯视图上方到下方排列。选择时按投影位置合并近重合的标注面，避免选择重复的观察位置。窗口尺寸差异通过重新渲染修正，而非像素编辑。原始铆钉面及所有相机参数、源 STEP/标签哈希记录在 `fig6_g280_fixed_camera_manifest.json`。第三阶段标签只用于统一灰色显示，不是新的语义真值。

## Figure 7 Airbus 配对截图

`--export-figure7 DIRECTORY` 读取 `data/plane_model/after_two/Airbus_wing_rivets.step` 和 `work/test_removal_pipeline/Airbus/Airbus_removed.step`，配对原始标签和存档的 `after_rivet_render.labels.json`，不重新运行铆钉删除或推理。

```powershell
& 'D:\Anaconda\envs\cad_graph_env\python.exe' python/visualize_cad_targets.py --export-figure7 paper/manuscript/figures
```

两个阶段共用严格俯视正交相机、观察中心、尺度和 2400×2100 导出尺寸；正文中两张整机图均宽 44.16 mm。三个不同铆钉位置各导出删除前、删除后 900×900 近景，共六张，框内保持正方形原图比例。Airbus 的候选范围限制为原始铆钉面中心 Y ≥ −0.5，并人工检查实际俯视渲染，排除被机体遮挡的位置；此限制不是通用可见性算法。当前位置对应原始面 1584、1588、393，处理后用同一世界坐标和相机展示，不沿用发生变化的面编号。源文件哈希、位置和相机记录在 `fig7_airbus_fixed_camera_manifest.json`。颜色只用于存档语义的可视化，不代表新的预测或所有铆钉已删除。

## Figure 8 Xian Y-20 配对截图

`--export-figure8 DIRECTORY` 读取 `data/plane_model/new_data/4-20 xi-an y20 STP STL_Xian Y-20 v3_wing_rivets.step`、对应原始标签，以及 `work/xian_y20_rivet_removal/Xian_Y20_after_rivets.step` 和 `after_rivet_render.labels.json`。只导出存档几何，不重新执行标注驱动的处理或模型推理。

```powershell
& 'D:\Anaconda\envs\cad_graph_env\python.exe' python/visualize_cad_targets.py --export-figure8 paper/manuscript/figures
```

沿用 Figure 7 的 2400×2100 严格俯视整机截图、44.16 mm 等宽显示及六张 900×900 正方形近景。依据实际渲染人工确认三个不同、可见且局部铆钉已移除的位置，编号 1–3 对应原始面 83、59、48；近景共用 0.1 的相机尺度，处理前后固定同一世界坐标。候选包围盒最大跨度小于 0.05 的限制仅用于筛选紧凑近景，不是新的识别或删除规则。未经采用的候选中存在遮挡或残余铆钉，整机输出仍完整保留这些几何；三组局部展示不证明全模型完全删除。相机、位置和源文件哈希记录在 `fig8_xian_y20_fixed_camera_manifest.json`。Figure 7、8 顶部只保留模型名称，不再使用 `(a)`、`(b)`，正文和图注同步去掉相应后缀。

## Figure 7、8 同页排版

两图的坐标布局和图片按 48% 等比例缩小：整机宽 44.16 mm、细节框宽 12.48 mm，标签保留 7 pt（编号 6 pt），不使用整体位图缩放来压缩字体。两图在同一排版块中上下放置、间隔 5 mm，各保留独立图号和原图注内容；图注字号只在这个块内设为 small。取消两图之间的强制分页，允许该块就近放在有空间的正文页，而不是仅限独立浮动页。当前两图位于第 20 页，与该页原有的两行正文共存，论文共 28 页。相机、源图片及渲染器均不改变。
