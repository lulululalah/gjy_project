from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import topods
from OCC.Display.SimpleGui import init_display
from OCC.Core.Quantity import Quantity_NOC_RED, Quantity_NOC_GRAY, Quantity_NOC_WHITE
import sys

def visualize_cad_results(step_path, rivet_ids):
    # 1. 加载模型
    reader = STEPControl_Reader()
    if reader.ReadFile(step_path) != 1:
        print(f"错误: 无法读取文件 {step_path}")
        return

    reader.TransferRoots()
    shape = reader.OneShape()

    # 2. 初始化 3D 窗口
    display, start_display, add_menu, add_function_to_menu = init_display()
    print(f"正在显示模型: {step_path}")
    print(f"被标记为红色的面 ID: {rivet_ids}")

    # 3. 遍历面并按 ID 染色渲染
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_idx = 1
    
    while explorer.More():
        face = topods.Face(explorer.Current())
        
        # 如果该面在 AI 识别的列表里，染成红色
        if face_idx in rivet_ids:
            display.DisplayShape(face, color=Quantity_NOC_RED, update=False)
        else:
            # 其他面设为灰色半透明
            display.DisplayShape(face, color=Quantity_NOC_GRAY, transparency=0.7, update=False)
        
        face_idx += 1
        explorer.Next()

    display.FitAll()
    # display.View_Isometric() # 移除此行以兼容不同版本的 pythonocc
    start_display()

if __name__ == "__main__":
    # 这里填入您训练脚本输出的 ID 列表
    # 比如之前的结果是 [7, 9]
    target_ids = [7, 9]
    
    model_path = "data/mixed_test.stp"
    
    visualize_cad_results(model_path, target_ids)
