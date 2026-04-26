from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import topods
from OCC.Display.SimpleGui import init_display
from OCC.Core.Quantity import Quantity_NOC_RED, Quantity_NOC_GRAY, Quantity_NOC_GREEN

def visualize_cad_results(step_path, pred_labels):
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

    # 3. 遍历面并按预测标签染色渲染
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_idx = 0 # Data 是 0-indexed 的，我们这里直接按顺序匹配
    
    while explorer.More():
        face = topods.Face(explorer.Current())
        label = pred_labels[face_idx]
        
        if label == 1: # 零件
            display.DisplayShape(face, color=Quantity_NOC_GREEN, update=False)
        elif label == 2: # 垃圾
            display.DisplayShape(face, color=Quantity_NOC_RED, update=False)
        else: # 背景
            display.DisplayShape(face, color=Quantity_NOC_GRAY, transparency=0.7, update=False)
        
        face_idx += 1
        explorer.Next()

    display.FitAll()
    # display.View_Isometric() # 移除此行以兼容不同版本的 pythonocc
    start_display()

import torch
from train_rivet_gcn import load_cad_data, RivetGNN

def run_inference(csv_path, model_path):
    # 1. 加载数据
    data = load_cad_data(csv_path)
    
    # 2. 初始化模型
    in_channels = data.num_node_features
    model = RivetGNN(node_features=in_channels, edge_features=1, hidden_dim=32)
    
    # 3. 加载权重
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 4. 推理
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
    
    # 返回所有预测标签列表
    return pred.tolist()

if __name__ == "__main__":
    csv_path = "data/train_data.csv"
    model_weight = "rivet_gnn.pth"
    step_model = "data/rivet_plate.stp"
    
    print("正在运行 AI 推理识别 (三分类: 基体/零件/垃圾)...")
    predicted_labels = run_inference(csv_path, model_weight)
    
    print(f"AI 识别完成。")
    visualize_cad_results(step_model, predicted_labels)
