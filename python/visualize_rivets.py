from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import topods
from OCC.Display.SimpleGui import init_display
from OCC.Core.Quantity import Quantity_NOC_RED, Quantity_NOC_GRAY, Quantity_NOC_GREEN

def visualize_cad_results(step_path, pred_labels):
    # 1. 加载模型 (简化版读取，确保与 C++ 对齐)
    reader = STEPControl_Reader()
    if reader.ReadFile(step_path) != 1:
        print(f"错误: 无法读取文件 {step_path}")
        return

    reader.TransferRoots()
    shape = reader.OneShape() # 恢复为 OneShape，但在 C++ 端也保持一致
    
    # 2. 初始化 3D 窗口
    display, start_display, add_menu, add_function_to_menu = init_display()
    print(f"正在显示模型: {step_path}")

    # 3. 遍历面并按预测标签染色渲染 (使用 IndexedMap 确保顺序与 C++ 一致)
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape
    from OCC.Core.TopExp import topexp
    
    face_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, face_map)
    
    num_faces = face_map.Size()
    print(f"模型总面数: {num_faces}, 推理标签数: {len(pred_labels)}")

    for i in range(1, num_faces + 1):
        face = topods.Face(face_map.FindKey(i))
        if (i-1) < len(pred_labels):
            label = pred_labels[i-1]
            if label == 1: # 零件
                display.DisplayShape(face, color=Quantity_NOC_GREEN, update=False)
            elif label == 2: # 垃圾
                display.DisplayShape(face, color=Quantity_NOC_RED, update=False)
            else: # 背景
                display.DisplayShape(face, color=Quantity_NOC_GRAY, transparency=0.7, update=False)

    display.FitAll()
    # display.View_Isometric() # 移除此行以兼容不同版本的 pythonocc
    start_display()

import torch
from train_rivet_gcn import load_cad_data, RivetGNN

def run_inference(csv_path, model_path):
    # 1. 加载数据
    data = load_cad_data(csv_path)
    
    # 2. 初始化模型 (需与训练脚本参数 64 一致)
    in_channels = data.num_node_features
    model = RivetGNN(node_features=in_channels, edge_features=1, hidden_dim=64)
    
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
    csv_path = "data/current_inference.csv"
    model_weight = "rivet_gnn.pth"
    # 这里我们还是先写死 model_69.stp，之后你可以从命令行传入
    step_model = R"D:\Projects\NLib\data\ABCDataset\random_100\model_69.stp"
    
    print(f"正在分析模型: {step_model}")
    predicted_labels = run_inference(csv_path, model_weight)
    
    print(f"AI 识别完成。")
    visualize_cad_results(step_model, predicted_labels)
