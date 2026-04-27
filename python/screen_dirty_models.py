import os
import shutil
import pandas as pd
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import topods
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
import math

def analyze_model(step_path):
    """分析单个模型，返回其'脏几何'评分"""
    reader = STEPControl_Reader()
    if reader.ReadFile(step_path) != 1:
        return None

    reader.TransferRoots()
    shape = reader.OneShape()

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    
    total_faces = 0
    sliver_faces = 0
    tiny_faces = 0
    total_area = 0.0
    
    face_data = []

    while explorer.More():
        face = topods.Face(explorer.Current())
        total_faces += 1
        
        # 计算面积和周长
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        area = props.Mass()
        total_area += area
        
        # 计算周长用于紧致度
        line_props = GProp_GProps()
        brepgprop.LinearProperties(face, line_props)
        perimeter = line_props.Mass()
        
        # 计算紧致度 (Perimeter^2 / (4 * PI * Area))
        compactness = 999.0
        if area > 1e-7:
            compactness = (perimeter * perimeter) / (4.0 * math.pi * area)
        
        if compactness > 50: # 极其细长的面
            sliver_faces += 1
        
        face_data.append(area)
        explorer.Next()

    if total_faces == 0: return None

    # 计算微小面 (面积小于平均面积的 1/1000)
    avg_area = total_area / total_faces
    tiny_faces = sum(1 for a in face_data if a < avg_area * 0.001)

    return {
        "path": step_path,
        "total_faces": total_faces,
        "sliver_ratio": sliver_faces / total_faces,
        "tiny_ratio": tiny_faces / total_faces,
        "score": (sliver_faces * 2 + tiny_faces) / total_faces # 综合评分
    }

def screen_dataset(input_dir, output_dir, threshold=0.05):
    """
    input_dir: ABC Dataset 存放路径
    output_dir: 选出的“脏几何”存放路径
    threshold: 脏面占比阈值，超过这个值就认为是典型脏几何
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = []
    print(f"正在扫描目录: {input_dir}")
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.stp', '.step')):
                full_path = os.path.join(root, file)
                print(f"分析中: {file}...", end="\r")
                
                analysis = analyze_model(full_path)
                if analysis and analysis['score'] > threshold:
                    # 发现目标，复制过去
                    shutil.copy(full_path, os.path.join(output_dir, file))
                    results.append(analysis)
                    print(f"\n[发现脏几何] {file} | 脏面占比: {analysis['score']:.2%}")

    # 保存结果报告
    if results:
        df = pd.DataFrame(results)
        df.to_csv("dirty_models_report.csv", index=False)
        print(f"\n扫描完成！共选出 {len(results)} 个典型模型，报告已生成。")
    else:
        print("\n未发现符合条件的脏几何模型。")

if __name__ == "__main__":
    # --- 已经根据你的路径修改 ---
    ABC_DIR = r"D:/Projects/NLib/data/ABCDataset/random_100" 
    DIRTY_OUT = r"data\dirty_training_set" # 筛选出的模型存到这里
    
    screen_dataset(ABC_DIR, DIRTY_OUT)
