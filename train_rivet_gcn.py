import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import NNConv
import torch.nn.functional as F

# 1. 加载并处理 CSV 数据
def load_cad_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # --- 节点特征 (Node Features) ---
    # 我们选择：相对面积, 紧致度, 表面类型, 法向X, Y, Z
    # 注意：表面类型是类别，理想情况应做 One-hot，这里先作为数值处理
    node_features = df[['relativeArea', 'compactness', 'surfaceType', 'nx', 'ny', 'nz']].values
    x = torch.tensor(node_features, dtype=torch.float)
    
    # --- 标签 (Labels) ---
    y = torch.tensor(df['label'].values, dtype=torch.long)
    
    # --- 边与边属性 (Edges & Edge Attributes) ---
    edge_index = []
    edge_attr = []
    
    for i, row in df.iterrows():
        # 处理邻居字符串 "8 2 10" -> [7, 1, 9] (转为 0-indexed)
        neighbors = str(row['neighbors']).split()
        edge_types = str(row['edge_types']).split()
        
        for nb_id_str, e_type_str in zip(neighbors, edge_types):
            if not nb_id_str: continue
            
            u = int(row['id']) - 1 # 0-indexed
            v = int(nb_id_str) - 1
            e_type = float(e_type_str)
            
            # 添加双向边
            edge_index.append([u, v])
            edge_attr.append([e_type])
            
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

# 2. 定义图卷积神经网络
class RivetGNN(torch.nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim):
        super(RivetGNN, self).__init__()
        
        # 边缘条件神经网络：根据边属性（1, 0, -1）生成卷积权重
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(edge_features, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, node_features * hidden_dim)
        )
        self.conv1 = NNConv(node_features, hidden_dim, nn1)
        
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(edge_features, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, hidden_dim * hidden_dim)
        )
        self.conv2 = NNConv(hidden_dim, hidden_dim, nn2)
        
        self.classifier = torch.nn.Linear(hidden_dim, 2) # 二分类：背景 vs 铆钉

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 第一层卷积
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        
        # 第二层卷积
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        
        # 分类层
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

# 3. 训练脚本
def train():
    # 加载数据
    data = load_cad_data('data/train_data.csv')
    
    # 初始化模型 (节点特征数=6, 边特征数=1, 隐藏层=32)
    model = RivetGNN(node_features=6, edge_features=1, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("开始训练 RivetGNN 模型...")
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss.item():.4f}')
            
    # 推理验证
    model.eval()
    pred = model(data).argmax(dim=1)
    
    print("\n训练完成！")
    print("AI 识别出的铆钉面 ID 列表:")
    rivet_indices = (pred == 1).nonzero(as_tuple=True)[0]
    rivet_ids = [idx.item() + 1 for idx in rivet_indices]
    print(rivet_ids)
    
    # 保存模型
    torch.save(model.state_dict(), 'rivet_gnn.pth')
    print("\n模型已保存至: rivet_gnn.pth")

if __name__ == "__main__":
    train()
