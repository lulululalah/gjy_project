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
    feature_cols = ['relativeArea', 'compactness', 'surfaceType', 'nx', 'ny', 'nz', 
                    'centerZ', 'meanCurvature', 'radius', 'numWires', 'numEdges']
    node_features = df[feature_cols].values
    
    # --- 特征标准化 (Normalization) ---
    # 工业模型尺度差异巨大，必须进行标准化处理
    feat_mean = node_features.mean(axis=0)
    feat_std = node_features.std(axis=0) + 1e-6
    node_features = (node_features - feat_mean) / feat_std
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    # --- 标签 (Labels) ---
    y = torch.tensor(df['label'].values, dtype=torch.long)
    
    # --- 边与边属性 (Edges & Edge Attributes) ---
    edge_index = []
    edge_attr = []
    
    for i, row in df.iterrows():
        nb_str = str(row['neighbors']) if pd.notna(row['neighbors']) else ""
        et_str = str(row['edge_types']) if pd.notna(row['edge_types']) else ""
        
        neighbors = nb_str.split()
        edge_types = et_str.split()
        
        for nb_id_str, e_type_str in zip(neighbors, edge_types):
            if not nb_id_str: continue
            u = int(row['id']) - 1
            v = int(nb_id_str) - 1
            e_type = float(e_type_str)
            edge_index.append([u, v])
            edge_attr.append([e_type])
            
    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        # 如果没有边，创建一个空的 [2, 0] 形状的 tensor 供 GCN 使用
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

# 2. 定义增强型图卷积神经网络
class RivetGNN(torch.nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim):
        super(RivetGNN, self).__init__()
        
        # 卷积权重生成网络
        def create_nn(in_f, out_f):
            return torch.nn.Sequential(
                torch.nn.Linear(edge_features, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, in_f * out_f)
            )

        self.conv1 = NNConv(node_features, hidden_dim, create_nn(node_features, hidden_dim))
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        
        self.conv2 = NNConv(hidden_dim, hidden_dim, create_nn(hidden_dim, hidden_dim))
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        
        self.conv3 = NNConv(hidden_dim, hidden_dim, create_nn(hidden_dim, hidden_dim))
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim)
        
        self.classifier = torch.nn.Linear(hidden_dim, 3) 

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 第一层 + 残差准备
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        
        # 第二层 (带残差连接)
        identity = x
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
        x = x + identity # Residual connection
        
        # 第三层
        x = F.relu(self.bn3(self.conv3(x, edge_index, edge_attr)))
        
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

# 3. 训练脚本
def train():
    data = load_cad_data('data/full_training_set.csv')
    in_channels = data.num_node_features
    model = RivetGNN(node_features=in_channels, edge_features=1, hidden_dim=64) # 提升隐藏层维度
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005) # 稍微调低学习率
    
    print("开始训练增强型 RivetGNN 模型 (Residual + BatchNorm)...")
    model.train()
    for epoch in range(150): # 增加训练轮数
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss.item():.4f}')
            
    torch.save(model.state_dict(), 'rivet_gnn.pth')
    print("\n增强型模型已保存至: rivet_gnn.pth")

if __name__ == "__main__":
    train()
