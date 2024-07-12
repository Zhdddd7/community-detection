from sklearn.preprocessing import StandardScaler
from dataCenter import data_generator

X = data_generator()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).T
print(f"the size of X_scaled is {X_scaled.shape}")

import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 参数设置
input_dim = 1000
hidden_dim = 128
output_dim = 64

# 初始化模型
model = MLP(input_dim, hidden_dim, output_dim)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

margin = 1.0
criterion = ContrastiveLoss(margin)
optimizer = optim.Adam(model.parameters(), lr=0.001)

from torch.utils.data import Dataset, DataLoader

class FunctionDataset(Dataset):
    def __init__(self, X):
        self.X = X
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x1 = self.X[idx]
        x2 = self.X[(idx + 1) % len(self.X)]
        label = torch.tensor(1.0 if idx % 2 == 0 else 0.0)
        return x1, x2, label

dataset = FunctionDataset(X_scaled)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

num_epochs = 300
def train(num_epochs):
    for epoch in range(num_epochs):
        for x1, x2, label in dataloader:
            x1, x2, label = x1.float(), x2.float(), label.float()
            
            # 前向传播
            output1 = model(x1)
            output2 = model(x2)
            loss = criterion(output1, output2, label)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
train(num_epochs)
with torch.no_grad():
    function_embeddings = model(torch.tensor(X_scaled).float()).numpy()
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)  
labels = kmeans.fit_predict(function_embeddings)
from utils import print_labels
print_labels(labels)
