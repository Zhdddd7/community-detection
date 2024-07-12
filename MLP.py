from sklearn.preprocessing import StandardScaler
from dataCenter import data_generator

X = data_generator()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).T
print(f"the size of X_scaled is {X_scaled.shape}")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = MLP(input_dim, hidden_dim, output_dim).to(device)
model = MLP(input_dim, hidden_dim, output_dim).to(device)
from utils import DynamicContrastiveLoss
margin = 1.0
criterion = DynamicContrastiveLoss(margin).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

from dataCenter import FunctionDataset, DataLoader

dataset = FunctionDataset(X_scaled)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
num_epochs = 300
print(f"running on {device}")
def train(num_epochs):
    for epoch in range(num_epochs):
        for x1, x2 in dataloader:     
            # 前向传播
            x1 = x1.to(device)
            x2 = x2.to(device)
            output1 = model(x1)
            output2 = model(x2)
            loss = criterion(output1, output2)           
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
train(num_epochs)
with torch.no_grad():
    function_embeddings = model(torch.tensor(X_scaled, dtype=torch.float32).to(device)).cpu().numpy()
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)  
labels = kmeans.fit_predict(function_embeddings)
from utils import print_labels
print_labels(labels)
