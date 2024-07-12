# use Node2Vec + Kmeans
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans

# 设置环境变量以便调试
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class Node2VecDataset(Dataset):
    def __init__(self, walks, context_size):
        self.walks = walks
        self.context_size = context_size

    def __len__(self):
        return len(self.walks)

    def __getitem__(self, idx):
        walk = self.walks[idx]
        input_node = walk[0]
        context_nodes = walk[1:self.context_size + 1]
        return torch.tensor(input_node, dtype=torch.long), torch.tensor(context_nodes, dtype=torch.long)

class Node2VecModel(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(Node2VecModel, self).__init__()
        self.embeddings = nn.Embedding(num_nodes, embedding_dim)

    def forward(self, nodes):
        return self.embeddings(nodes)

def build_correlation_network(data, tau=0.7, K=5):
    n_features = data.shape[1]
    corr_matrix = np.corrcoef(data, rowvar=False)  # 行为样本，列为特征
    
    G = nx.Graph()
    for i in range(n_features):
        G.add_node(i)
        
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if corr_matrix[i, j] >= tau:
                G.add_edge(i, j, weight=corr_matrix[i, j])
    
    for i in range(n_features):
        neighbors = sorted([(j, corr_matrix[i, j]) for j in range(n_features) if j != i], key=lambda x: -x[1])[:K]
        for j, weight in neighbors:
            if not G.has_edge(i, j):
                G.add_edge(i, j, weight=weight)
    
    return G

def generate_walks(G, num_walks, walk_length):
    walks = []
    nodes = list(G.nodes())
    for _ in range(num_walks):
        for node in nodes:
            walk = [node]
            for _ in range(walk_length - 1):
                neighbors = list(G.neighbors(walk[-1]))
                if neighbors:
                    walk.append(np.random.choice(neighbors))
                else:
                    break
            walks.append(walk)
    return walks

def train_node2vec(G, embedding_dim=64, walk_length=30, num_walks=200, context_size=10, epochs=100, batch_size=128, lr=0.01, K = 5, tau = 0.7):
    walks = generate_walks(G, num_walks, walk_length)
    dataset = Node2VecDataset(walks, context_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Node2VecModel(len(G.nodes()), embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for input_node, context_nodes in dataloader:
            input_node = input_node.to(device)
            context_nodes = context_nodes.to(device).view(-1)

            optimizer.zero_grad()
            input_embeddings = model(input_node)

            # Expand input_embeddings to match context_nodes size
            input_embeddings = input_embeddings.repeat(context_nodes.size(0) // input_embeddings.size(0), 1)
            # Ensure context_nodes are within valid range
            if context_nodes.max() >= input_embeddings.size(1):
                print(f"Error: context_nodes value {context_nodes.max()} exceeds number of classes {input_embeddings.size(1)}")
                continue

            loss = criterion(input_embeddings, context_nodes)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
        if epoch % 100 ==0:
            file_path = "./models/"
            torch.save(model, f"{file_path}dim_{embedding_dim}walk_{walk_length}_K_{K}T_{tau}_{epoch}_model.pt")

    return model

def get_embeddings(model, G):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        embeddings = model.embeddings.weight.cpu().numpy()
    return embeddings

def community_detection(embeddings, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    return kmeans.labels_
# test
tau = 0.7
K = 4
embedding_dim=128
walk_length=30
num_walks=200
context_size=10
epochs= 300
batch_size=128
lr=0.01
n_clusters = 3
sample_dim = 1000
from dataCenter import data_generator
data = data_generator(sample_dim) # 假设有1000个样本，每个样本20个特征
print("the data dimension is", data.shape)
# 构建相关性网络
G = build_correlation_network(data, tau, K)

# training 
def train():
    # 训练Node2Vec模型
    model = train_node2vec(G, embedding_dim, walk_length, num_walks, context_size, epochs, batch_size, lr, K, tau)
    # 获取节点嵌入
    embeddings = get_embeddings(model, G)
    # 进行社区检测
    labels = community_detection(embeddings, n_clusters)
    # 创建一个空字典
    result = {}
    for index, value in enumerate(labels):
        if value not in result:
            result[value] = []
        result[value].append(index)
    print("----Node2Vec检测结果为----")
    print(result)

# load the model to do the clustering
def predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_file = "./models/K_4T_0.7_200_model.pt"
    model = torch.load(model_file).to(device)
    print("running on:", device)
    embeddings = get_embeddings(model, G)
    labels = community_detection(embeddings, n_clusters)
    result = {}
    for index, value in enumerate(labels):
        if value not in result:
            result[value] = []
        result[value].append(index)

    print("----Node2Vec检测结果为----")
    print(result)

predict()