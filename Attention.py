import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np

class Head(nn.Module):
    def __init__(self, num_heads, hidden):
        super(Head, self).__init__()
        self.num_heads = num_heads
        self.hidden = hidden
        self.head_dim = hidden // num_heads
        self.linear_val = nn.Linear(hidden, hidden, bias=False)
        self.linear_key = nn.Linear(hidden, hidden, bias=False)
        self.linear_query = nn.Linear(hidden, hidden, bias=False)

    def forward(self, inputs):
        batch_size, num_nodes, hidden_dim = inputs.size()
        vals = self.linear_val(inputs).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        keys = self.linear_key(inputs).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        queries = self.linear_query(inputs).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        queries = queries / (self.head_dim ** 0.5)
        return vals, keys, queries

class GraphAttention(nn.Module):
    def __init__(self, hidden, num_heads):
        super(GraphAttention, self).__init__()
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.head = Head(num_heads, hidden)
        self.output_proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        self.norm1 = nn.BatchNorm1d(hidden)
        self.norm2 = nn.BatchNorm1d(hidden)

    def forward(self, inputs, adj):
        batch_size, num_nodes, hidden_dim = inputs.size()
        vals, keys, queries = self.head(inputs)
        
        # Compute attention scores
        attention_scores = torch.einsum('bnqh,bmqh->bnmq', queries, keys)
        attention_scores = attention_scores / np.sqrt(self.head_dim)
        attention_scores = F.softmax(attention_scores, dim=-1)
        
        # Compute new node features
        context = torch.einsum('bnmq,bmvh->bnqh', attention_scores, vals).contiguous()
        context = context.view(batch_size, num_nodes, self.hidden)
        
        output_projected = self.output_proj(context)
        output_projected = self.norm1((output_projected + inputs).view(-1, self.hidden)).view(batch_size, num_nodes, self.hidden)
        normalized = self.norm2((self.mlp(output_projected) + output_projected).view(-1, self.hidden)).view(batch_size, num_nodes, self.hidden)
        return normalized

class AttentionGNN(nn.Module):
    def __init__(self, num_classes, num_nodes, emb_size, hidden, num_heads, num_iters):
        super(AttentionGNN, self).__init__()
        self.num_classes = num_classes
        self.num_nodes = num_nodes
        self.emb_size = emb_size
        self.hidden = hidden
        self.num_heads = num_heads
        self.num_iters = num_iters
        self.embed = nn.Embedding(num_nodes, emb_size)
        self.linear = nn.Linear(emb_size, hidden)
        self.norm = nn.BatchNorm1d(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        self.gnn_attentions = nn.ModuleList([GraphAttention(hidden, num_heads) for _ in range(num_iters)])
        self.final = nn.Linear(hidden, num_classes)

    def forward(self, adj, inputs):
        if inputs is not None:
            embs = self.linear(inputs)
            embs = F.relu(embs)
            embs = self.norm(embs.view(-1, self.hidden)).view(inputs.size(0), inputs.size(1), self.hidden)
            embs = self.mlp(embs) + embs
        else:
            embs = self.embed(torch.arange(self.num_nodes))

        for i in range(self.num_iters):
            embs = self.gnn_attentions[i](embs, adj)

        logits = self.final(embs)
        return logits

    def predict(self, logits):
        return F.softmax(logits, dim=-1)


# Loss Function
def modularity_loss(U, A):
    d = A.sum(1)
    D = torch.outer(d, d) / A.sum()
    B = A - D
    M = -torch.trace(torch.matmul(U.T, torch.matmul(B, U))) / torch.norm(A, 1)
    return M

def regularizer(U, C):
    reg = torch.sum((U.sum(0) - 1/C)**2)
    return reg

def community_detection_loss(U, A, C, lambda_reg):
    M = modularity_loss(U, A)
    R = regularizer(U, C)
    loss = M + lambda_reg * R
    return loss

# build a graph
from dataCenter import data_generator
data = data_generator()
m = data.shape[1]
labels = [f'feature_{i}' for i in range(m)]
from communityDetection import Community
test_c = Community(data, labels)
cor= test_c.get_cor_matrix()
from utils import k_tau_network
tau = 0.7
K = 5
G = k_tau_network(cor,tau, K)
num_nodes = G.number_of_nodes()
adj = nx.to_numpy_array(G)  # 转换为邻接矩阵
print('the node num is', num_nodes)

# 为每个节点生成随机特征
emb_size = 1000
hidden = 128
num_classes = 3
node_features = torch.Tensor(data.T)
node_features = node_features.unsqueeze(0)

# 初始化模型
num_heads = 8
num_iters = 5
model = AttentionGNN(num_classes, num_nodes, emb_size, hidden, num_heads, num_iters)


# 训练模型
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lambda_reg = 0.5


def train(model, node_features, num_epochs, saved = False):
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(torch.tensor(adj, dtype=torch.float32).unsqueeze(0), node_features)
        U = model.predict(logits).squeeze(0)  # 节点的社区分布概率
        loss = community_detection_loss(U, torch.tensor(adj, dtype=torch.float32), num_classes, lambda_reg)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
        if saved:
            if (epoch +1) % 100 ==0:
                file_path = "./models/"
                torch.save(model, f"{file_path}attention_{epoch}_model.pt")
                print(f"the {epoch} model is saved.")

num_epochs = 300
# trian the model and save the model to ./models
# train(model, node_features, num_epochs, True)

# 输出最终的社区分配
model_file = "./models/attention_299_model.pt"
model = torch.load(model_file)
logits = model(torch.tensor(adj, dtype=torch.float32).unsqueeze(0), node_features)
predictions = model.predict(logits)
pre = predictions.squeeze(dim=0)
labels = pre.argmax(dim=1)
pre = labels.tolist()
from utils import print_labels
print("----The Attention Graph prediction----")
print_labels(pre)
