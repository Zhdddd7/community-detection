from dataCenter import data_generator, fill_random_zeros
from matrixFill import matrixFill
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import networkx as nx
nan_rate = 0.1
X = data_generator()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).T
print(f"the type of X_scaled is {type(X_scaled)}")
X = fill_random_zeros(X_scaled, nan_rate)
X_rec = matrixFill(X, False)
print(f"the shape of X_rec is {X_rec.shape}")

#############################
# use MLP + KMeans

# from MLP import MLP, train
# # 参数设置
# input_dim = 1000
# hidden_dim = 128
# output_dim = 64
# # 初始化模型
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # model = MLP(input_dim, hidden_dim, output_dim).to(device)
# model = MLP(input_dim, hidden_dim, output_dim).to(device)
# num_epochs = 300
# print(f"training MLP, running on {device}")
# # train(model, X_rec, num_epochs, True)
# model_file = "models/MLP_epoch_299_model.pt"
# model = torch.load(model_file).to(device)
# with torch.no_grad():
#     function_embeddings = model(torch.tensor(X_scaled, dtype=torch.float32).to(device)).cpu().numpy()
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=3)  
# labels = kmeans.fit_predict(function_embeddings)
# from utils import print_labels
# print("----MLP + KMeans Clustering----")
# print_labels(labels)


############################
# use GraphAttention to cluster
# build a graph

data = X_rec.T
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

emb_size = 1000
hidden = 128
num_classes = 3
node_features = torch.Tensor(data.T)
node_features = node_features.unsqueeze(0)
print(node_features.shape)
from Attention import AttentionGNN, train
# 初始化模型
num_heads = 8
num_iters = 5
model = AttentionGNN(num_classes, num_nodes, emb_size, hidden, num_heads, num_iters)
# 训练模型
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lambda_reg = 0.5
num_epochs = 300
# trian the model and save the model to ./models
# train(model, node_features, adj,num_epochs, True)

# 输出最终的社区分配
model_file = "./models/new_attention_299_model_withmask.pt"
model = torch.load(model_file)
logits = model(torch.tensor(adj, dtype=torch.float32).unsqueeze(0), node_features)
predictions = model.predict(logits)
pre = predictions.squeeze(dim=0)
labels = pre.argmax(dim=1)
pre = labels.tolist()
from utils import print_labels
print("----The Attention Graph prediction----")
print_labels(pre)