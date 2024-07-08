import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.linalg import logm
from sklearn.cluster import KMeans

def random_walk_matrix(G, L):
    """计算随机游走矩阵"""
    A = nx.to_numpy_array(G)
    D = np.diag(np.sum(A, axis=1))
    T = np.linalg.inv(D).dot(A)
    
    # Compute T^L (L-step transition matrix)
    T_L = np.linalg.matrix_power(T, L)
    
    return T_L

def compute_measure_space_embedding(G, L):
    """计算度量空间嵌入"""
    T_L = random_walk_matrix(G, L)
    measures = np.zeros_like(T_L)
    for i in range(len(G.nodes)):
        measures[i] = np.sum([np.linalg.matrix_power(T_L, l)[:,i] for l in range(1, L+1)], axis=0) / L
    return measures

def k_means_clustering(measures, k):
    """在度量空间中进行k-means聚类"""
    kmeans = KMeans(n_clusters=k, random_state=0).fit(measures)
    return kmeans.labels_

def DER(G, L, k, max_iter=100):
    """DER算法的主循环"""
    measures = compute_measure_space_embedding(G, L)
    labels = k_means_clustering(measures, k)
    
    for _ in range(max_iter):
        clusters = {i: [] for i in range(k)}
        for node, label in enumerate(labels):
            clusters[label].append(node)
        
        new_measures = np.zeros_like(measures)
        for i in range(k):
            cluster = clusters[i]
            if len(cluster) > 0:
                new_measures[cluster] = np.mean(measures[cluster], axis=0)
        
        new_labels = k_means_clustering(new_measures, k)
        
        if np.array_equal(labels, new_labels):
            break
        
        labels = new_labels
    
    return labels

# 示例
from dataCenter import data_generator
import pandas as pd
sample_dim = 1000
data = data_generator(sample_dim)
df = pd.DataFrame(data)
adj_array = np.array(df.corr())
print("the adj_array type:", type(adj_array))
print("the adj_array size:", adj_array.shape)
G = nx.from_numpy_array(adj_array)

L = 20  # 随机游走的步数
k = 3  # 社区数量

labels = DER(G, L, k)

result = {}
for index, value in enumerate(labels):
    if value not in result:
        result[value] = []
    result[value].append(index)

print("----DER检测结果为----")
print(result)


from utils import k_tau_network
# 网络构建参数
tau = 0.7
K = 4
G = k_tau_network(adj_array, tau, K)
labels = DER(G, L, k)
result = {}
for index, value in enumerate(labels):
    if value not in result:
        result[value] = []
    result[value].append(index)

print("----DER + K-tau检测结果为----")
print(result)