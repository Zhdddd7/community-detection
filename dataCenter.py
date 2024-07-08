import numpy as np
import pandas as pd

def data_generator(sample_dim = 1000):
    """
    产生一个sample_dim * 20维的数组
    """
    sample_dim = 1000
    # [x1, x2,...xm]
    # 每一个x服从正态分布
    m = 10
    samples = np.zeros((m, sample_dim))
    for i in range(samples.shape[0]):
        samples[i] = np.random.randn(sample_dim)
        
    # 定义函数
    def func0(x): return x[0] + x[1]
    def func1(x): return x[2] - x[0]
    def func2(x): return x[4] * x[5]
    def func3(x): return x[6] + x[8] + x[7]
    def func4(x): return x[0]**2 + 2*x[0]*x[1] + x[1]**2
    def func5(x): return 6* (x[4] )
    def func6(x): return np.exp(x[3] + x[2])
    def func7(x): return x[4] + x[5] - x[6]
    def func8(x): return x[7] * x[8] * x[6]
    def func9(x): return x[0]**3 + 3*x[0]**2*x[1] + 3*x[0]*x[1]**2 + x[1]**3
    def func10(x): return 5* (x[2] + x[3] )
    def func11(x): return np.exp(x[4] - x[5])
    def func12(x): return x[0]**2 - x[1]**2
    def func13(x): return  - x[0] -x[1]
    def func14(x): return x[1] * x[2] + x[0]
    def func15(x): return x[4]**2 + 2*x[4]*x[5] - x[5]**2
    def func16(x): return 0.5 * (x[6] * x[7] )
    def func17(x): return np.exp(x[6]* x[7])
    def func18(x): return x[0] - x[1] + x[2]
    def func19(x): return -x[0] - 0.2* x[1]

    functions = [func1, func2, func3, func4, func5, func6, func7, func8, func9, func10,
                func11, func12, func13, func14, func15, func16, func17, func18, func19, func0]
    res = []
    for func in functions:
        res.append(func(samples))
    res = np.array(res)
    res = res.T
    
    return res

data = data_generator()
m = data.shape[1]
labels = [f'feature_{i}' for i in range(m)]
from communityDetection import Community
test_c = Community(data, labels)
cor= test_c.get_cor_matrix()

# test on graph-building 

from utils import k_tau_network, epsilon_network, draw_custom_colored_graph
tau = 0.7
K = 5
epsilon = 0.5
G = k_tau_network(cor,tau, K)
# G = epsilon_network(cor, epsilon)

color_groups = [
    [0, 1, 4, 9, 12, 13, 8, 19],  # Group 1
    [2, 3, 7, 8, 11, 15, 16, 17],  # Group 2
    [5, 6, 10, 14]                 # Group 3
]

# draw_custom_colored_graph(G, color_groups)
# com_lou = test_c.BGLL(G)
# print("----The Louvain Detection----")
# print(com_lou)
# draw_custom_colored_graph(G, com_lou)

import igraph as ig
import networkx as nx
import leidenalg
# cov_matrix = test_c.get_cor_matrix()
# cov_matrix[cov_matrix < 0] = 0
# G = ig.Graph.Weighted_Adjacency(cov_matrix.tolist(), mode=ig.ADJ_UNDIRECTED, attr="weight")
# part = leidenalg.find_partition(G, leidenalg.RBConfigurationVertexPartition, weights='weight')
# print("----Leidenalg检测结果为----")
# print("the leidenalg's partition is:", part)
# print(type(part))

# color_leiden = [
#     [3, 8, 13, 19], 
#     [4, 6, 10],
#     [5, 9, 17], 
#     [7, 15, 16], 
#     [0, 12, 18],
#     [1, 14],
#     [2, 11]
# ]
# draw_custom_colored_graph(G, color_leiden)

# Node2vec = [[0, 1, 8, 9, 12, 17, 18, 19], 
#             [2, 7],
#             [3, 4, 5, 6, 10, 11, 13, 14, 15, 16]
#             ]
# draw_custom_colored_graph(G, Node2vec)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralBiclustering

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
# 计算特征之间的相似性矩阵（余弦相似性）
similarity_matrix = cosine_similarity(data_scaled.T)
# 使用双向聚类对特征进行聚类
n_clusters = 4  
bicluster = SpectralBiclustering(n_clusters=n_clusters, method='log')
bicluster.fit(similarity_matrix)
# 聚类结果
feature_labels = bicluster.row_labels_

# 打印每个特征的聚类标签
feature_names = [f"feature{i}" for i in range(data.shape[1])]
feature_cluster_mapping = pd.DataFrame({'Feature': feature_names, 'Cluster': feature_labels})
feature_cluster_mapping = feature_cluster_mapping.sort_values(by='Cluster')

# 格式化输出
unique_labels = set(feature_labels)
print("----SpectralBiclustering检测结果为----")
for cluster in unique_labels:
    features_in_cluster = feature_cluster_mapping[feature_cluster_mapping['Cluster'] == cluster]['Feature'].tolist()
    print(f"类别 {cluster}: {', '.join(features_in_cluster)}")
