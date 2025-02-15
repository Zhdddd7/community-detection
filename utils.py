import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import copy
from dataCenter import data_generator
def k_tau_network(corr_matrix, tau, K, draw = False):
    """
    使用K-tau方法基于相关性矩阵构建网络图。

    参数：
    corr_matrix (numpy.ndarray): m * m的相关性矩阵
    tau (float): 阈值，过滤掉低于tau的相关性
    K (int): 最小连接数，确保每个节点至少有K个连接

    返回：
    networkx.Graph: 构建的网络图
    """
    m = corr_matrix.shape[0]
    G = nx.Graph()
    
    # 添加节点
    for i in range(m):
        G.add_node(i)
    
    # τ-step: 添加边，保留大于等于tau的相关性
    for i in range(m):
        for j in range(i + 1, m):
            if corr_matrix[i, j] >= tau:
                G.add_edge(i, j, weight=corr_matrix[i, j])
    
    # K-step: 确保每个节点至少有K个连接
    for i in range(m):
        neighbors = list(G.neighbors(i))
        if len(neighbors) < K:
            sorted_indices = np.argsort(corr_matrix[i])[::-1]
            count = 0
            for index in sorted_indices:
                if index != i and index not in neighbors:
                    G.add_edge(i, index, weight=corr_matrix[i, index])
                    count += 1
                if count >= K - len(neighbors):
                    break
    

    if draw:
        # 绘制网络图
        pos = nx.spring_layout(G)
        weights = nx.get_edge_attributes(G, 'weight').values()
        nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, width=list(weights))
        plt.show()
    return G

def epsilon_network(corr_matrix, epsilon, draw = False):
    """
    使用epsilon方法基于相关性矩阵构建网络图。

    参数：
    corr_matrix (numpy.ndarray): m * m的相关性矩阵
    epsilon (float): 阈值，过滤掉绝对值低于epsilon的相关性

    返回：
    networkx.Graph: 构建的网络图
    """
    m = corr_matrix.shape[0]
    G = nx.Graph()
    
    # 添加节点
    for i in range(m):
        G.add_node(i)
    
    # 添加边，保留绝对值大于或等于epsilon的相关性
    for i in range(m):
        for j in range(i + 1, m):
            if abs(corr_matrix[i, j]) >= epsilon:
                G.add_edge(i, j, weight=corr_matrix[i, j])
    if draw:
        # 绘制网络图
        pos = nx.spring_layout(G)
        weights = nx.get_edge_attributes(G, 'weight').values()
        nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, width=list(weights))
        plt.show()
    return G

def draw_custom_colored_graph(G, color_groups):
    """
    使用color-map来进行颜色映射，把其中一部分的节点映射到一个颜色，而其中的一些节点映射
    到另一个颜色，主要用于展示图的聚类效果。
    """
    pos = nx.spring_layout(G)
    weights = nx.get_edge_attributes(G, 'weight').values()

    # Create a color map for nodes
    color_map = []
    for node in G:
        for group_index, group in enumerate(color_groups):
            if node in group:
                color_map.append(f'C{group_index}')  # Assign a color from the default color cycle

    nx.draw(G, pos, with_labels=True, node_size=500, node_color=color_map, font_size=10, width=list(weights))
    plt.show()

def print_labels(labels):
    result = {}
    for index, value in enumerate(labels):
        if value not in result:
            result[value] = []
        result[value].append(index)
    print(result)

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

import torch.nn as nn
class DynamicContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(DynamicContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        label = (euclidean_distance < self.margin).float()  # 动态调整标签
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
