import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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

def sim_network(sim_matrix, draw = False):
    pass

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

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

def build_unweighted_bethe_hessian(adj, associative=True, r=None):
    n = adj.shape[0]
    degrees = np.array(adj.sum(axis=1)).flatten()
    if r is None:
        r = np.sqrt(degrees.mean())
    
    if associative:
        D_r = sp.diags(degrees - r)
    else:
        D_r = sp.diags(degrees + r)
    
    I = sp.identity(n)
    H = (r**2 - 1) * I + D_r - adj
    return H

def bethe_hessian_node_features(adj, emb_size, associative=True):
    H = build_unweighted_bethe_hessian(adj, associative=associative)
    vals, vecs = eigsh(H, k=emb_size, which="SA")
    return vecs.astype(np.float32)

def graph2graph_tuple(graph_nx, node_features):
    graph_nx = graph_nx.to_directed()
    n = len(graph_nx.nodes)
    m = len(graph_nx.edges)
    senders, receivers = zip(*graph_nx.edges)
    
    dt = {
        "nodes": node_features if node_features is not None else [None for _ in range(n)],
        "edges": None,
        "receivers": np.array(receivers, dtype=np.int32),
        "senders": np.array(senders, dtype=np.int32),
        "globals": None,
        "n_node": n,
        "n_edge": m
    }
    
    return dt
   