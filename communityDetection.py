import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns
import networkx as nx
import matplotlib.cm as cm
import community as community_louvain
from networkx.algorithms.community import louvain_communities

class Community:
    def __init__(self, samples, labels) -> None:
        self.samples = samples
        self.labels = labels
        m = samples.shape[1]
        self.num_nodes = m
        self.cor =np.zeros((m, m))
        self.adj_matrix = np.zeros((m, m))

    def get_cor_matrix(self, draw = False):
        m = self.samples.shape[1]
        df = pd.DataFrame(self.samples, columns=self.labels)
        correlation = df.corr()
        if draw:
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
            plt.title('Correlation Matrix')
            plt.show()
        return np.array(correlation)

    def build_graph(self, epsilon):
        # if |cor(i, j)| < epsilon, we will not connect it
        m = self.num_nodes
        self.cor = self.get_cor_matrix()
        for i in range(m):
            for j in range(m):
                if abs(self.cor[i][j]) < epsilon:
                    self.adj_matrix[i][j] = 0
                else:
                    self.adj_matrix[i][j] = self.cor[i][j]
        return self.adj_matrix
    
    def calculate_signed_modularity(self, G, communities):
        w = np.sum([abs(data['weight']) for u, v, data in G.edges(data=True)])
        w_plus = np.sum([data['weight'] for u, v, data in G.edges(data=True) if data['weight'] > 0])
        w_minus = w - w_plus

        modularity = 0.0
        for u in G.nodes:
            for v in G.nodes:
                if communities[u] == communities[v]:
                    w_ij = G[u][v]['weight'] if G.has_edge(u, v) else 0
                    w_i_plus = sum([G[u][n]['weight'] for n in G[u] if G[u][n]['weight'] > 0])
                    w_j_plus = sum([G[v][n]['weight'] for n in G[v] if G[v][n]['weight'] > 0])
                    w_i_minus = sum([G[u][n]['weight'] for n in G[u] if G[u][n]['weight'] < 0])
                    w_j_minus = sum([G[v][n]['weight'] for n in G[v] if G[v][n]['weight'] < 0])
                    
                    if w_plus > 0:
                        modularity += w_ij - (w_i_plus * w_j_plus / (2 * w_plus))
                    if w_minus > 0:
                        modularity += w_i_minus * w_j_minus / (2 * w_minus)

        modularity /= (2 * w)
        return modularity

    def BGLL(self, G, target_communities = None):
        

        n = len(G.nodes)
        communities = {i: i for i in range(n)}
        
        best_modularity = self.calculate_signed_modularity(G, communities)
        improvement = True
        
        while improvement:
            improvement = False
            for u in G.nodes:
                best_community = communities[u]
                best_gain = 0
                current_community = communities[u]
                
                for v in G.neighbors(u):
                    if communities[v] != current_community:
                        communities[u] = communities[v]
                        new_modularity = self.calculate_signed_modularity(G, communities)
                        gain = new_modularity - best_modularity
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_community = communities[v]
                
                communities[u] = best_community
                if best_gain > 0:
                    best_modularity += best_gain
                    improvement = True
            # 当设定的目标社区数量大于无限制下分组的社区数量，提前结束分组
            if target_communities:
                if len(set(communities.values())) < target_communities:
                    break
            
        
        # 当设定的目标社区数量小于无限制下分组的社区数量
        if target_communities:
            while len(set(communities.values())) > target_communities:
                community_pairs = list(nx.edge_betweenness_centrality(G).items())
                community_pairs.sort(key=lambda x: x[1], reverse=True)
                for (u, v), _ in community_pairs:
                    if communities[u] != communities[v]:
                        old_community = communities[v]
                        new_community = communities[u]
                        for node in communities:
                            if communities[node] == old_community:
                                communities[node] = new_community
                        break
        # 如果社区数量仍然小于目标社区数量，则强制分裂社区
            while len(set(communities.values())) < target_communities:
                largest_community = max(set(communities.values()), key=list(communities.values()).count)
                nodes_in_largest = [node for node, comm in communities.items() if comm == largest_community]
                for i, node in enumerate(nodes_in_largest):
                    if len(set(communities.values())) >= target_communities:
                        break
                    communities[node] = max(communities.values()) + 1 + i
                    
        final_communities = {}
        for node, community in communities.items():
            if community in final_communities:
                final_communities[community].append(node)
            else:
                final_communities[community] = [node]
        return list(final_communities.values())

    def Visialize(self, communities): 
        adj_matrix = self.adj_matrix
        G = nx.from_numpy_array(adj_matrix)
        pos = nx.spring_layout(G)
        colors = cm.rainbow(np.linspace(0, 1, len(communities)))

        for community, color in zip(communities, colors):
            nx.draw_networkx_nodes(G, pos, nodelist=community, node_color=[color] * len(community), label=f'Community {communities.index(community) + 1}', edgecolors='face', linewidths=0.5)

        nx.draw_networkx_edges(G, pos, alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')
        plt.legend()
        plt.show()
    
    def partition(self, cov_matrix):
        # Step 1: Construct the graph
        G = nx.Graph()
        m, _ = cov_matrix.shape
        for i in range(m):
            for j in range(i+1, m):
                if cov_matrix[i, j] > 0:  # Only consider positive correlations
                    G.add_edge(i, j, weight=cov_matrix[i, j])
        
        # Step 2: Apply Louvain method
        communities = louvain_communities(G, weight='weight')
        # Convert to list of lists format
        community_list = [list(community) for community in communities]
        return community_list
# test 
# n = 100  # 样本数
# m = 20    # 特征数
# epsilon = 0.1  # 相关性阈值
# target_communities = 3 # 目标社区数

# from utils import get_random_array, visualize_joint_distribution
# dim = [n, m]
# distribution = 'multivariate_normal'
# A = np.random.rand(m, m)
# cov = np.dot(A, A.T)
# mean = np.zeros(m)
# params = {"mean": mean, 
#           "cov": cov}
# data = get_random_array(dim, distribution, params)
# print(data.shape)
# visualize_joint_distribution(data)

# 

# test = Community(data, labels)
# res = test.get_cor_matrix(True)
# communities = test.BGLL(epsilon)
# communities_2 = test.BGLL(epsilon, target_communities)
# print("----无限制的社区检测结果为----")
# print(communities)
# print(f"----分为{target_communities}个社区检测结果为----")
# print(communities_2)
# test.Visialize(communities_2)













    
    


        