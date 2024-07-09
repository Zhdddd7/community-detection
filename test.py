
from utils import draw_custom_colored_graph
from dataCenter import data_generator
data = data_generator()
m = data.shape[1]
labels = [f'feature_{i}' for i in range(m)]
from communityDetection import Community
test_c = Community(data, labels)
cor= test_c.get_cor_matrix()
from utils import k_tau_network, epsilon_network, draw_custom_colored_graph
tau = 0.7
K = 5
G = k_tau_network(cor,tau, K)
labels =[[0, 2, 4, 7, 14, 16, 19], [1, 8, 11, 12, 18],[3, 5, 6, 9, 10, 13, 15, 17]]
draw_custom_colored_graph(G, labels)
