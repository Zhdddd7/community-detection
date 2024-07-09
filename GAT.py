import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

import utils

class Head(nn.Module):
    def __init__(self, num_heads, hidden):
        super(Head, self).__init__()
        self.num_heads = num_heads
        self.hidden = hidden

        self.linear_val = nn.Linear(hidden, hidden, bias=False)
        self.linear_key = nn.Linear(hidden, hidden, bias=False)
        self.linear_query = nn.Linear(hidden, hidden, bias=False)

    def forward(self, inputs):
        vals = self.linear_val(inputs)
        keys = self.linear_key(inputs)
        queries = self.linear_query(inputs)

        vals = vals.view(-1, self.num_heads, self.hidden // self.num_heads).transpose(0, 1)
        keys = keys.view(-1, self.num_heads, self.hidden // self.num_heads).transpose(0, 1)
        queries = queries.view(-1, self.num_heads, self.hidden // self.num_heads).transpose(0, 1)

        queries /= np.sqrt(self.hidden / self.num_heads)
        
        return vals, keys, queries

class GraphAttention(nn.Module):
    def __init__(self, hidden, num_heads):
        super(GraphAttention, self).__init__()
        self.hidden = hidden
        self.num_heads = num_heads
        
        self.head = Head(num_heads, hidden)
        self.output_proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.norm1 = nn.BatchNorm1d(hidden)
        self.norm2 = nn.BatchNorm1d(hidden)
    
    def forward(self, inputs, graph):
        vals, keys, queries = self.head(inputs)
        
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / np.sqrt(self.hidden // self.num_heads)
        attention = F.softmax(scores, dim=-1)
        attended_nodes = torch.matmul(attention, vals)
        
        attended_nodes = attended_nodes.transpose(0, 1).contiguous().view(-1, self.hidden)
        output_projected = self.output_proj(attended_nodes)
        
        output_projected = self.norm1(output_projected + inputs.view(-1, self.hidden))
        normalized = self.norm2(self.mlp(output_projected) + output_projected)
        
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
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.final = nn.Linear(hidden, num_classes)
        self.gnn_attentions = nn.ModuleList([GraphAttention(hidden, num_heads) for _ in range(num_iters)])

    def forward(self, graph_tuple):
        nodes = graph_tuple["nodes"]
        if nodes[0] is not None:
            embs = torch.FloatTensor(nodes)
            embs = F.relu(self.linear(embs))
            embs = self.norm(self.mlp(embs) + embs)
        else:
            embs = self.embed(torch.arange(self.num_nodes))

        for gnn_attention in self.gnn_attentions:
            embs = gnn_attention(embs, graph_tuple)

        logits = self.final(embs)
        return logits

    def predict(self, logits):
        return F.softmax(logits, dim=1)

class AttentionModularityNet:
    def __init__(self, associative=True, epochs=150, learning_rate=0.01, num_heads=3, num_layers=2, hidden1=16*3,
                 emb_size=5, lam=0.5, bethe_hessian_init=False, verbose=False):
        self.associative = associative
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden1 = hidden1
        self.emb_size = emb_size
        self.lam = lam
        self.bethe_hessian_init = bethe_hessian_init
        self.verbose = verbose

    @staticmethod
    def graph2graph_tuple(graph_nx, node_features):
        return utils.graph2graph_tuple(graph_nx, node_features)

    def fit_transform(self, graph):
        num_communities = len(graph.graph['partition'])
        adjacency_matrix = nx.adjacency_matrix(graph)
        n = adjacency_matrix.shape[0]
        model = AttentionGNN(
            num_classes=num_communities,
            num_nodes=n,
            emb_size=self.emb_size,
            hidden=self.hidden1,
            num_heads=self.num_heads,
            num_iters=self.num_layers
        )

        if self.bethe_hessian_init:
            features = utils.bethe_hessian_node_features(adjacency_matrix, self.emb_size, self.associative)
        else:
            features = None

        graph_tuple = self.graph2graph_tuple(graph, features)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        model.train()
        for epoch in range(self.epochs):
            logits = model(graph_tuple)
            loss = F.cross_entropy(logits, torch.LongTensor(graph.graph['partition']))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.verbose and epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')

        logits = model(graph_tuple)
        proba = model.predict(logits)
        return self._fit_transform(graph, proba)

    def compute_bethe_hessian_features(self, adjacency_matrix):
        return utils.bethe_hessian_node_features(adjacency_matrix, self.emb_size, self.associative)

    def _fit_transform(self, graph, proba):
        # Implement your method for fitting and transforming the graph
        # using the predicted probabilities `proba`
        
        pass
