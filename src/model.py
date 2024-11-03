import torch
import torch.nn as nn
import os
from torch_geometric.nn.sequential import Sequential as Seq
import torch.nn.functional as F
from torch_geometric.utils import subgraph
from torch_geometric.nn import GraphConv, global_add_pool


class FFN(nn.Module):
    def __init__(self, in_size, hidden_size=None):
        super(FFN, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(in_size, 512),
            nn.BatchNorm1d(512),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, in_size),
            nn.BatchNorm1d(in_size),
        )

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x += shortcut

        return x
    
class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_size):
        super(Grapher, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(in_size, 256),
            nn.BatchNorm1d(256),
        )
        self.graph_conv = GraphConv(256, 256)

        self.fc2 = nn.Sequential(
            nn.Linear(256, in_size),
            nn.BatchNorm1d(in_size),
        )

    def forward(self, x, edge_index):
        _tmp = x
        x = self.fc1(x)
        x = F.gelu(self.graph_conv(x, edge_index))
        x = self.fc2(x)
        x += _tmp

        return x
    
class ViGBlock(nn.Module):
    def __init__(self, in_channels):
        super(ViGBlock, self).__init__()
        
        self.grapher = Grapher(in_channels)
        self.ffn = FFN(in_channels, in_channels * 4)
        
    def forward(self, x, edge_index):
        x = self.grapher(x, edge_index)
        x = self.ffn(x)
        
        return x
    
class ViGNN(nn.Module):
    def __init__(self, n_blocks, channels, embedding_size, hidden_size, n_classes=10):
        super().__init__()

        self.blocks = torch.nn.ModuleDict()
        for layer_id in range(n_blocks):
            self.blocks[str(layer_id)] = ViGBlock(channels)

        self.fc1 = nn.Linear(embedding_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_classes)
        
    def forward(self, x, edge_index, batch=None):
        for vig in self.blocks.values():
            x = vig.forward(x, edge_index)

        x = global_add_pool(x, batch)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)

        return x
    
class PipeViGNN(nn.Module):
    def __init__(self, n_blocks, channels, embedding_size, hidden_size, n_classes, edges):
        super().__init__()

        self.edges = edges
        self.blocks = torch.nn.ModuleDict()
        for layer_id in range(n_blocks):
            self.blocks[str(layer_id)] = ViGBlock(channels)

        self.fc1 = nn.Linear(embedding_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_classes)
        
    def forward(self, x, batch=None):
        features = x[:, :-1]
        indices = x[..., -1]
        sub_edge_index, _ = subgraph(indices.long(), self.edges, relabel_nodes=True)
        for vig in self.blocks.values():
            features = vig.forward(features, sub_edge_index)

        features = torch.cat((features, indices.view(-1, 1)), dim=1)  

        if self.fc1:
            features = global_add_pool(features[:, :-1], batch)
            features = F.gelu(self.fc1(features))
            features = self.fc2(features)

        return features