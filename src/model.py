import torch
import torch.nn as nn
import os
from torch_geometric.nn.sequential import Sequential as Seq
import torch.nn.functional as F
from torch_geometric.utils import subgraph
from torch_geometric.nn import GraphConv, global_add_pool


class FFN(nn.Module):
    '''
    FFN Module with 2 linear layers and batch normalization

    Attributes:
    -----------
    fc1: nn.Sequential
        First linear layer with batch normalization
    fc2: nn.Sequential
        Second linear layer with batch normalization
    '''
    def __init__(self, in_size, hidden_size=512):
        '''
        Initializes the FFN module
        
        Parameters:
        ----------
        in_size: int
            Number of input channels
        hidden_size: int
            Size of the hidden layer
        '''
        super(FFN, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, in_size),
            nn.BatchNorm1d(in_size),
        )

    def forward(self, x:torch.Tensor):
        shortcut = x
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x += shortcut

        return x
    
class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers

    Attributes:
    fc1: nn.Sequential
        First linear layer with batch normalization
    graph_conv: GraphConv
        Graph convolution layer
    fc2: nn.Sequential
        Second linear layer with batch normalization
    """
    def __init__(self, in_size:int, hidden_size:int=256):
        '''
        Initializes the Grapher module

        Parameters:
        ----------
        in_size: int
            Number of input channels
        hidden_size: int
            Size of the hidden layer
        '''
        super(Grapher, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
        )
        self.graph_conv = GraphConv(hidden_size, hidden_size)

        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, in_size),
            nn.BatchNorm1d(in_size),
        )

    def forward(self, x:torch.Tensor, edge_index:torch.Tensor):
        _tmp = x
        x = self.fc1(x)
        x = F.gelu(self.graph_conv(x, edge_index))
        x = self.fc2(x)
        x += _tmp

        return x
    
class ViGBlock(nn.Module):
    '''
    ViGBlock module with Grapher and FFN

    Attributes:
    -----------
    grapher: Grapher
        Grapher module
    ffn: FFN
        FFN module
    '''
    def __init__(self, in_channels:int, ffn_embedding_size:int=512, grapher_embedding_size:int=256):
        '''
        Initializes the ViGBlock module
        
        Parameters:
        in_channels: int
            Number of input channels
        ffn_embedding_size: int
            Size of the embedding of the FFN module
        grapher_embedding_size: int
            Size of the embedding of the Grapher module
        '''
        super(ViGBlock, self).__init__()
        
        self.grapher = Grapher(in_channels, grapher_embedding_size)
        self.ffn = FFN(in_channels, ffn_embedding_size)
        
    def forward(self, x:torch.Tensor, edge_index:torch.Tensor):
        x = self.grapher(x, edge_index)
        x = self.ffn(x)
        
        return x
    
class ViGNN(nn.Module):
    '''
    ViGNN module with multiple ViGBlock and linear layers

    Attributes:
    -----------
    blocks: torch.nn.ModuleDict
        Dictionary of ViGBlock modules
    '''
    def __init__(self, n_blocks:int, channels:int, embedding_size:tuple, hidden_readout_size:int, n_classes=10):
        '''
        Initializes the ViGNN module

        Parameters:
        ----------
        n_blocks: int
            Number of ViGBlock modules
        channels: int
            Number of input channels
        embedding_size: tuple
            Tuple of the embedding sizes for the ViGBlock modules (pos 0 - FFN, pos 1 - Grapher)
        hidden_size: int
            Size of the hidden layer of the readout
        n_classes: int
            Number of classes (output layer size)
        '''
        super().__init__()

        self.blocks = torch.nn.ModuleDict()
        for layer_id in range(n_blocks):
            self.blocks[str(layer_id)] = ViGBlock(channels, embedding_size[0], embedding_size[1])

        self.fc1 = nn.Linear(embedding_size[0], hidden_readout_size)
        self.fc2 = nn.Linear(hidden_readout_size, n_classes)
        
    def forward(self, x:torch.Tensor, edge_index:torch.Tensor, batch:torch.Tensor=None):
        for vig in self.blocks.values():
            x = vig.forward(x, edge_index)

        x = global_add_pool(x, batch)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)

        return x
    
class PipeViGNN(ViGNN):
    def __init__(self, *args, edges:torch.Tensor, graph_nodes:int, microbatch_size:int, **kwargs):
        '''
        Manual Pipeline implementation of the ViGNN model

        Parameters:
        ----------
        edges: torch.tensor
            Tensor of edges
        graph_nodes: int
            Number of nodes in a single graph (assume al graphs have the same number of nodes)
        microbatch_size: int
            Number of graphs in a single microbatch  
        '''
        super().__init__(*args, **kwargs)

        self.edges = edges
        self.graph_nodes = graph_nodes
        self.microbatch_size = microbatch_size
        self.batch = torch.tensor([i for i in range(microbatch_size) for _ in range(graph_nodes)])
        
    def forward(self, x:torch.Tensor):
        features = x[:, :-1]  # Separate features and indices
        indices = x[..., -1]
        
        # Generate subgraph edges for the batch
        sub_edge_index, _ = subgraph(indices.long(), self.edges, relabel_nodes=True)
        
       
        for vig in self.blocks.values():
            features = vig.forward(features, sub_edge_index)
        
        # Combine features with indices
        features = torch.cat((features, indices.view(-1, 1)), dim=1)
        
        # Apply readout and fully connected layers if defined
        if self.fc1:
            features = global_add_pool(features[:, :-1], batch=self.batch)
            features = F.gelu(self.fc1(features))
            features = self.fc2(features)
        
        return features