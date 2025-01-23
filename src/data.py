import torch_geometric
from torch_geometric.utils import grid, remove_self_loops

def image_to_graph(image_dataset, image_size = (32,32)):
    '''
    Transforms an image dataset into a graph dataset.

    Parameters:
    ----------
    model: image_dataset
        The dataset to be transformed.
    image_size: tuple
        The size of the images in the dataset.
        
    Returns:
    -------
    return: list
        A list of torch_geometric.data.Data objects containing the graph representation of the images.
    '''

    edge_index, node_pos = grid(image_size[0], image_size[1]) # HxW grid presented as a graph
    edge_index, _ = remove_self_loops(edge_index)

    dataset = []
    for im, label in image_dataset:
        im = im.permute(1, 2, 0) # permute so to have the channels as the last dimension
        rgb_values = im[node_pos[:, 0].long(), node_pos[:, 1].long()]  # each pixel is a node of the graph

        dataset.append(torch_geometric.data.Data(x=rgb_values, edge_index=edge_index, y=label))
    
    return dataset


class CustomDataset(torch_geometric.data.Dataset):
    def __init__(self, dataset:list, distributed: int = 0):
        '''
        Initializes the data handler.
        Parameters:
        dataset: list
            The dataset to be handled.
        distributed: int
            If 0, the dataset is not distributed. If 1, the first half of the dataset is used. If 2, the second half of the dataset is used
            (Applied only in a data parallel distributed setting with 2 copies).
        '''
        self.dist = distributed

        self.dataset = dataset

        midpoint = len(self.dataset) // 2

        if self.dist: # split the dataset in half if in a data parallel distributed setting
            if self.dist == 1:
                self.dataset = self.dataset[:midpoint]
            else:
                self.dataset = self.dataset[midpoint:]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    
