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
    def __init__(self, dataset, length=100):
        if length > 0:
            self.dataset = dataset[:length]
        else:
            self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]