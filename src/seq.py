from torchvision.datasets import CIFAR10
import torchvision.transforms as T
import torch
import time
import argparse
import csv
from torch_geometric.loader import DataLoader

from model import *
from data import CustomDataset, image_to_graph

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)

def train(model, optimizer:torch.optim, criterion:torch.nn, 
          train_loader:torch_geometric.loader, val_loader:torch_geometric.loader, 
          epoch:int, device:str, filename:str):
    '''
    Train the model and compute the performance metrics

    Parameters:
    ----------
    model: torch.nn.Module
        Model to train
    optimizer: torch.optim.Optimizer
        Optimizer to use
    criterion: torch.nn.Module
        Loss function
    train_loader: torch_geometric.loader.DataLoader
        DataLoader for training
    val_loader: torch_geometric.loader.DataLoader
        DataLoader for validation (test)
    epoch: int
        Number of epochs
    device: torch.device
        Device to use
    filename: str
        Name of the log file where to store time metrics
        

    Returns:
    -------
    return: None
    '''

    # Log the training and validation (test) time
    with open(filename, 'w+') as log_file: 
        csv_writer = csv.writer(log_file)
        header = ['epoch', 'batch', 'batch_time(s)', 'loss', 'phase'] # Phase: 0 - train, 1 - val
        csv_writer.writerow(header)
        log_file.flush()

        for epoch in range(epoch):
            model.train()
            total_loss = 0
            for i, data in enumerate(train_loader):
                csv_row = []
                csv_row.append(epoch)
                csv_row.append(i)
                
                start_batch_time = time.time()
                
                data = data.to(device)
                optimizer.zero_grad()
                output = model(data.x, data.edge_index, data.batch)
                loss = criterion(output, data.y)
                loss.backward()

                optimizer.step()
                total_loss += loss.item()
                
                end_batch_time = time.time()
                
                csv_row.append(end_batch_time - start_batch_time)
                csv_row.append(loss.item())
                csv_row.append(0) # Phase 0 - train
                # The row contains the epoch_id, the batch_id, the time spent in the batch, the loss and the phase
                csv_writer.writerow(csv_row)
                print(f'Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}')

            model.eval()
            with torch.no_grad():
                total_loss = 0
                for i, data in enumerate(val_loader):
                    csv_row = []
                    csv_row.append(epoch)
                    csv_row.append(i)
                    
                    start_batch_time = time.time()
                    
                    data = data.to(device)
                    output = model(data.x, data.edge_index, data.batch)
                    loss = criterion(output, data.y)
                    total_loss += loss.item()
                    
                    end_batch_time = time.time()

                    csv_row.append(end_batch_time - start_batch_time)
                    csv_row.append(loss.item())
                    csv_row.append(1) # Phase 1 - val
                    csv_writer.writerow(csv_row)

            log_file.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, help='Name of the log file', default='loss.csv')
    args = parser.parse_args()

    device = 'cpu'
    batch_size = 1000
    epoch = 10

    gnn = ViGNN(8, 3, 3, 1024, 10).to(device)

    optimizer = torch.optim.Adam(gnn.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    transform = T.ToTensor()

    train_dataset = CIFAR10(root='../data', train=True, download=False, transform=transform)
    test_dataset = CIFAR10(root='../data', train=False, download=False, transform=transform)

    train_dataset = CustomDataset(image_to_graph(train_dataset), length=args.l)
    test_dataset = CustomDataset(image_to_graph(test_dataset), length=args.l)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train(gnn, optimizer, criterion, train_loader, val_loader, epoch, device, args.filename)