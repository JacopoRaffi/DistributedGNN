from torchvision.datasets import CIFAR10
import torchvision.transforms as T
import torch
import time
import random
import argparse
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
import csv
from torch_geometric.loader import DataLoader


from model import *
from pipe import manual_split
from data import CustomDataset, image_to_graph

global rank, device, pipe_group, ddp_group, stage_index, num_stages

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)

def init_distributed():
    global rank, device, pipe_group, ddp_group, stage_index, num_stages
    rank = int(os.environ["RANK"])
    os.environ["GLOO_SOCKET_IFNAME"] = "ib0"
    #world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device('cpu')
    dist.init_process_group()

    # Define pipeline groups
    pipe_group_a_ranks = [0, 1]  # First copy of the model
    pipe_group_b_ranks = [2, 3]  # Second copy of the model

    # Define DDP groups
    ddp_group_c_ranks = [0, 2]  # First stage
    ddp_group_d_ranks = [1, 3]  # Second stage

    # All processes must create all groups
    pipe_group_a = dist.new_group(ranks=pipe_group_a_ranks)
    pipe_group_b = dist.new_group(ranks=pipe_group_b_ranks)
    ddp_group_c = dist.new_group(ranks=ddp_group_c_ranks)
    ddp_group_d = dist.new_group(ranks=ddp_group_d_ranks)

    # Determine pipeline group
    if rank in pipe_group_a_ranks:
        pipe_group = pipe_group_a
    elif rank in pipe_group_b_ranks:
        pipe_group = pipe_group_b

    # Determine DDP group
    if rank in ddp_group_c_ranks:
        ddp_group = ddp_group_c
    elif rank in ddp_group_d_ranks:
        ddp_group = ddp_group_d
    
    stage_index = pipe_group.rank()
    num_stages = pipe_group.size()

def train(stage:PipelineStage, criterion:torch.nn, optimizer:torch.optim, 
          train_loader:torch_geometric.loader.DataLoader, val_loader:torch_geometric.loader.DataLoader, 
          epoch:int, device:str, filename:str):
    '''
    Train the model and compute the performance metrics

    Parameters:
    ----------
    stage: PipelineStage
        Stage of the pipeline
    criterion: torch.nn._WeightedLoss
        Loss function to use
    optimizer: torch.optim.Optimizer
        Optimizer to use
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

    
    stage.submod = DDP(stage.submod, process_group=ddp_group) # Wrap the model with DDP to synchronize the gradients
     
    train_schedule = ScheduleGPipe(stage, n_microbatches=n_microbatch, loss_fn=criterion)
    val_schedule = ScheduleGPipe(stage, n_microbatches=n_microbatch)

    # Log the training and validation (test) time
    print(f'RANK_{rank}_START_TRAINING')
    with open(filename, 'w+') as log_file: 
        csv_writer = csv.writer(log_file)
        header = ['epoch', 'batch', 'batch_time(s)', 'loss', 'phase'] # Phase: 0 - train, 1 - val
        csv_writer.writerow(header)
        log_file.flush()

        for epoch in range(epoch):
            stage.submod.train()
            for i, data in enumerate(train_loader):
                csv_row = []
                csv_row.append(epoch)
                csv_row.append(i)
                
                start_batch_time = time.time()

                data = data.to(device)
                optimizer.zero_grad()

                if stage_index == 0:
                    indices = torch.arange(data.x.size(0) , dtype=torch.float32).view(-1, 1)
                    data_x_with_index = torch.cat((data.x, indices), dim=1)  
                    train_schedule.step(data_x_with_index)
                else:
                    output = train_schedule.step(target=data.y)
                    loss = criterion(output, data.y)
                
                optimizer.step()
                
                end_batch_time = time.time()
                
                csv_row.append(end_batch_time - start_batch_time)
                if stage_index == 0:
                    csv_row.append(-1)
                else:
                    csv_row.append(loss.item())
                csv_row.append(0)
                csv_writer.writerow(csv_row) # The row contains the epoch_id, the batch_id, the time spent in the batch and the phase (0 - train, 1 - val)

            stage.submod.eval()
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    csv_row = []
                    csv_row.append(epoch)
                    csv_row.append(i)
                    
                    start_batch_time = time.time()
                    
                    data = data.to(device)
                    if stage_index == 0:
                        indices = torch.arange(data.x.size(0) , dtype=torch.float32).view(-1, 1)
                        data_x_with_index = torch.cat((data.x, indices), dim=1)  
                        val_schedule.step(data_x_with_index)
                    else:
                        output = val_schedule.step()
                        loss = criterion(output, data.y)

                    
                    end_batch_time = time.time()

                    csv_row.append(end_batch_time - start_batch_time)
                    if stage_index == 0:
                        csv_row.append(-1)
                    else:
                        csv_row.append(loss.item())
                    csv_row.append(1)
                    csv_writer.writerow(csv_row)

            log_file.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, help='Name of the log file', default='loss.csv')
    args = parser.parse_args()

    init_distributed()

    device = 'cpu'
    batch_size = 1000
    n_microbatch = 5
    epoch = 10

    transform = T.ToTensor()
    train_dataset = CIFAR10(root='../data', train=True, download=False, transform=transform)
    test_dataset = CIFAR10(root='../data', train=False, download=False, transform=transform)

    # Shuffle the *indices* before creating the CustomDataset so to have "balanced" classes in each copy
    train_indices = list(range(len(train_dataset)))
    random.shuffle(train_indices)

    test_indices = list(range(len(test_dataset)))
    random.shuffle(test_indices)

    if rank < 2: # First copy
        train_dataset = CustomDataset(image_to_graph(torch.utils.data.Subset(train_dataset, train_indices)), distributed=1)
        test_dataset = CustomDataset(image_to_graph(test_dataset), distributed=1)
    else: # Second copy
        train_dataset = CustomDataset(image_to_graph(torch.utils.data.Subset(train_dataset, train_indices)), distributed=2)
        test_dataset = CustomDataset(image_to_graph(test_dataset), distributed=2)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    data = next(iter(train_loader))
    stage = manual_split(data, n_microbatches=n_microbatch, batch_size=batch_size, n_classes=10)

    optim = torch.optim.Adam(stage.submod.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    train(stage, criterion, optim, train_loader, test_loader, epoch, device, args.filename)

    print(f'RANK_{rank}_DONE')
    dist.destroy_process_group(group=dist.group.WORLD)
