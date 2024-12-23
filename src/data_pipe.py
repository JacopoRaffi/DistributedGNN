from torchvision.datasets import CIFAR10
import torchvision.transforms as T
import torch
import time
import argparse
import torch_geometric
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
import csv
from torch_geometric.loader import DataLoader


from model import *
from data import CustomDataset, image_to_graph

#TODO: refactor code to make more elegant

global rank, device, pipe_group, ddp_group, stage_index, num_stages


def init_distributed():
    global rank, device, pipe_group, ddp_group, stage_index, num_stages
    rank = int(os.environ["RANK"])
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

def train(stage, criterion, optimizer, train_loader, val_loader, epoch, device, filename):
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

    
    stage.submod = DDP(stage.submod, process_group=ddp_group)
     
    train_schedule = ScheduleGPipe(stage, n_microbatches=n_microbatch, loss_fn=criterion)
    val_schedule = ScheduleGPipe(stage, n_microbatches=n_microbatch)

    # Log the training and validation (test) time
    print(f'RANK_{rank}_START_TRAINING')
    with open(filename, 'w+') as log_file: 
        csv_writer = csv.writer(log_file)
        header = ['epoch', 'batch', 'batch_time(s)', 'phase'] # Phase: 0 - train, 1 - val
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
                
                optimizer.step()
                
                end_batch_time = time.time()
                
                csv_row.append(end_batch_time - start_batch_time)
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
                    csv_row.append(1)
                    csv_writer.writerow(csv_row)

            log_file.flush()

def manual_split(data, n_microbatches=10, batch_size=20, n_classes=10):
    torch.manual_seed(42)
    model = PipeViGNN(8, 3, 3, 1024, n_classes, data.edge_index, 1024, batch_size//n_microbatches).to(device)
    indices = torch.arange(data.x.size(0) , dtype=torch.float32).view(-1, 1)
    data_x_with_index = torch.cat((data.x, indices), dim=1)  
    features_chunk = torch.chunk(data_x_with_index, n_microbatches, dim=0)[0]

    if (rank % 2) == 0: # First stage
        for i in range(4, 8):
            del model.blocks[str(i)]
            model.fc1 = None
            model.fc2 = None
            input_args = (features_chunk,)
            output_args = (features_chunk, )

    else: # Second stage
        out_chunk = torch.rand(batch_size//n_microbatches, n_classes, dtype=torch.float32)
        for i in range(0, 4):
            del model.blocks[str(i)]
            input_args = (features_chunk, )
            output_args = (out_chunk, )

    stage = PipelineStage(
            submodule=model,
            stage_index=(rank % 2),
            num_stages=num_stages,
            device=device,
            input_args=input_args,
            output_args=output_args,
            group=pipe_group
        )

    return stage

def get_num_parameters(model):
    params = [p.numel() for p in model.parameters()]
    total_params = sum(params)
    print(f'RANK_{rank}_params: {params}')

    return total_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', type=int, help='Length of the dataset to consider', default=0)
    args = parser.parse_args()

    init_distributed()

    device = 'cpu'
    batch_size = 1000
    n_microbatch = 5
    filename = f'../log/datapipe_{rank}_micro{n_microbatch}.csv'

    transform = T.ToTensor()
    train_dataset = CIFAR10(root='../data', train=True, download=False, transform=transform)
    test_dataset = CIFAR10(root='../data', train=False, download=False, transform=transform)

    if rank < 2:
        train_dataset = CustomDataset(image_to_graph(train_dataset), length=args.l, distributed=1)
        test_dataset = CustomDataset(image_to_graph(test_dataset), length=args.l, distributed=1)
    else:
        train_dataset = CustomDataset(image_to_graph(train_dataset), length=args.l, distributed=2)
        test_dataset = CustomDataset(image_to_graph(test_dataset), length=args.l, distributed=2)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    data = next(iter(train_loader))
    stage = manual_split(data, n_microbatches=n_microbatch, batch_size=batch_size, n_classes=10)

    optim = torch.optim.Adam(stage.submod.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    train(stage, criterion, optim, train_loader, test_loader, 4, device, filename)

    print(f'RANK_{rank}_DONE')
    dist.destroy_process_group(group=dist.group.WORLD)
