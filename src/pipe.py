from torchvision.datasets import CIFAR10
import torchvision.transforms as T
import torch
import time
import argparse
import os
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, SplitPoint, PipelineStage, ScheduleGPipe
import csv
from torch_geometric.loader import DataLoader


from model import *
from data import CustomDataset, image_to_graph

#TODO: refactor code to make more elegant
#   change microbatch_size in n_microbatch
#   same for minibatch_size

global rank, device, pp_group, stage_index, num_stages
def init_distributed():
   global rank, device, pp_group, stage_index, num_stages
   rank = int(os.environ["RANK"])
   world_size = int(os.environ["WORLD_SIZE"])
   device = torch.device('cpu')
   dist.init_process_group()

   # This group can be a sub-group in the N-D parallel case
   pp_group = dist.new_group()
   stage_index = rank
   num_stages = world_size

#TODO: need 2 schedules per stage, 1 for training (with loss parameters) and 1 for validation (without loss parameters)
def train(schedule, stage, optimizer, train_loader, val_loader, epoch, device, filename):
    '''
    Train the model and compute the performance metrics

    Parameters:
    ----------
    schedule: PipelineScheduleSingle
        Schedule for the pipeline
    stage: PipelineStage
        Stage of the pipeline
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

    # Log the training and validation (test) time
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
                    schedule.step(data_x_with_index)
                else:
                    output = schedule.step(target=data.y)
                
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
                        schedule.step(data_x_with_index)
                    else:
                        output = schedule.step()
                    
                    end_batch_time = time.time()

                    csv_row.append(end_batch_time - start_batch_time)
                    csv_row.append(1)
                    csv_writer.writerow(csv_row)

            log_file.flush()

def manual_split(data, n_microbatches=10, batch_size=20, n_classes=10):
    model = PipeViGNN(8, 3, 3, 1024, n_classes, data.edge_index, 1024, batch_size//n_microbatches).to(device)
    indices = torch.arange(data.x.size(0) , dtype=torch.float32).view(-1, 1)
    data_x_with_index = torch.cat((data.x, indices), dim=1)  
    features_chunk = torch.chunk(data_x_with_index, n_microbatches, dim=0)[0]

    if stage_index == 0:
        for i in range(4, 8):
            del model.blocks[str(i)]
            model.fc1 = None
            model.fc2 = None
            input_args = (features_chunk,)
            output_args = (features_chunk, )

    else:
        out_chunk = torch.rand(batch_size//n_microbatches, n_classes, dtype=torch.float32)
        for i in range(0, 4):
            del model.blocks[str(i)]
            input_args = (features_chunk, )
            output_args = (out_chunk, )

    stage = PipelineStage(
            submodule=model,
            stage_index=stage_index,
            num_stages=num_stages,
            device=device,
            input_args=input_args,
            output_args=output_args,
        )
    
    return stage

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, help='Name of the log file', default=f'tmp.csv')
    parser.add_argument('-l', type=int, help='Length of the dataset to consider', default=60)
    args = parser.parse_args()

    init_distributed()

    device = 'cpu'
    batch_size = 20
    n_microbatch = 4

    transform = T.ToTensor()
    train_dataset = CIFAR10(root='../data', train=True, download=False, transform=transform)
    train_dataset = CustomDataset(image_to_graph(train_dataset), length=args.l)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = CIFAR10(root='../data', train=False, download=False, transform=transform)
    test_dataset = CustomDataset(image_to_graph(test_dataset), length=args.l)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    data = next(iter(train_loader))
    stage = manual_split(data, n_microbatches=n_microbatch, batch_size=batch_size, n_classes=10)

    criterion = torch.nn.CrossEntropyLoss()
    schedule = ScheduleGPipe(stage, n_microbatches=n_microbatch, loss_fn=criterion)
    optim = torch.optim.Adam(stage.submod.parameters(), lr=0.001)

    train(schedule, stage, optim, train_loader, train_loader, 2, device, args.filename)

    '''if stage_index == 0:
        optimizer = torch.optim.Adam(stage.submod.parameters(), lr=0.001)
        indices = torch.arange(data.x.size(0) , dtype=torch.float32).view(-1, 1)
        optimizer.zero_grad()
        data_x_with_index = torch.cat((data.x, indices), dim=1)  
        schedule.step(data_x_with_index)
        optimizer.step()
    else:
        optimizer = torch.optim.Adam(stage.submod.parameters(), lr=0.001)
        optimizer.zero_grad()
        out = schedule.step(target=data.y)
        optimizer.step()'''

    print(f'RANK_{stage_index}_DONE')
    dist.destroy_process_group()
