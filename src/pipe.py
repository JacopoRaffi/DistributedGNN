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


def manual_split(data, n_microbatches=10, minibatch_size=20, n_classes=10):
    model = PipeViGNN(8, 3, 3, 1024, n_classes, data.edge_index, 1024, minibatch_size//n_microbatches).to(device)
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
        out_chunk = torch.rand(minibatch_size//n_microbatches, n_classes, dtype=torch.float32)
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
    parser.add_argument('--filename', type=str, help='Name of the log file', default='tmp.csv')
    parser.add_argument('-l', type=int, help='Length of the dataset to consider', default=0)
    args = parser.parse_args()

    init_distributed()

    device = 'cpu'

    criterion = torch.nn.CrossEntropyLoss()

    transform = T.ToTensor()
    train_dataset = CIFAR10(root='../data', train=True, download=False, transform=transform)
    train_dataset = CustomDataset(image_to_graph(train_dataset), length=args.l)
    train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)

    data = next(iter(train_loader))
    stage = manual_split(data, n_microbatches=10, minibatch_size=1000, n_classes=10)

    schedule = ScheduleGPipe(stage, n_microbatches=10, loss_fn=criterion)
    indices = torch.arange(data.x.size(0), dtype=torch.float32).view(-1, 1)  # Reshape to a column vector

    if stage_index == 0:
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
        optimizer.step()

    print("DONE")
    dist.destroy_process_group()
