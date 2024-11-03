#TODO: leggiti il paper su come fare GPipe con PyG, per i primi stage evita di passare anche il parametro batch
#TODO: controllare la divisione in microbatch del parametro batch nell'ultimo stage

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

global rank, device, pp_group, stage_index, num_stages
def init_distributed():
   global rank, device, pp_group, stage_index, num_stages
   rank = int(os.environ["LOCAL_RANK"])
   world_size = int(os.environ["WORLD_SIZE"])
   device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
   dist.init_process_group()

   # This group can be a sub-group in the N-D parallel case
   pp_group = dist.new_group()
   stage_index = rank
   num_stages = world_size


def manual_split(model, data):
    if stage_index == 0:
        for i in range(4, 8):
            del model.blocks[str(i)]
            model.fc1 = None

        stage = PipelineStage(
            submodule=model,
            stage_index=stage_index,
            num_stages=num_stages,
            device='cpu',
            input_args=(data.x, data.edge_index),
            output_args=(data.x, data.edge_index),
        )

        return stage

    else:
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, help='Name of the log file', default='tmp.csv')
    parser.add_argument('-l', type=int, help='Length of the dataset to consider', default=100)
    args = parser.parse_args()

    init_distributed()

    device = 'cpu'

    gnn = ViGNN(8, 3, 3, 1024, 10).to(device)

    optimizer = torch.optim.Adam(gnn.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    transform = T.ToTensor()
    train_dataset = CIFAR10(root='../data', train=True, download=False, transform=transform)
    train_dataset = CustomDataset(image_to_graph(train_dataset), length=args.l)
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)

    data = next(iter(train_loader))
    stage = manual_split(gnn, data)

    print(stage)

    if stage_index == 0:
        schedule = ScheduleGPipe(stage, n_microbatches=10, loss_fn=criterion)
        schedule.step(data.x, data.edge_index, data.batch)
