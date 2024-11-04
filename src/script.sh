#!/bin/bash

srun --nodelist=node01 singularity exec $MY_UBUNTU torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 --rdzv_endpoint=10.0.1.1:29500 pipe.py --filename=pipe0.csv &
srun --nodelist=node02 singularity exec $MY_UBUNTU torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 --rdzv_endpoint=10.0.1.1:29500 pipe.py --filename=pipe1.csv &
