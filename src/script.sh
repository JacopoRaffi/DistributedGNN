#!/bin/bash

srun --nodelist=node01 singularity exec $MY_UBUNTU torchrun --nnodes=4 --nproc_per_node=1 --node_rank=0 --rdzv_endpoint=10.0.1.1:29500 pipe.py &
srun --nodelist=node02 singularity exec $MY_UBUNTU torchrun --nnodes=4 --nproc_per_node=1 --node_rank=1 --rdzv_endpoint=10.0.1.1:29500 pipe.py &
srun --nodelist=node03 singularity exec $MY_UBUNTU torchrun --nnodes=4 --nproc_per_node=1 --node_rank=2 --rdzv_endpoint=10.0.1.1:29500 pipe.py &
srun --nodelist=node04 singularity exec $MY_UBUNTU torchrun --nnodes=4 --nproc_per_node=1 --node_rank=3 --rdzv_endpoint=10.0.1.1:29500 pipe.py &
