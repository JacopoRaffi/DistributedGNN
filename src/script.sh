#!/bin/bash

export GLOO_SOCKET_IFNAME=ib0

srun --nodelist=node11 singularity exec $MY_UBUNTU torchrun --nnodes=4 --nproc_per_node=1 --node_rank=0 --rdzv_endpoint=10.0.1.11:29500 data_pipe.py &
srun --nodelist=node12 singularity exec $MY_UBUNTU torchrun --nnodes=4 --nproc_per_node=1 --node_rank=1 --rdzv_endpoint=10.0.1.11:29500 data_pipe.py &
srun --nodelist=node13 singularity exec $MY_UBUNTU torchrun --nnodes=4 --nproc_per_node=1 --node_rank=2 --rdzv_endpoint=10.0.1.11:29500 data_pipe.py &
srun --nodelist=node14 singularity exec $MY_UBUNTU torchrun --nnodes=4 --nproc_per_node=1 --node_rank=3 --rdzv_endpoint=10.0.1.11:29500 data_pipe.py &
