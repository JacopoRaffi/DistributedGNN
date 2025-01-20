#!/bin/bash

export GLOO_SOCKET_IFNAME=ib0

srun --nodelist=node25 singularity exec $MY_UBUNTU torchrun --nnodes=4 --nproc_per_node=1 --node_rank=0 --rdzv_endpoint=10.0.1.25:29500 data_pipe.py &
srun --nodelist=node26 singularity exec $MY_UBUNTU torchrun --nnodes=4 --nproc_per_node=1 --node_rank=1 --rdzv_endpoint=10.0.1.25:29500 data_pipe.py &
srun --nodelist=node27 singularity exec $MY_UBUNTU torchrun --nnodes=4 --nproc_per_node=1 --node_rank=2 --rdzv_endpoint=10.0.1.25:29500 data_pipe.py &
srun --nodelist=node28 singularity exec $MY_UBUNTU torchrun --nnodes=4 --nproc_per_node=1 --node_rank=3 --rdzv_endpoint=10.0.1.25:29500 data_pipe.py &
