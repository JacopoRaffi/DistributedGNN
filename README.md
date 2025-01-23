# DistributedGNN

## Introduction
DistrGNN is a research project supported by a scholarship from the University of Pisa. It focuses on investigating and implementing pipeline parallelism in Graph Neural Networks (GNNs). The study case is based on the methodology outlined in the paper **[Vision GNN: An Image is Worth Graph of Nodes](https://proceedings.neurips.cc/paper_files/paper/2022/file/3743e69c8e47eb2e6d3afaea80e439fb-Paper-Conference.pdf)** by Han et al. (2022), presented at *Advances in Neural Information Processing Systems (NeurIPS)*. The project explores the efficient distribution of GNN computations across multiple computing nodes, with a particular emphasis on pipeline parallelism. Additionally, it examines the combination of **pipeline parallelism** and **data parallelism** to optimize performance and scalability in large-scale GNN training tasks.

More details on this study, including technical insights and experimental results, are available in the project's report **report.pdf**.

## Repository Structure
The project code is located in the `src` directory. Here's a breakdown of the key files:

* `model.py`: The file contains the implementation of the model (both the sequential and the "pipeline" version).
* `seq.py`: Script for running the sequential GNN model.
* `pipe.py`: Script for running the GNN model with pipeline parallelism.
* `data_pipe.py`: Script for running the GNN model with combined data and pipeline parallelism.
* `report.pdf`: Project report with detailed technical insights and experimental results.

## Installation
1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/JacopoRaffi/DistributedGNN.git
   cd DistriutedGNN
   ```
2. Install all the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Usage
### Sequential
Execution of the sequential model
```bash
   cd src
   python3 seq.py --filename log_file.csv
```
### Pipeline Parallelism
Example of execution of a 2-stage pipeline
```bash
   cd src
   torchrun --nproc_per_node=1 --nnodes=2 pipe.py --filename log_file.csv
```
### Data + Pipeline Parallelism
Executing the combination wit two copies and each copies splitted in a 2-stage Pipeline
```bash
   cd src
   torchrun --nproc_per_node=1 --nnodes=4 data_pipe.py --filename log_file.csv
```
## Acknowledgements
I would like to thank Prof. Patrizio Dazzi and the University of Pisa for this opportunity.
