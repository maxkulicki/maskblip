#!/bin/bash

module purge
module load 2022
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
module load Python/3.10.4-GCCcore-11.3.0
source activate maskblip
