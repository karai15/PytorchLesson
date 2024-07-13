#!/bin/bash
### Job Name
#PBS -N CE
#PBS -l walltime=399:00:00
#PBS -q workq
### Select 1 node with 20 CPUs
#PBS -l select=1:ncpus=112:ngpus=0:host=klab-vador
#PBS -e Errors.err
#PBS -o Output.out

source ~/.bashrc
conda activate Numpy3.9
cd $PBS_O_WORKDIR
python main_CS.py