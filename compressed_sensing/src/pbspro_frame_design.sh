#!/bin/bash
### Job Name
#PBS -N frame
#PBS -l walltime=399:00:00
#PBS -q workq
### Select 1 node with 20 CPUs
#PBS -l select=1:ncpus=1:ngpus=1:host=klab-luke
#PBS -e Errors.err
#PBS -o Output.out

source ~/.bashrc
conda activate Pytorch
cd $PBS_O_WORKDIR
python main_frame_design.py