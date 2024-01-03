#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --partition=cpar
#SBATCH --exclusive
#SBATCH --constraint=k20

time nvprof ./bin/MDpar < inputdata.txt 