#!/bin/bash



module load papi/5.4.1 
#./MDseq.exe <inputdata.txt
export OMP_NUM_THREADS=2
./MDpar.exe < inputdata.txt 
export OMP_NUM_THREADS=4
./MDpar.exe < inputdata.txt  
export OMP_NUM_THREADS=8
./MDpar.exe < inputdata.txt   
