#!/bin/bash



module load papi/5.4.1 
#./MDseq.exe <inputdata.txt
export OMP_NUM_THREADS=2
time ./MDpar.exe < inputdata.txt 
export OMP_NUM_THREADS=4
time ./MDpar.exe < inputdata.txt  
export OMP_NUM_THREADS=8
time ./MDpar.exe < inputdata.txt   
