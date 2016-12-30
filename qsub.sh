#!/bin/sh
#PBS -N sparsemax-benchmark
#PBS -l walltime=02:00:00
#PBS -l nodes=1:ppn=4:gpus=1
#PBS -m eba
#PBS -M amwebdk@gmail.com
#PBS -q k40_interactive

cd $PBS_O_WORKDIR

# Enable python3
export PYTHONPATH=
source ~/stdpy3/bin/activate

make clean
make build
make benchmark

echo "=== DONE ==="
