#!/bin/bash

#SBATCH --constraint='sirocco'
#SBATCH --job-name=unknown
#SBATCH --exclude=sirocco[01-05,16,17,21,22]

nodes=$( scontrol show hostnames $SLURM_JOB_NODELIST )
if [[ $SLURM_JOB_NUM_NODES -gt 1 ]] ; then
    nodes_array=($nodes)

    head_node=${nodes_array[0]}

    echo Head node : $head_node
    options="--rdzv-id $RANDOM --rdzv-backend c10d --rdzv-endpoint $head_node:26501"
fi

echo Options : ${options}
echo Using ${SLURM_JOB_NUM_NODES} nodes \($nodes\) and every GPU on every node.

srun singularity exec --nv /beegfs/aaguilam/images/nanotron.sif \
torchrun --nnodes ${SLURM_JOB_NUM_NODES} --nproc-per-node=2 ${options} $1 --filename $2 --learning_rate $3 --batch_size $4 --epochs $5 --num_classes $6 --noise_type $7 --load $8 --save $9 --model $10
