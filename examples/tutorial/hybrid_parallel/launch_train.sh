#!/bin/bash
#SBATCH --job-name=colo            # 作业名
#SBATCH --partition=AI4GOV         # 队列（分区）
#SBATCH --nodes=1                   # 使用的节点数量
#SBATCH --ntasks-per-node=4         # 每个节点的任务数
#SBATCH --cpus-per-task=6           # 每个任务的CPU核心数
#SBATCH --gres=gpu:4                # 每个节点的GPU数量
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

export OMP_NUM_THREADS=1

MASTER_ADDR="127.0.0.1"
MASTER_PORT=29500

echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"

torchrun --nnodes=$SLURM_NNODES --nproc_per_node=$SLURM_NTASKS_PER_NODE --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT train.py