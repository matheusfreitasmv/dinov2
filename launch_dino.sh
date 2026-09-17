#!/bin/bash

# ================================================================
# CONFIGURAÇÃO DO SLURM
# ================================================================

#SBATCH --job-name=dinov2
#SBATCH --partition=high_performance
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=90:00:00
#SBATCH --output=result_train_%j.out

# ================================================================
# INÍCIO
# ================================================================

echo "================================================"
echo "INÍCIO DO JOB"
echo "================================================"

echo "Data/Hora: $(date)"
echo "Hostname: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs alocadas: $SLURM_CPUS_PER_TASK"

echo "================================"
echo "INFORMAÇÕES DO AMBIENTE"
echo "================================"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURMD_NODENAME=$SLURMD_NODENAME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

echo "================================================"

# 1. Garante um ambiente limpo de módulos
echo "Limpando módulos..."
module purge
module load python/3.10.13

export PYTHONPATH=/mnt/users/matheus.oliveira/python_packages:/mnt/users/matheus.oliveira/dinov2:$PYTHONPATH

cd /mnt/users/matheus.oliveira/dinov2

echo "Python:"
python --version

echo "================================"
echo "Iniciando DINOv2"
echo "================================"

# ================================================================
# EXECUÇÃO
# ================================================================

echo "Executando"

python dinov2/run/train/train.py \
    --config-file dinov2/configs/train/vits14_test.yaml \
    --output-dir ./dinov2/output \
    train.dataset_path=ImageNet:split=TRAIN:root=./dinov2/data/datasets/images/imagenette2:extra=./dinov2/data/datasets/images/metadata

# ================================================================
# FINALIZAÇÃO
# ================================================================

echo "================================================"
echo "FIM DO JOB"
echo "Data/Hora: $(date)"
echo "================================================"
