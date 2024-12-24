#!/bin/bash
#SBATCH --account=nvr_lacr_llm
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=nvr_lacr_llm-pretrain.olmo2-7b
#SBATCH --output="slurm/train/slurm-%J-%x.out"

cat $0
echo "--------------------"

export MASTER_PORT=$(shuf -i 49152-65535 -n 1)
echo "MASTER_PORT="$MASTER_PORT
echo "NODELIST="${SLURM_NODELIST}
echo "SLURM_NTASKS="$SLURM_NTASKS
echo "SLURM_PROCID="$SLURM_PROCID

config=configs/alisa/OLMo2-7B-neox.yaml
wandb_run_id=o862zead
srun --ntasks=$SLURM_NTASKS --ntasks-per-node=8 \
    python scripts/train.py $config \
        --save_overwrite \
        --try_load_latest_save \
        --wandb.resume=must \
        --wandb.run_id=$wandb_run_id
