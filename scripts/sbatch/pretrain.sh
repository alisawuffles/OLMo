#!/bin/bash
#SBATCH --account=nvr_lacr_llm
#SBATCH --partition=backfill
#SBATCH --time=04:00:00
#SBATCH --nodes=4
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
srun --ntasks=$SLURM_NTASKS --ntasks-per-node=8 \
    python scripts/train.py $config \
        --save_overwrite  \
        --save_interval_ephemeral=50 \
        --save_folder=/lustre/fsw/nvr_lacr_llm/aliliu/olmo/models/OLMo2-7B-neox-4nodes

# nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
# nodes_array=($nodes)
# head_node=${nodes_array[0]}
# head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" $(which hostname) --ip-address)
# echo nodes: ${nodes}
# echo head node: ${head_node} at ${head_node_ip}
# srun --export=all torchrun \
#     --nnodes=$SLURM_NNODES \
#     --nproc_per_node=$SLURM_NTASKS_PER_NODE \
#     --start_method=fork \
#     --rdzv_id=$RANDOM \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint=${head_node_ip}:29500 \
#     scripts/train.py $config \
#         --save_overwrite  \
#         --save_interval_ephemeral=50 \
#         --save_folder=/lustre/fsw/nvr_lacr_llm/aliliu/olmo/models/OLMo2-7B-neox-4nodes
