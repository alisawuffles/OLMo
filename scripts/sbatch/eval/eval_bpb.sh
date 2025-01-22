#!/bin/bash
#SBATCH --account=nvr_lacr_llm
#SBATCH --partition=batch,interactive
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --job-name=nvr_lacr_llm-eval.bpb
#SBATCH --output="slurm/eval/slurm-%J-%x.out"

cat $0
echo "--------------------"
date

max_context_length=512
echo "Evaluating bits per byte of $model_name at step $step"
python -m eval.eval_bpb \
    --model_name_or_path models/hf_models/$model_name \
    --max_num_examples 1000 \
    --max_context_length $max_context_length \
    --eval_batch_size 2 \
    --output_dir results/bpb-ctx${max_context_length}/$model_name/$step \
    ${step:+--step "$step"} \
    ${add_bos_token:+--add_bos_token}
