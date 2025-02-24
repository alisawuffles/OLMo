#!/bin/bash
#SBATCH --account=nvr_lacr_llm
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --job-name=nvr_lacr_llm-eval.bpb
#SBATCH --output="slurm/eval/slurm-%J-%x.out"

cat $0
echo "--------------------"
date

echo "Evaluating bits per byte of $model_name at step $step"
python -m eval.eval_bpb \
    --model_name_or_path models/hf_models/$model_name \
    --output_dir $output_dir \
    --eval_batch_size $eval_batch_size \
    ${max_num_examples:+--max_num_examples "$max_num_examples"} \
    ${eval_batch_size:+--eval_batch_size "$eval_batch_size"} \
    ${step:+--step "$step"}
