#!/bin/bash
#SBATCH --account=nvr_lacr_llm
#SBATCH --partition=batch
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --job-name=nvr_lacr_llm-eval.arc_easy
#SBATCH --output="slurm/eval/slurm-%J-%x.out"

cat $0
echo "--------------------"
date

echo "Evaluating ARC-Easy with $model_name"
python -m eval.arc.run_eval \
    --model_name_or_path models/hf_models/$model_name \
    --output_dir results/arc-easy/$model_name/$step \
    ${step:+--step "$step"} \
    ${add_bos_token:+--add_bos_token}
