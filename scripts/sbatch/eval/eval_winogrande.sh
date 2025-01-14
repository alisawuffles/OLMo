#!/bin/bash
#SBATCH --account=nvr_lacr_llm
#SBATCH --partition=batch
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --job-name=nvr_lacr_llm-eval.winogrande
#SBATCH --output="slurm/eval/slurm-%J-%x.out"

cat $0
echo "--------------------"
date

echo "Evaluating $model_name at step $step on Winogrande"
python -m eval.winogrande.run_eval \
    --model_name_or_path models/hf_models/$model_name \
    --output_dir results/winogrande/$model_name/$step \
    --max_num_examples 1000 \
    ${step:+--step "$step"} \
    ${add_bos_token:+--add_bos_token}
