#!/bin/bash
#SBATCH --account=nvr_lacr_llm
#SBATCH --partition=batch
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --job-name=nvr_lacr_llm-eval.repeat_after_me
#SBATCH --output="slurm/eval/slurm-%J-%x.out"

cat $0
echo "--------------------"
date

echo "Evaluating $model_name at step $step on RepeatAfterMe"
python -m eval.repeat_after_me.run_eval \
    --model_name_or_path models/hf_models/$model_name \
    --output_dir results/repeat_after_me/$output_dir \
    --max_num_examples 10000 \
    ${step:+--step "$step"} \
    ${add_bos_token:+--add_bos_token}
