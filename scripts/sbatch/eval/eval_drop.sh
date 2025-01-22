#!/bin/bash
#SBATCH --account=nvr_lacr_llm
#SBATCH --partition=batch,interactive
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --job-name=nvr_lacr_llm-eval.drop
#SBATCH --output="slurm/eval/slurm-%J-%x.out"

cat $0
echo "--------------------"
date

echo "Evaluating $model_name at step $step on DROP"
python -m eval.drop.run_eval \
    --model_name_or_path models/hf_models/$model_name \
    --output_dir results/drop-${num_incontext_examples}/$model_name/$step \
    --max_num_examples 1000 \
    --num_incontext_examples $num_incontext_examples \
    ${step:+--step "$step"} \
    ${add_bos_token:+--add_bos_token}
