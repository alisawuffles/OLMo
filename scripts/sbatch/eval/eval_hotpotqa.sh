#!/bin/bash
#SBATCH --account=nvr_lacr_llm
#SBATCH --partition=batch
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --job-name=nvr_lacr_llm-eval.hotpotqa
#SBATCH --output="slurm/eval/slurm-%J-%x.out"

cat $0
echo "--------------------"
date

echo "Evaluating HotpotQA with $model_name"
python -m eval.hotpotqa.run_eval \
    --model_name_or_path models/hf_models/$model_name \
    --output_dir results/hotpotqa/$model_name/$step \
    --max_num_examples 1000 \
    ${step:+--step "$step"} \
    ${add_bos_token:+--add_bos_token}
