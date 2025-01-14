#!/bin/bash
#SBATCH --account=nvr_lacr_llm
#SBATCH --partition=batch
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --job-name=nvr_lacr_llm-eval.bpb
#SBATCH --output="slurm/eval/slurm-%J-%x.out"

cat $0
echo "--------------------"
date

echo "Evaluating bits per byte with $model_name at step $step"
python -m eval.eval_bpb \
    --model_name_or_path models/hf_models/$model_name \
    --max_num_examples 1000 \
    --max_context_length 512 \
    --eval_batch_size 8 \
    --output_dir results/bpb/$model_name/$step \
    ${step:+--step "$step"} \
    ${add_bos_token:+--add_bos_token}
