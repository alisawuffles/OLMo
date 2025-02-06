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

if [ -z "$max_context_length" ]; then
    echo "max_context_length will be the model's max sequence length"
    output_dir=results/bpb/$model_name/$step
else
    echo "max_context_length is set to $max_context_length"
    output_dir=results/bpb-ctx${max_context_length}/$model_name/$step
fi

echo "Evaluating bits per byte of $model_name at step $step"
python -m eval.eval_bpb \
    --model_name_or_path models/hf_models/$model_name \
    --max_num_examples 1000 \
    --eval_batch_size 2 \
    --output_dir $output_dir \
    ${eval_batch_size:+--eval_batch_size "$eval_batch_size"} \
    ${step:+--step "$step"} \
    ${qa_format:+--qa_format "$qa_format"}
