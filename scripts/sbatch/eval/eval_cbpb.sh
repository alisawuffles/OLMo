#!/bin/bash
#SBATCH --account=nvr_lacr_llm
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --job-name=nvr_lacr_llm-eval.corrected_bpb
#SBATCH --output="slurm/eval/slurm-%J-%x.out"

cat $0
echo "--------------------"
date

model_name_or_path=models/hf_models/$model_name/step$step
batch_size=2

if [ -z "$max_context_length" ]; then
    echo "max_context_length will be the model's max sequence length"
    output_dir=results/cbpb/$model_name/$step
else
    echo "max_context_length is set to $max_context_length"
    output_dir=results/cbpb-ctx${max_context_length}/$model_name/$step
fi

python -m eval.eval_corrected_bpb \
    --model_name_or_path $model_name_or_path \
    --start_idx $start_idx \
    --num_examples $num_examples \
    --output_dir $output_dir \
    ${max_context_length:+--max_context_length $max_context_length} \
    ${batch_size:+--batch_size $batch_size}
