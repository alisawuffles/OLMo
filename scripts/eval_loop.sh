model_name=OLMo2-7B-pt200k
# model_name=OLMo2-7B-pts200k-t100k-ctx2759
# model_name=OLMo2-7B-pts200k-t140k-ctx2818
# model_name=OLMo2-7B-pts200k-t160k-ctx2881
# model_name=OLMo2-7B-pts200k-t180k-ctx2995
echo "Model: $model_name"

# collect steps in a list
steps=()
for dir in $(ls models/hf_models/$model_name)
do
    step=$(echo $dir | sed 's/step//')
    steps+=($step)
done

# sort steps as integers in reverse order
steps=($(echo ${steps[@]} | tr ' ' '\n' | sort -nr | tr '\n' ' '))

# set number of in-context examples
num_incontext_examples=5

for step in ${steps[@]}
do
    if [ ! -d results/winogrande-ice5/$model_name/$step ]; then    
        id=$(sbatch --parsable --export=all,model_name=$model_name,step=$step scripts/sbatch/eval/eval_all.sh)
        echo "$step: Submitted batch job $id"
    fi
done
