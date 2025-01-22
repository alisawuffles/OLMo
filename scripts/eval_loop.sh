## 7B models
# model_name=OLMo2-7B-pt200k
# model_name=OLMo2-7B-pts200k
# model_name=OLMo2-7B-pts200k-t180k-ctx2995
# model_name=OLMo2-7B-pt200k-dolm
model_name=OLMo2-7B-npt200k-dolm
# model_name=OLMo2-7B-npt200k-drop0.1-noRspace-dolm
## 1B models
# model_name=OLMo2-1B-pt200k
# model_name=OLMo2-1B-pts200k
# model_name=OLMo2-1B-pts200k-t180k
# model_name=OLMo2-1B-pts200k-t180k-ctx1498
echo "Model: $model_name"

tasks=("arc-easy" "arc-challenge" "boolq" "copa" "coqa" "drop" "hellaswag" "hotpotqa" "jeopardy" "lambada" "mmlu" "openbookqa" "squad" "tofu" "triviaqa" "wikidataqa" "winogrande")
# tasks=("bpb")

# collect steps in a list
steps=()
for dir in $(ls models/hf_models/$model_name)
do
    step=$(echo $dir | sed 's/step//')
    steps+=($step)
done

# sort steps as integers
steps=($(echo ${steps[@]} | tr ' ' '\n' | sort -n | tr '\n' ' '))

# set number of in-context examples
num_incontext_examples=5

# loop over steps
for step in ${steps[@]}
do
    echo "Step $step"
    for task in ${tasks[@]}
    do
        if [ ! -d results/$task-ice${num_incontext_examples}/$model_name/$step ]; then
            id=$(sbatch --parsable --export=all,step=$step,model_name=$model_name,num_incontext_examples=$num_incontext_examples scripts/sbatch/eval/eval_${task}.sh)
            echo "  $task: Submitted batch job $id"
        fi
    done
done
