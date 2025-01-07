model_name=OLMo2-7B-pt200k
# model_name=OLMo2-7B-npt200k
# model_name=OLMo2-7B-npt200k-drop0.1
# model_name=OLMo2-7B-npt200k-drop0.1-noRspace
echo "Model: $model_name"

tasks=("arc_easy" "arc_challenge" "boolq" "bpb" "copa" "coqa" "drop" "hellaswag" "hotpotqa" "jeopardy" "lambada" "mmlu" "repeat_after_me" "squad" "tofu" "triviaqa" "wikidataqa" "winogrande")

# collect steps in a list
steps=()
for dir in $(ls models/hf_models/$model_name)
do
    step=$(echo $dir | sed 's/step//')
    steps+=($step)
done

# sort steps as integers
steps=($(echo ${steps[@]} | tr ' ' '\n' | sort -n | tr '\n' ' '))

# loop over steps
for step in ${steps[@]}
do
    echo "Step $step"
    for task in ${tasks[@]}
    do
        if [ ! -d results/$task/$model_name/$step ]; then
            id=$(sbatch --parsable --export=all,step=$step,model_name=$model_name scripts/sbatch/eval/eval_${task}.sh)
            echo "  $task: Submitted batch job $id"
        fi
    done
done