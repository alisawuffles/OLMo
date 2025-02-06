# model_name=OLMo2-1B-pt200k
# model_name=OLMo2-7B-pt200k
# model_name=OLMo2-7B-pts200k-t100k-ctx2759
# model_name=OLMo2-7B-pts200k-t140k-ctx2818
# model_name=OLMo2-7B-pts200k-t160k-ctx2881
# model_name=OLMo2-7B-pts200k-t180k-ctx2995
model_name=OLMo2-7B-pts200k-t80k-ctx2756-colon
# model_name=OLMo2-7B-pts200k-t180k-ctx3000-colon

# other eval params
qa_format=qnan
num_incontext_examples=5
# eval_batch_size=16

echo "Model: $model_name"
echo "QA format: $qa_format"

# complete list of tasks
tasks=("arc-easy" "arc-challenge" "boolq" "copa" "coqa" "drop" "hellaswag" "hotpotqa" "jeopardy" "lambada" "mmlu" "openbookqa" "squad" "tofu" "triviaqa" "wikidataqa" "winogrande")
# tasks=("arc-easy" "arc-challenge" "hellaswag" "mmlu" "openbookqa" "winogrande")

# collect steps in a list
steps=()
for dir in $(ls models/hf_models/$model_name)
do
    step=$(echo $dir | sed 's/step//')
    steps+=($step)
done

# sort steps as integers in reverse order
# steps=($(echo ${steps[@]} | tr ' ' '\n' | sort -nr | tr '\n' ' '))
# take only first element (last step)
# steps=(${steps[0]})
steps=(16000)

# loop over steps
for step in ${steps[@]}
do
    echo "Step $step"
    for task in ${tasks[@]}
    do
        if [ $task == "coqa" ]; then
            output_dir=results/$task-$qa_format/$model_name/$step
        else
            output_dir=results/$task-$qa_format-ice${num_incontext_examples}/$model_name/$step
        fi

        if [ ! -d $output_dir ]; then
            id=$(sbatch --parsable --export=all,model_name=$model_name,step=$step,num_incontext_examples=$num_incontext_examples,eval_batch_size=$eval_batch_size,output_dir=$output_dir,qa_format=$qa_format scripts/sbatch/eval/eval_${task}.sh)
            echo "  $task: Submitted batch job $id"
        fi
    done
done
