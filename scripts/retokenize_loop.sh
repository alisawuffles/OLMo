data_dir=olmo-mix-v2
tokenizer_name=dolma-pt200k

for file in $(ls $data_dir)
do
    id=$(sbatch --parsable --export=ALL,filename=$file,tokenizer=$tokenizer_name scripts/sbatch/retokenize.sh)
    echo "$file: Submitted job with id $id"
done