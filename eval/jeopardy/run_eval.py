import click
import pandas as pd
from eval.util import (
    load_model_and_tokenizer,
    batched_generate,
    format_example,
    prep_incontext_examples,
    write_results,
)
from olmo.util import seed_all
from tqdm import tqdm

seed_all(42)


def evaluate_jeopardy(model, tokenizer, test_df, batch_size, num_incontext_examples, qa_format):
    test_df = test_df.reset_index(drop=True)
    incontext_indices = prep_incontext_examples(test_df, num_incontext_examples)

    prompts = []
    for i, row in tqdm(test_df.iterrows()):
        prompt = ""
        for j in incontext_indices[i]:
            ic_row = test_df.iloc[j]
            prompt += (
                format_example(question=ic_row[" Question"], answer=ic_row[" Answer"], qa_format=qa_format)
                + "\n\n"
            )

        prompt += format_example(question=row[" Question"], qa_format=qa_format)
        prompts.append(prompt)

    print(f"--- Jeopardy example prompt ---\n{prompts[0]}\n----------------------")

    outputs = batched_generate(
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        do_sample=False,
        max_new_tokens=20,
        batch_size=batch_size,
    )

    results = []
    for prompt, output, answer in zip(prompts, outputs, test_df[" Answer"]):
        output = output.split("\n\n")[0]
        results.append({"prompt": prompt, "output": output, "answer": answer, "correct": answer in output})

    return results


@click.command()
@click.option("--model_name_or_path", type=str, default="pile-npt25k")
@click.option("--step", type=int, default=None)
@click.option("--output_dir", type=str, default="results/squad/olmo-20k")
@click.option("--num_incontext_examples", type=int, default=5)
@click.option("--max_num_examples", type=int, default=10000)
@click.option("--eval_batch_size", type=int, default=128)
@click.option("--qa_format", type=str, default="qnan")
def main(
    model_name_or_path: str,
    step: int,
    output_dir: str,
    max_num_examples: int,
    num_incontext_examples: int,
    eval_batch_size: int,
    qa_format: str,
):
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, step=step)
    test_df = pd.read_json("olmo_data/eval/jeopardy/test.jsonl", lines=True)

    if max_num_examples:
        test_df = test_df.sample(min(len(test_df), max_num_examples))

    results = evaluate_jeopardy(
        model,
        tokenizer,
        test_df,
        batch_size=eval_batch_size,
        num_incontext_examples=num_incontext_examples,
        qa_format=qa_format,
    )
    write_results(results, output_dir, print_metrics=True)


if __name__ == "__main__":
    main()
