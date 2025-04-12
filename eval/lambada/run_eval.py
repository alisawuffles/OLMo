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

seed_all(42)


def evaluate_lambada(model, tokenizer, test_df, batch_size, num_incontext_examples, qa_format):
    test_df = test_df.reset_index(drop=True)
    incontext_indices = prep_incontext_examples(test_df, num_incontext_examples)

    prompts, continuations = [], []
    for i, row in test_df.iterrows():
        prompt = "For each incomplete passage below, determine from context what the next word should be.\n\n"
        for j in incontext_indices[i]:
            ic_row = test_df.iloc[j]
            context, next_word = ic_row["text"].rsplit(" ", 1)
            if qa_format == "cont":
                prompt += f"{context} {next_word}\n\n"
            else:
                prompt += (
                    format_example(
                        question=context, question_prefix="Passage:", answer=next_word, qa_format=qa_format
                    )
                    + "\n\n"
                )

        context, next_word = row["text"].rsplit(" ", 1)
        if qa_format == "cont":
            prompt += f"{context}"
        else:
            prompt += format_example(context, question_prefix="Passage:", qa_format=qa_format)
        prompts.append(prompt)
        continuations.append(next_word)

    print(f"--- Lambada example prompt ---\n{prompts[0]}\n----------------------")

    outputs = batched_generate(
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        do_sample=False,
        max_new_tokens=20,
        batch_size=batch_size,
    )

    results = []
    for prompt, output, continuation in zip(prompts, outputs, continuations):
        output = output.split("\n\n")[0]
        parsed_output = output.lstrip().split(" ")[0].rstrip(".,!?")
        results.append(
            {
                "prompt": prompt,
                "output": parsed_output,
                "continuation": continuation,
                "correct": continuation == parsed_output,
            }
        )

    return results


@click.command()
@click.option("--model_name_or_path", type=str, default="pile-npt25k")
@click.option("--step", type=int, default=None)
@click.option("--output_dir", type=str, default="results/squad/olmo-20k")
@click.option("--num_incontext_examples", type=int, default=5)
@click.option("--max_num_examples", type=int, default=None)
@click.option("--eval_batch_size", type=int, default=64)
@click.option("--qa_format", type=str, default="qnan")
def main(
    model_name_or_path: str,
    step: int,
    output_dir: str,
    num_incontext_examples: int,
    max_num_examples: int,
    eval_batch_size: int,
    qa_format: str,
):
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, step=step)
    test_df = pd.read_json("olmo_data/eval/lambada/test.jsonl", lines=True)

    if max_num_examples:
        test_df = test_df.sample(min(len(test_df), max_num_examples), random_state=42)

    results = evaluate_lambada(
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
