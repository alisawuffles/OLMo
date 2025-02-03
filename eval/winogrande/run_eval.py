"""Data downloaded from https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1.zip"""

import click
import pandas as pd
import json
import numpy as np
from pathlib import Path
from eval.util import (
    load_model_and_tokenizer,
    batched_generate,
    format_example,
    parse_mc_pred,
    prep_incontext_examples,
)
from olmo.util import ensure_dir, seed_all

seed_all(42)


def evaluate_winogrande(model, tokenizer, test_df, batch_size, num_incontext_examples):
    test_df = test_df.reset_index(drop=True)
    incontext_indices = prep_incontext_examples(test_df, num_incontext_examples)

    prompts = []
    for i, row in test_df.iterrows():
        prompt = ""
        for j in incontext_indices[i]:
            ic_row = test_df.iloc[j]
            question = ic_row["sentence"].strip() + " What goes in the blank?\n"
            options = [ic_row["option1"], ic_row["option2"]]
            prompt += format_example(question, choices=options, answer="AB"[ic_row["answer"] - 1]) + "\n\n"

        question = row["sentence"].strip() + " What goes in the blank?\n"
        options = [row["option1"], row["option2"]]
        prompt += format_example(question, choices=options)
        prompts.append(prompt)

    print(f"--- Winogrande example prompt ---\n{prompts[0]}\n----------------------")

    outputs = batched_generate(
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        do_sample=False,
        max_new_tokens=20,
        batch_size=batch_size,
    )

    results = []
    for prompt, output, answer in zip(prompts, outputs, test_df["answer"]):
        output = output.split("\n")[0]
        parsed_pred = parse_mc_pred(output, num_options=2)
        results.append(
            {
                "prompt": prompt,
                "output": output,
                "answer": "AB"[answer - 1],
                "valid": parsed_pred is not None,
                "correct": parsed_pred == "AB"[answer - 1],
            }
        )

    return results


@click.command()
@click.option("--model_name_or_path", type=str, default="pile-npt25k")
@click.option("--step", type=int, default=None)
@click.option("--output_dir", type=str, default="results/squad/olmo-20k")
@click.option("--num_incontext_examples", type=int, default=1)
@click.option("--max_num_examples", type=int, default=None)
@click.option("--eval_batch_size", type=int, default=128)
@click.option("--add_bos_token", is_flag=True, default=False)
def main(
    model_name_or_path: str,
    step: int,
    output_dir: str,
    num_incontext_examples: int,
    max_num_examples: int,
    eval_batch_size: int,
    add_bos_token: bool,
):
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, step=step, add_bos_token=add_bos_token)
    test_df = pd.read_json("olmo_data/eval/winogrande/dev.jsonl", lines=True)

    if max_num_examples:
        test_df = test_df.sample(min(len(test_df), max_num_examples), random_state=42)

    results = evaluate_winogrande(model, tokenizer, test_df, eval_batch_size, num_incontext_examples)
    metrics = {
        "accuracy": np.mean([r["correct"] for r in results]),
        "valid_answer": np.mean([r["valid"] for r in results]),
        "num_examples": len(results),
    }
    # print metrics
    for k, v in metrics.items():
        print(f"{k}: {v}")

    if add_bos_token:
        output_dir += "-bos"
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    print(f"Saving results to {output_dir}")

    with open(output_dir / "metrics.json", "w") as fo:
        json.dump(metrics, fo, indent=4)
    with open(output_dir / "example_prompt.txt", "w") as fo:
        fo.write(results[0]["prompt"])
    pd.DataFrame(results).to_json(output_dir / "predictions.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    main()
