import click
import pandas as pd
import json
import numpy as np
from pathlib import Path
from eval.util import load_model_and_tokenizer, batched_generate
from olmo.util import ensure_dir
from tqdm import tqdm

np.random.seed(42)


def evaluate_jeopardy(model, tokenizer, test_df, batch_size, num_incontext_examples):
    test_df = test_df.reset_index(drop=True)
    indices = np.arange(len(test_df))
    incontext_indices = {
        i: np.random.choice(indices[indices != i], size=num_incontext_examples, replace=False)
        for i in tqdm(indices, desc="Precomputing in-context examples")
    }

    prompts = []
    for i, row in tqdm(
        test_df.iterrows(), desc=f"Constructing prompts with {num_incontext_examples} in-context examples"
    ):
        prompt = ""
        for j in incontext_indices[i]:
            prompt += (
                "Question: " + test_df.iloc[j][" Question"] + "\nAnswer: " + test_df.iloc[j][" Answer"] + "\n\n"
            )

        prompt += "Question: " + row[" Question"] + "\nAnswer"
        prompts.append(prompt)

    print(f"--- Example prompt ---\n{prompts[0]}\n----------------------")

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
        output = output.split("\n")[0]
        results.append({"prompt": prompt, "output": output, "answer": answer, "correct": answer in output})

    return results


@click.command()
@click.option("--model_name_or_path", type=str, default="pile-npt25k")
@click.option("--step", type=int, default=None)
@click.option("--output_dir", type=str, default="results/squad/olmo-20k")
@click.option("--max_num_examples", type=int, default=10000)
@click.option("--num_incontext_examples", type=int, default=1)
@click.option("--eval_batch_size", type=int, default=32)
@click.option("--add_bos_token", is_flag=True, default=False)
def main(
    model_name_or_path: str,
    step: int,
    output_dir: str,
    max_num_examples: int,
    num_incontext_examples: int,
    eval_batch_size: int,
    add_bos_token: bool,
):
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, step=step, add_bos_token=add_bos_token)
    test_df = pd.read_json("olmo_data/eval/jeopardy/test.jsonl", lines=True)

    if max_num_examples:
        test_df = test_df.sample(min(len(test_df), max_num_examples))

    results = evaluate_jeopardy(model, tokenizer, test_df, eval_batch_size, num_incontext_examples)
    metrics = {
        "accuracy": np.mean([r["correct"] for r in results]),
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

    pd.DataFrame(results).to_json(output_dir / "predictions.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    main()
