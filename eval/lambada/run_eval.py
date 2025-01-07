import click
import pandas as pd
import json
import numpy as np
from pathlib import Path
from eval.util import load_model_and_tokenizer, batched_generate
from olmo.util import ensure_dir


def evaluate_lambada(model, tokenizer, test_df, batch_size):
    prompts, continuations = [], []
    for _, row in test_df.iterrows():
        prompt, continuation = row["text"].rsplit(" ", 1)
        prompts.append(prompt)
        continuations.append(continuation)

    print(f"--- Example prompt ---\n{prompts[0]}\n----------------------")

    outputs = batched_generate(
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        do_sample=False,
        max_new_tokens=5,
        batch_size=batch_size,
    )

    results = []
    for prompt, output, continuation in zip(prompts, outputs, continuations):
        output = output.split("\n")[0]
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
@click.option("--max_num_examples", type=int, default=None)
@click.option("--eval_batch_size", type=int, default=32)
@click.option("--add_bos_token", is_flag=True, default=False)
def main(
    model_name_or_path: str,
    step: int,
    output_dir: str,
    max_num_examples: int,
    eval_batch_size: int,
    add_bos_token: bool,
):
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, step=step, add_bos_token=add_bos_token)
    test_df = pd.read_json("olmo_data/eval/lambada/test.jsonl", lines=True)

    if max_num_examples:
        test_df = test_df.sample(min(len(test_df), max_num_examples), random_state=42)

    results = evaluate_lambada(model, tokenizer, test_df, eval_batch_size)
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
