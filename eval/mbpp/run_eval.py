import json
import os
import re
from pathlib import Path

import click
import pandas as pd

from eval.humaneval.evaluation import evaluate_functional_correctness
from eval.util import batched_generate, load_model_and_tokenizer
from olmo.util import ensure_dir, seed_all

seed_all(42)


def format_mbpp_program(problem, completion):
    tests = "\n".join(problem["test_list"])
    check_program = format_prompt(problem) + completion + "\n" + tests
    return check_program


def clean_mbpp_output(output):
    """Sometimes it starts generating assert statements outside the function, in which case we should truncate."""
    first_line, lines = output.split("\n")[0], output.split("\n")[1:]
    truncated_output = [first_line]
    for line in lines:
        if line and (not line.startswith(" ")) and (not line.startswith("def")):
            break
        truncated_output.append(line)
    return "\n".join(truncated_output)


def format_prompt(row):
    lines = row["code"].split("\n")
    header_lines = []
    for line in lines:
        header_lines.append(line.rstrip())
        if line.startswith("def"):
            break
    header = "\n".join(header_lines) + "\n    "

    prompt = (
        '"""\n'
        + row["text"]
        + " Your code should pass these tests:\n"
        + "\n".join(row["test_list"])
        + '\n"""\n'
        + header
    )
    return prompt


def evaluate_mbpp(model, tokenizer, test_df, batch_size):
    test_df = test_df.reset_index(drop=True)
    prompts = [format_prompt(row) for i, row in test_df.iterrows()]

    print(f"--- MBPP example prompt ---\n{prompts[0]}\n----------------------")

    outputs = batched_generate(
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        do_sample=True,
        top_p=0.95,
        temperature=0.8,
        max_new_tokens=128,
        batch_size=batch_size,
        stop_strings=["\nclass", "\ndef", "\n#", "\nif"],
    )
    # remove stop_strings if they are at the end of the string
    outputs = [clean_mbpp_output(re.sub(r"\n(class|def|#|if)$", "", out)) for out in outputs]
    predictions = [
        {"task_id": ex["task_id"], "prompt": prompt, "completion": out}
        for ex, prompt, out in zip(test_df.to_dict(orient="records"), prompts, outputs)
    ]
    return predictions


@click.command()
@click.option("--model_name_or_path", type=str, default=None)
@click.option("--output_dir", type=str)
@click.option("--step", type=int, default=None)
@click.option("--max_num_examples", type=int, default=None)
@click.option("--eval_batch_size", type=int, default=32)
@click.option("--pass_at_k", type=int, default=10)
@click.option("--unbiased_sampling_size_n", type=int, default=20)
@click.option("--overwrite_samples", is_flag=True)
def main(
    model_name_or_path: str,
    output_dir: str,
    step: int,
    max_num_examples: int,
    eval_batch_size: int,
    pass_at_k: int,
    unbiased_sampling_size_n: int,
    overwrite_samples: bool,
):
    output_dir = Path(output_dir)
    if not os.path.exists(output_dir / "predictions.jsonl") or overwrite_samples:
        model, tokenizer = load_model_and_tokenizer(model_name_or_path, step=step)
        test_df = pd.read_json("olmo_data/eval/mbpp/mbpp.jsonl", lines=True)

        if max_num_examples:
            test_df = test_df.sample(min(len(test_df), max_num_examples))

        # duplicate test_df unbiased_sampling_size_n times
        test_df = pd.concat([test_df] * unbiased_sampling_size_n, ignore_index=True)

        predictions = evaluate_mbpp(model, tokenizer, test_df, batch_size=eval_batch_size)

        ensure_dir(output_dir)
        pd.DataFrame(predictions).to_json(output_dir / "predictions.jsonl", orient="records", lines=True)

    predictions = pd.read_json(output_dir / "predictions.jsonl", lines=True)
    metrics = evaluate_functional_correctness(
        sample_file=str(output_dir / "predictions.jsonl"),
        k=[pass_at_k],
        n_workers=64,
        problem_file="olmo_data/eval/mbpp/mbpp.jsonl",
        format_fn=format_mbpp_program,
    )
    metrics["num_examples"] = len(predictions) // unbiased_sampling_size_n
    for k, v in metrics.items():
        print(f"{k}: {v}")

    results = pd.read_json(output_dir / "scored_predictions.jsonl", lines=True)
    with open(output_dir / "example_prompt.txt", "w") as fo:
        fo.write(results.iloc[0]["prompt"])
    with open(output_dir / "metrics.json", "w") as fo:
        json.dump(metrics, fo, indent=4)


if __name__ == "__main__":
    main()
