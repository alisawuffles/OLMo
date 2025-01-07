"""Data downloaded from https://huggingface.co/datasets/allenai/openbookqa"""

import click
import pandas as pd
import json
import numpy as np
from pathlib import Path
from eval.util import load_model_and_tokenizer, batched_generate
from olmo.util import ensure_dir, read_json


def evaluate_coqa(model, tokenizer, test_df, batch_size):
    prompts = []
    answers = []
    for _, row in test_df.iterrows():
        prompt = row["story"].strip() + "\n\n"
        for i, question in enumerate(row["questions"]):
            prompt += f"\nQuestion: {question['input_text']}"
            prompt += "\nAnswer"
            prompts.append(prompt)

            # parse answers
            main_answer = row["answers"][i]["span_text"]
            all_answers = [main_answer]
            for key in row["additional_answers"]:
                all_answers.extend(row["additional_answers"][key][i]["span_text"])
            answers.append(all_answers)

            prompt += f": {main_answer}\n\n"

    print("--- Example prompts ---\n" + "\n----\n".join(prompts[:5]) + "\n----------------------")

    outputs = batched_generate(
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        do_sample=False,
        max_new_tokens=20,
        batch_size=batch_size,
    )

    results = []
    for prompt, output, answers in zip(prompts, outputs, answers):
        output = output.split("\n")[0]
        results.append(
            {"prompt": prompt, "output": output, "answer": answers, "correct": output.strip() in answers}
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
    test_df = pd.DataFrame(read_json("olmo_data/eval/coqa/coqa-dev-v1.0.json")["data"])

    if max_num_examples:
        test_df = test_df.sample(min(len(test_df), max_num_examples), random_state=42)

    results = evaluate_coqa(model, tokenizer, test_df, batch_size=eval_batch_size)
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
