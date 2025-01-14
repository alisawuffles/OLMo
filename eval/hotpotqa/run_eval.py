import click
import pandas as pd
import json
import numpy as np
from pathlib import Path
from eval.util import load_model_and_tokenizer, batched_generate, format_example
from olmo.util import ensure_dir, read_json, seed_all

seed_all(42)


def evaluate_hotpotqa(model, tokenizer, test_df, open_domain, full_passage):
    prompts = []
    for _, row in test_df.iterrows():
        contexts = {k: v for k, v in row["context"]}
        prompt = ""
        for title, sentence_id in row["supporting_facts"]:
            if open_domain:
                if full_passage:
                    context = "".join(contexts[title])
                else:
                    try:
                        context = contexts[title][sentence_id].strip()
                    except IndexError:
                        continue
        prompt = format_example(row["question"], passage=context)
        prompts.append(prompt)

    print(f"--- Example prompt ---\n{prompts[0]}\n----------------------")
    outputs = batched_generate(
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        do_sample=False,
        max_new_tokens=20,
        batch_size=32,
    )

    results = []
    for prompt, output, answer in zip(prompts, outputs, test_df.answer):
        output = output.split("\n")[0]
        results.append(
            {
                "prompt": prompt,
                "output": output,
                "answer": answer,
                "correct": answer.lower() in output.lower(),
            }
        )

    return results


@click.command()
@click.option("--model_name_or_path", type=str, default="pile-npt25k")
@click.option("--step", type=int, default=None)
@click.option("--output_dir", type=str, default="results/hotpotqa/olmo-20k")
@click.option("--max_num_examples", type=int, default=None)
@click.option("--eval_batch_size", type=int, default=32)
@click.option("--open_domain", is_flag=True, default=True)
@click.option("--full_passage", is_flag=True, default=False)
@click.option("--add_bos_token", is_flag=True, default=False)
def main(
    model_name_or_path: str,
    step: int,
    output_dir: str,
    max_num_examples: int,
    eval_batch_size: int,
    open_domain: bool,
    full_passage: bool,
    add_bos_token: bool,
):
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, step=step, add_bos_token=add_bos_token)
    test_df = pd.DataFrame(read_json("olmo_data/eval/hotpotqa/hotpot_dev_distractor_v1.json"))

    if max_num_examples:
        test_df = test_df.sample(min(len(test_df), max_num_examples))

    results = evaluate_hotpotqa(model, tokenizer, test_df, open_domain=open_domain, full_passage=full_passage)
    metrics = {
        "accuracy": np.mean([r["correct"] for r in results]),
        "open_domain": open_domain,
        "full_passage": full_passage,
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
