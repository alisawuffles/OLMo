import click
import pandas as pd
import json
import numpy as np
from pathlib import Path
from eval.util import load_model_and_tokenizer, batched_generate, format_example, prep_incontext_examples
from olmo.util import ensure_dir, read_json, seed_all

seed_all(42)


def evaluate_hotpotqa(model, tokenizer, test_df, with_passage, full_passage, num_incontext_examples):
    test_df = test_df.reset_index(drop=True)
    incontext_indices = prep_incontext_examples(test_df, num_incontext_examples)

    def get_hotpotqa_context(row, contexts_dict):
        contexts = set()
        for title, sentence_id in row["supporting_facts"]:
            if with_passage:
                if full_passage:
                    contexts.add("".join(contexts_dict[title]))
                elif sentence_id < len(contexts_dict[title]):
                    contexts.add(contexts_dict[title][sentence_id].strip())
        return "\n".join(contexts)

    prompts = []
    for i, row in test_df.iterrows():
        prompt = ""
        for j in incontext_indices[i]:
            ic_row = test_df.iloc[j]
            contexts_dict = {k: v for k, v in ic_row["context"]}
            context = get_hotpotqa_context(ic_row, contexts_dict)
            prompt += format_example(ic_row["question"], passage=context, answer=ic_row["answer"]) + "\n\n"
        contexts_dict = {k: v for k, v in row["context"]}
        context = get_hotpotqa_context(row, contexts_dict)
        prompt += format_example(row["question"], passage=context)
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
@click.option("--num_incontext_examples", type=int, default=1)
@click.option("--max_num_examples", type=int, default=None)
@click.option("--eval_batch_size", type=int, default=64)
@click.option("--with_passage", is_flag=True, default=False)
@click.option("--full_passage", is_flag=True, default=False)
@click.option("--add_bos_token", is_flag=True, default=False)
def main(
    model_name_or_path: str,
    step: int,
    output_dir: str,
    num_incontext_examples: int,
    max_num_examples: int,
    eval_batch_size: int,
    with_passage: bool,
    full_passage: bool,
    add_bos_token: bool,
):
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, step=step, add_bos_token=add_bos_token)
    test_df = pd.DataFrame(read_json("olmo_data/eval/hotpotqa/hotpot_dev_distractor_v1.json"))

    if max_num_examples:
        test_df = test_df.sample(min(len(test_df), max_num_examples))

    results = evaluate_hotpotqa(model, tokenizer, test_df, with_passage, full_passage, num_incontext_examples)
    metrics = {
        "accuracy": np.mean([r["correct"] for r in results]),
        "open_domain": not with_passage,
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
    with open(output_dir / "example_prompt.txt", "w") as fo:
        fo.write(results[0]["prompt"])
    pd.DataFrame(results).to_json(output_dir / "predictions.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    main()
