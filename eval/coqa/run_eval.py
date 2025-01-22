"""
Data downloaded from https://huggingface.co/datasets/allenai/openbookqa

------ Example prompt ------
Princess Ellen wanted nothing more than to be a singer when she grew up. She had a beautiful voice and everyone who heard it said she was the best singer in the land. But her uncle believed singing would keep her from her job as princess, so he found a witch and paid her to steal Princess Ellen's voice. The witch made a spell which gave Ellen the witch's voice. The spell also gave Ellen's voice to the witch. The witch went on to become famous as a singer, and Ellen grew up to be Queen. One day Queen Ellen heard of a singer who was the best in the land. She went to hear this singer, and was surprised to hear her own voice coming from the woman on stage. When the show was over, Ellen found the singer and gave her a penny. Ellen told the singer, "You have a magical voice". The witch was so touched by Ellen's kindness, that she gave Ellen her voice back.

Question: Who is the main character in the story?
Answer: Princess Ellen

Question: Who is the villain?
Answer
"""

import click
import pandas as pd
import json
import numpy as np
from pathlib import Path
from eval.util import load_model_and_tokenizer, batched_generate, format_example
from olmo.util import ensure_dir, read_json, seed_all

seed_all(42)


def evaluate_coqa(model, tokenizer, test_df, batch_size):
    prompts = []
    answers = []
    for _, row in test_df.iterrows():
        prompt = row["story"].strip() + "\n\n"
        for i, question in enumerate(row["questions"]):
            main_answer = row["answers"][i]["span_text"].rstrip(".,!?").capitalize()
            prompt += format_example(question["input_text"].strip())
            prompts.append(prompt)

            # add answer for the next example
            prompt += main_answer + "\n\n"

            # parse answers
            answer_aliases = [main_answer]
            for key in row["additional_answers"]:
                answer_aliases.append(row["additional_answers"][key][i]["span_text"].rstrip(".,!?").capitalize())
            answers.append(answer_aliases)

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
        output = output.split("\n")[0].rstrip(".,!?")
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
        "num_examples": len(results),
        "accuracy": np.mean([r["correct"] for r in results]),
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
        fo.write(results[1]["prompt"])
    pd.DataFrame(results).to_json(output_dir / "predictions.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    main()
