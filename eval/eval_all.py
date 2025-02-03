import os

import click
import pandas as pd
from eval.util import load_model_and_tokenizer, write_results
from eval.evaluation_config import EVALUATION_CONFIGS
from olmo.util import read_json, seed_all

seed_all(42)


@click.command()
@click.option("--model_name_or_path", type=str, default="pile-npt25k")
@click.option("--step", type=int, default=None)
def main(
    model_name_or_path: str,
    step: int,
):
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, step=step)

    for task_name in EVALUATION_CONFIGS:
        model_name = os.path.basename(model_name_or_path)
        num_incontext_examples = EVALUATION_CONFIGS[task_name].get("num_incontext_examples", 5)
        max_num_examples = EVALUATION_CONFIGS[task_name].get("max_num_examples", 1000)
        eval_batch_size = EVALUATION_CONFIGS[task_name].get("eval_batch_size", 32)

        output_dir = f"results/{task_name.lower()}-qa"
        if num_incontext_examples:
            output_dir += f"-ice{num_incontext_examples}"
        output_dir += f"/{model_name}/{step}"

        if not os.path.exists(output_dir):
            test_path = EVALUATION_CONFIGS[task_name]["path"]
            if test_path.endswith(".jsonl"):
                test_df = pd.read_json(test_path, lines=True)
            elif test_path.endswith(".json"):
                test_data = read_json(test_path)
                for field in ["data", "examples"]:
                    if field in test_data:
                        test_data = test_data[field]
                        break
                test_df = pd.DataFrame(test_data)

            if max_num_examples:
                test_df = test_df.sample(min(len(test_df), max_num_examples), random_state=42)
            results = EVALUATION_CONFIGS[task_name]["eval_func"](
                model, tokenizer, test_df, eval_batch_size, num_incontext_examples
            )
            write_results(results, output_dir)


if __name__ == "__main__":
    main()
