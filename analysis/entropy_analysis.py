import json
import os
import random

import click
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from eval.util import load_model_and_tokenizer


@click.command()
@click.option("--pt_model_path", type=str)
@click.option("--pts_tokenizer_path", type=str)
@click.option(
    "--eval_dir",
    type=str,
    help="Path to folder with jsonl files",
    default="olmo_data/olmo2_shuffle",
)
@click.option("--max_num_examples", type=int, default=1)
@click.option("--max_context_length", type=int, default=None)
@click.option("--constraint_beam_size", type=int, default=4)
@click.option("--batch_size", type=int, default=1)
@click.option("--overwrite", is_flag=True, default=False)
def main(
    pt_model_path: str,
    pts_tokenizer_path: str,
    eval_dir: str,
    max_num_examples: int,
    max_context_length: int,
    constraint_beam_size: int,
    batch_size: int,
    overwrite: bool,
):
    model, pt_tokenizer = load_model_and_tokenizer(pt_model_path, padding_side="right")
    max_context_length = max_context_length or model.config.max_position_embeddings

    pts_tokenizer = AutoTokenizer.from_pretrained(pts_tokenizer_path)

    # read .jsonl files from eval_dir until we have max_num_examples
    eval_data = []
    for eval_file in os.listdir(eval_dir):
        nrows = max_num_examples - len(eval_data)
        df = pd.read_json(os.path.join(eval_dir, eval_file), lines=True, compression="zstd", nrows=nrows)
        eval_data.extend(df.text.tolist())
        if len(eval_data) >= max_num_examples:
            break
    print(f"Loaded {len(eval_data)} examples.")

    t = os.path.dirname(pts_tokenizer_path).split("-")[-3]
    fo_path = f"analysis/data/entropy_results_{t}.jsonl"
    if os.path.exists(fo_path) and not overwrite:
        print(f"File {fo_path} already exists. Skipping.")
        return

    fo = open(fo_path, "w")

    for i in tqdm(range(0, len(eval_data), batch_size)):
        batch_texts = eval_data[i : i + batch_size]
        pts_inputs = pts_tokenizer(
            [pts_tokenizer.eos_token + text for text in batch_texts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_context_length,
        )
        pt_inputs = pt_tokenizer(
            [pt_tokenizer.eos_token + text for text in batch_texts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_context_length,
        )
        pts_input_ids = pts_inputs.input_ids
        pt_input_ids = pt_inputs.input_ids

        with torch.no_grad():
            # entropy of output logits at every time step
            outputs = model(input_ids=pt_input_ids, labels=pt_input_ids, return_dict=True)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            entropies = -torch.sum(probs * torch.log(probs), dim=-1)

        for i in range(len(batch_texts)):
            # collect token boundaries to determine whether n-grams cross token boundaries
            pts_tokenized = pts_tokenizer.convert_ids_to_tokens(pts_input_ids[i])
            pts_token_boundaries = [0]
            for token in pts_tokenized:
                pts_token_boundaries.append(pts_token_boundaries[-1] + len(token))

            pt_tokenized = pt_tokenizer.convert_ids_to_tokens(
                pt_input_ids[i][pt_input_ids[i] != pt_tokenizer.pad_token_id]
            )
            pt_token_boundaries = [0]
            for token in pt_tokenized:
                pt_token_boundaries.append(pt_token_boundaries[-1] + len(token))

            # for each token, determine whether it's contained entirely in a superword token
            data = []
            for idx in range(1, len(pt_tokenized)):  # start at idx 1 to skip EOS token
                a, b = pt_token_boundaries[idx], pt_token_boundaries[idx + 1]
                within_superword_token = False
                if any(
                    [
                        (l <= a < b < r) or (l < a < b <= r)
                        for l, r in zip(pts_token_boundaries, pts_token_boundaries[1:])
                    ]
                ):
                    within_superword_token = True

                # get the superword token that contains this subword token
                pts_token = None
                if within_superword_token:
                    token_idx = [
                        i
                        for i in range(len(pts_token_boundaries) - 1)
                        if pts_token_boundaries[i] <= a < pts_token_boundaries[i + 1]
                    ][0]
                    pts_token = pts_tokenized[token_idx]

                data.append((pt_tokenized[idx], entropies[i][idx - 1].item(), within_superword_token, pts_token))
            print(data)

            ex = {
                "text": batch_texts[i],
                "pt_tokens": pt_tokenized[1:],
                "pts_tokens": pts_tokenized[1:],
                "data": data,
            }
            fo.write(json.dumps(ex) + "\n")


if __name__ == "__main__":
    main()
