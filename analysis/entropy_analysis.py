import json
import os

import click
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from eval.util import load_model_and_tokenizer


@click.command()
@click.option("--model_path", type=str)
@click.option("--pts_tokenizer_path", type=str)
@click.option(
    "--eval_dir",
    type=str,
    help="Path to folder with jsonl files",
    default="olmo_data/olmo2_shuffle",
)
@click.option("--max_num_examples", type=int, default=1)
@click.option("--max_context_length", type=int, default=None)
@click.option("--batch_size", type=int, default=1)
@click.option("--overwrite", is_flag=True, default=False)
def main(
    model_path: str,
    pts_tokenizer_path: str,
    eval_dir: str,
    max_num_examples: int,
    max_context_length: int,
    batch_size: int,
    overwrite: bool,
):
    model, tokenizer = load_model_and_tokenizer(model_path, padding_side="right")
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

    # cursed naming convention
    model_size_str = [s for s in os.path.dirname(model_path).split("-") if s.endswith("B")][0]
    if "pts200k" in model_path:
        transition_point_str = [s for s in os.path.dirname(model_path).split("-") if s.startswith("t")][0]
        t1 = f"{model_size_str}-{transition_point_str}-step8000"
    else:
        t1 = f"{model_size_str}-pt"

    t2 = os.path.dirname(pts_tokenizer_path).split("-")[-3]  # tokenizer transition point
    fo_path = f"analysis/data/loss_analysis_{t1}_{t2}.jsonl"
    if os.path.exists(fo_path) and not overwrite:
        print(f"File {fo_path} already exists. Skipping.")
        return
    print(f"Results will be written to {fo_path}", flush=True)

    fo = open(fo_path, "w")

    for i in tqdm(range(0, len(eval_data), batch_size)):
        batch_texts = eval_data[i : i + batch_size]
        inputs = tokenizer(
            [tokenizer.eos_token + text for text in batch_texts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_context_length,
        )
        pts_inputs = pts_tokenizer(
            [pts_tokenizer.eos_token + text for text in batch_texts],
            return_tensors="pt",
            padding="longest",
        )
        pts_input_ids = pts_inputs.input_ids
        input_ids = inputs.input_ids

        # entropy of output logits at every time step
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids, return_dict=True)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            entropies = -torch.sum(probs * torch.log(probs), dim=-1).detach()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            losses = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=tokenizer.pad_token_id,
                reduction="none",
            ).view(input_ids.size(0), -1)

        for i in range(len(batch_texts)):
            # collect token boundaries to determine whether n-grams cross token boundaries
            pts_tokenized = pts_tokenizer.convert_ids_to_tokens(pts_input_ids[i])
            pts_token_boundaries = [0]
            for token in pts_tokenized:
                pts_token_boundaries.append(pts_token_boundaries[-1] + len(token))

            tokenized = tokenizer.convert_ids_to_tokens(input_ids[i][input_ids[i] != tokenizer.pad_token_id])
            token_boundaries = [0]
            for token in tokenized:
                token_boundaries.append(token_boundaries[-1] + len(token))

            # for each token, determine whether it's contained entirely in a superword token
            data = []
            for idx in range(1, len(tokenized)):  # start at idx 1 to skip EOS token
                a, b = token_boundaries[idx], token_boundaries[idx + 1]
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

                data.append(
                    {
                        "token": tokenized[idx],
                        "entropy": entropies[i][idx - 1].item(),
                        "loss": losses[i][idx - 1].item(),
                        "within_superword_token": within_superword_token,
                        "pts_token": pts_token,
                    }
                )

            ex = {
                "text": batch_texts[i],
                "tokens": tokenized[1:],
                "pts_tokens": pts_tokenized[1:],
                "data": data,
            }
            fo.write(json.dumps(ex) + "\n")


if __name__ == "__main__":
    main()
