import torch
import torch.nn.functional as F
import json
import os
import click
from tqdm import tqdm
from pathlib import Path
from olmo.util import ensure_dir
import random
from eval.util import load_model_and_tokenizer

random.seed(42)


def get_bits_per_byte(model, tokenizer, eval_data, batch_size, max_context_length):
    total_accuracy = 0
    total_reciprocal_rank = 0
    total_loss = 0
    total_tokens = 0
    total_bytes = 0
    for i in tqdm(range(0, len(eval_data), batch_size)):
        batch_texts = [item["text"] for item in eval_data[i : i + batch_size]]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_context_length,
        )
        inputs.attention_mask = inputs.attention_mask.to(model.device)
        inputs.input_ids = inputs.input_ids.to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)
            logits = outputs.logits.to(model.device)
            labels = inputs.input_ids.clone()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask = inputs.attention_mask[..., :-1].contiguous()

            # calculate accuracy
            predictions = shift_logits.argmax(dim=-1)
            correct = (predictions == shift_labels).float() * shift_attention_mask
            total_accuracy = correct.sum()

            # calculate mean reciprocal rank
            ranks = torch.argsort(shift_logits, dim=-1, descending=True)
            masked_ranks = (ranks == shift_labels.unsqueeze(-1)).nonzero()[..., -1].view(
                shift_attention_mask.shape
            ) * shift_attention_mask  # mask out padding

            reciprocal_ranks = torch.where(
                shift_attention_mask > 0,  # Apply mask condition
                1.0 / masked_ranks.float(),  # Compute reciprocal for valid entries
                torch.zeros_like(masked_ranks.float()),  # Set invalid positions to 0
            )

            total_reciprocal_rank += reciprocal_ranks.sum()

            # get neg log likelihood
            loss = (
                F.cross_entropy(shift_logits.transpose(1, 2), shift_labels, reduction="none")
                * shift_attention_mask
            )
            total_loss += loss.sum()
            total_tokens += shift_attention_mask.sum()

        total_bytes += sum(
            len(text) for text in tokenizer.batch_decode(inputs.input_ids, skip_special_tokens=True)
        )

    bits_per_byte = total_loss / total_bytes / torch.log(torch.tensor(2.0))
    accuracy = total_accuracy / total_tokens
    # mrr = total_reciprocal_rank / total_tokens

    return {
        "bits_per_byte": bits_per_byte.item(),
        "accuracy": accuracy.item(),
        # "mean_reciprocal_rank": mrr.item(),
        "total_tokens": total_tokens.item(),
        "total_bytes": total_bytes,
        "num_examples": len(eval_data),
        "max_context_length": max_context_length,
    }


@click.command()
@click.option("--model_name_or_path", type=str, default="OLMo2-7B-npt200k")
@click.option("--step", type=int, default=None)
@click.option("--output_dir", type=str, default="results/bpb/olmo-20k")
@click.option(
    "--eval_dir",
    type=str,
    help="Path to folder with jsonl files",
    default="olmo_data/dolmino_shuffle/eval",
)
@click.option("--max_num_examples", type=int, default=None)
@click.option("--eval_batch_size", type=int, default=8)
@click.option("--add_bos_token", is_flag=True, default=False)
@click.option("--max_context_length", type=int, default=None)
def main(
    model_name_or_path: str,
    step: int,
    output_dir: str,
    eval_dir: str,
    max_num_examples: int,
    eval_batch_size: int,
    add_bos_token: bool,
    max_context_length: int,
):
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, step=step, add_bos_token=add_bos_token)
    max_context_length = max_context_length or model.config.max_position_embeddings
    print(f"Using max context length of {max_context_length}")

    # read .jsonl files from eval_dir until we have max_num_examples
    eval_data = []
    for eval_file in os.listdir(eval_dir):
        with open(os.path.join(eval_dir, eval_file), "r", encoding="utf-8") as file:
            for line in file:
                eval_data.append(json.loads(line))
            if max_num_examples and len(eval_data) >= max_num_examples:
                break

    eval_data = random.sample(eval_data, max_num_examples)
    print(f"Loaded {len(eval_data)} examples.")

    metrics = get_bits_per_byte(
        model, tokenizer, eval_data=eval_data, batch_size=eval_batch_size, max_context_length=max_context_length
    )

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


if __name__ == "__main__":
    main()
