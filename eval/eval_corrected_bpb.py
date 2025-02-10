import torch
import torch.nn.functional as F
import json
import os
import random
import pandas as pd
import click
from tqdm import tqdm
from pathlib import Path
from transformers import LogitsProcessorList
from olmo.util import ensure_dir
from eval.util import load_model_and_tokenizer
from eval.custom_logits_process import SurfaceFormConstraintLogitsProcessor

random.seed(42)


def get_corrected_bits_per_byte(model, tokenizer, eval_data, batch_size, max_context_length, constraint_beam_size):
    total_loss = 0
    total_corrected_loss = 0
    total_tokens = 0
    total_bytes = 0

    def get_segmentations(input_ids, constraint):
        logits_processor = LogitsProcessorList([SurfaceFormConstraintLogitsProcessor(constraint, tokenizer)])
        model.generation_config.early_stopping = True
        outputs = model.generate(
            input_ids,
            logits_processor=logits_processor,
            num_beams=constraint_beam_size,
            num_return_sequences=constraint_beam_size,
            max_new_tokens=model.config.max_position_embeddings,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
        return outputs

    examples = []

    # one document at a time
    for text in tqdm(eval_data):
        # text = "Lexical analysis is the conversion of a text into meaningful lexical tokens based on a lexical grammar. Learn about the stages, categories, and examples of lexical tokens, and the difference between lexical analysis and large language models."
        text = "I like pie."
        inputs = tokenizer(
            tokenizer.eos_token + text + tokenizer.eos_token,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_context_length,
        )
        input_ids = inputs.input_ids.to(model.device)

        with torch.no_grad():
            # 1. calculate loss of BPE segmentation
            per_token_loss = model(input_ids=input_ids, labels=input_ids, return_dict=True).loss
            bpe_loss = (per_token_loss * (input_ids.size(1) - 1)).item()

            # 2. calculate loss of other segmentations
            first_token = inputs.input_ids[0, 0].unsqueeze(0).unsqueeze(0)
            segmentations = get_segmentations(first_token, constraint=text)
            segmentation_ids = segmentations.sequences.to(input_ids.device)
            segmentation_scores = segmentations.sequences_scores.to(input_ids.device)

            # exclude segmentations found by beam search that are equivalent to BPE
            neq_bpe_mask = ~(segmentation_ids[:, : input_ids.size(1)] == input_ids).all(dim=-1)
            neq_bpe_mask = neq_bpe_mask.to(input_ids.device)
            segmentation_ids = segmentation_ids[neq_bpe_mask]
            segmentation_scores = segmentation_scores[neq_bpe_mask]

            # loss returned by beam search is at the token level, so we need to multiply by the length
            segmentation_lens = (segmentation_ids == tokenizer.eos_token_id).nonzero()[:, 1]
            segmentation_loss = -torch.mul(segmentation_scores, segmentation_lens)

        total_loss += bpe_loss
        total_tokens += inputs.attention_mask[..., 1:].sum()
        segmentation_loss = torch.cat(
            (segmentation_loss, torch.tensor([bpe_loss], device=segmentation_loss.device))
        )
        total_corrected_loss += -torch.logsumexp(-torch.tensor(segmentation_loss), dim=0)

        total_bytes += sum(
            len(text) for text in tokenizer.batch_decode(inputs.input_ids, skip_special_tokens=True)
        )

    bits_per_byte = total_loss / torch.log(torch.tensor(2.0)) / total_bytes
    corrected_bits_per_byte = total_corrected_loss / torch.log(torch.tensor(2.0)) / total_bytes
    metrics = {
        "bits_per_byte": bits_per_byte.item(),
        "corrected_bits_per_byte": corrected_bits_per_byte.item(),
        "total_tokens": total_tokens.item(),
        "total_bytes": total_bytes,
        "num_examples": len(eval_data),
        "max_context_length": max_context_length,
    }

    return metrics, examples


"""
model_name_or_path=models/hf_models/OLMo2-7B-pts200k-t180k-ctx2995
step=95972
python -m eval.eval_corrected_bpb \
    --model_name_or_path $model_name_or_path \
    --step $step \
    --output_dir results/bpb/temp \
    --max_num_examples 1
"""


@click.command()
@click.option("--model_name_or_path", type=str, default="OLMo2-7B-npt200k")
@click.option("--step", type=int, default=None)
@click.option("--output_dir", type=str, default="results/bpb/olmo-20k")
@click.option(
    "--eval_dir",
    type=str,
    help="Path to folder with jsonl files",
    default="olmo_data/olmo2_shuffle/olmo2_shuffle",
)
@click.option("--max_num_examples", type=int, default=None)
@click.option("--eval_batch_size", type=int, default=8)
@click.option("--max_context_length", type=int, default=None)
@click.option("--constraint_beam_size", type=int, default=4)
def main(
    model_name_or_path: str,
    step: int,
    output_dir: str,
    eval_dir: str,
    max_num_examples: int,
    eval_batch_size: int,
    max_context_length: int,
    constraint_beam_size: int,
):
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, step=step, padding_side="right")
    max_context_length = max_context_length or model.config.max_position_embeddings
    print(f"Using max context length of {max_context_length}")

    # read .jsonl files from eval_dir until we have max_num_examples
    eval_data = []
    for eval_file in os.listdir(eval_dir):
        nrows = max_num_examples - len(eval_data)
        df = pd.read_json(os.path.join(eval_dir, eval_file), lines=True, compression="zstd", nrows=nrows)
        eval_data.extend(df.text.tolist())
        if len(eval_data) >= max_num_examples:
            break

    eval_data = random.sample(eval_data, max_num_examples)
    print(f"Loaded {len(eval_data)} examples.")

    metrics, examples = get_corrected_bits_per_byte(
        model,
        tokenizer,
        eval_data=eval_data,
        batch_size=eval_batch_size,
        max_context_length=max_context_length,
        constraint_beam_size=constraint_beam_size,
    )

    # print metrics
    for k, v in metrics.items():
        print(f"{k}: {v}")

    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    print(f"Saving results to {output_dir}")

    with open(output_dir / "metrics.json", "w") as fo:
        json.dump(metrics, fo, indent=4)
    pd.DataFrame(examples).to_json(output_dir / "predictions.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    main()
