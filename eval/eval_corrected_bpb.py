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
        logits_processor = LogitsProcessorList(
            [SurfaceFormConstraintLogitsProcessor(constraint, tokenizer, device=model.device)]
        )
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
        text = "Lexical analysis is the conversion of a text into meaningful lexical tokens based on a lexical grammar. Learn about the stages, categories, and examples of lexical tokens, and the difference between lexical analysis and large language models."
        # text = "I like pie."
        print(text)
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_context_length,
        )
        inputs.attention_mask = inputs.attention_mask.to(model.device)
        inputs.input_ids = inputs.input_ids.to(model.device)
        print(f"inputs.input_ids (BPE segmentation): {inputs.input_ids}")
        first_token = inputs.input_ids[0, 0].unsqueeze(0).unsqueeze(0)
        print(first_token)
        segmentations = get_segmentations(first_token, constraint=text)
        print(f"segmentations.sequences: {segmentations.sequences}")
        # stack inputs.input_ids and segmentation_ids, which will require resizing inputs.input_ids
        # all_input_ids = torch.cat([inputs.input_ids, segmentation_ids], dim=1)

        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)
            logits = outputs.logits.to(model.device)
            labels = inputs.input_ids.clone()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask = inputs.attention_mask[..., :-1].contiguous()

            seq_losses = []
            for seg_ids in segmentations.sequences:
                indices = (seg_ids == tokenizer.eos_token_id).nonzero().squeeze()
                end_idx = indices.item()
                seg_ids = seg_ids[:end_idx]
                print(tokenizer.convert_ids_to_tokens(seg_ids))
                if torch.equal(seg_ids, inputs.input_ids):
                    print("Equivalent to BPE segmentation.")
                    continue
                per_token_loss = model(seg_ids.unsqueeze(0), labels=seg_ids.unsqueeze(0)).loss  # mean reduction
                seq_loss = per_token_loss * seg_ids.size(0)
                print(f"Seq loss: {seq_loss}")
                seq_losses.append(seq_loss)

        # get neg log likelihood
        loss = F.cross_entropy(shift_logits.transpose(1, 2), shift_labels, reduction="none") * shift_attention_mask
        total_loss += loss.sum()
        seq_losses.append(loss.sum())
        total_tokens += shift_attention_mask.sum()

        # corrected bpb
        total_corrected_loss += -torch.logsumexp(-torch.tensor(seq_losses), dim=0)

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
@click.option("--add_bos_token", is_flag=True, default=False)
@click.option("--max_context_length", type=int, default=None)
@click.option("--constraint_beam_size", type=int, default=4)
def main(
    model_name_or_path: str,
    step: int,
    output_dir: str,
    eval_dir: str,
    max_num_examples: int,
    eval_batch_size: int,
    add_bos_token: bool,
    max_context_length: int,
    constraint_beam_size: int,
):
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, step=step, add_bos_token=add_bos_token)
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

    if add_bos_token:
        output_dir += "-bos"
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    print(f"Saving results to {output_dir}")

    with open(output_dir / "metrics.json", "w") as fo:
        json.dump(metrics, fo, indent=4)
    pd.DataFrame(examples).to_json(output_dir / "predictions.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    main()
