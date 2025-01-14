import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi
import numpy as np


def prep_incontext_examples(test_df, num_incontext_examples):
    indices = np.arange(len(test_df))
    incontext_indices = {
        i: np.random.choice(indices[indices != i], size=num_incontext_examples, replace=False)
        for i in tqdm(indices, desc="Precomputing in-context examples")
    }
    return incontext_indices


def format_example(question, passage=None, choices=None, answer=None):
    text = ""
    if passage:
        text += f"{passage.strip()}\n\n"

    text += f"Question:\n{question.strip()}\n"

    if choices:
        for label, choice in zip("ABCD", choices):
            text += f"{label}. {choice.strip()}\n"
    text += "Answer:\n"
    if answer:
        text += answer.strip()
    return text


def parse_mc_pred(output):
    parsed_answer = None
    if output and output[0] in "ABCD":
        parsed_answer = output[0]
    return parsed_answer


def get_checkpoints(model_name):
    refs = HfApi().list_repo_refs(model_name)
    checkpoints = []
    for branch in refs.branches:
        checkpoints.append(branch.name)
    return checkpoints


def batched_generate(prompts, model, tokenizer, do_sample=False, batch_size=1, max_new_tokens=20):
    generations = []
    pbar = tqdm(total=len(prompts), desc="Generating")
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        batch_inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            add_special_tokens=True,
            padding="longest",
        )
        batch_inputs.input_ids = batch_inputs.input_ids.to(model.device)
        batch_inputs.attention_mask = batch_inputs.attention_mask.to(model.device)
        batch_outputs = model.generate(
            **batch_inputs,
            num_return_sequences=1,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=1.0,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
            tokenizer=tokenizer,  # required for token healing
        )
        batch_generations = tokenizer.batch_decode(batch_outputs.sequences, skip_special_tokens=True)
        # remove the prompt from the generation
        batch_generations = [gen[len(prompt) :] for prompt, gen in zip(batch_prompts, batch_generations)]
        generations.extend(batch_generations)
        pbar.update(len(batch_prompts))
    return generations


def load_model_and_tokenizer(model_name_or_path, tokenizer_name_or_path=None, step=None, add_bos_token=False):
    revision = None
    if os.path.exists(model_name_or_path):
        if step:
            model_name_or_path += f"/step{step}"
    else:
        if step:
            try:
                revision = [r for r in get_checkpoints(model_name_or_path) if r.split("-")[1] == f"step{step}"][0]
                print(f"Revision: {revision}")
            except IndexError:
                raise ValueError(f"Checkpoint {step} not found")

    tokenizer_name_or_path = tokenizer_name_or_path or model_name_or_path

    print(f"Loading model from {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="balanced",
        revision=revision if "allenai" in model_name_or_path else None,
    )
    model.eval()

    print(f"Loading tokenizer from {tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    tokenizer.backend_tokenizer.model.dropout = 0.0  # always use dropout p = 0.0 for inference
    if add_bos_token:
        tokenizer.bos_token = tokenizer.eos_token
        tokenizer.bos_token_id = tokenizer.eos_token_id
        tokenizer.add_bos_token = True
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    return model, tokenizer
