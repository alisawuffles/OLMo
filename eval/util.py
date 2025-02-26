import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi
import numpy as np
from pathlib import Path
from olmo.util import ensure_dir
import json
import pandas as pd
from string import ascii_uppercase


def prep_incontext_examples(test_df, num_incontext_examples):
    indices = np.arange(len(test_df))
    incontext_indices = {
        i: np.random.choice(indices[indices != i], size=num_incontext_examples, replace=False)
        for i in tqdm(indices, desc="Precomputing in-context examples")
    }
    return incontext_indices


def parse_number(output_str, output_type="int"):
    output_str = output_str.strip().replace(",", "")
    output_num = None
    try:
        if output_type == "int":
            output_num = int(output_str)
        elif output_type == "float":
            output_num = float(output_str)
    except ValueError:
        print(f"Failed to parse number: {output_str}")
        pass
    return output_num


def format_example(
    question, passage=None, choices=None, answer=None, qa_format="qnan", question_prefix="Question:"
):
    """Options for QA format:
    qa: Question: {question}\nAnswer: {answer}
    qnan: Question:\n{question}\nAnswer:\n{answer}
    qna: Question:\n{question}\nAnswer: {answer}
    q: Question: {question} (if answer=None, else equivalent to qa)
    """
    text = ""
    if passage:
        text += f"{passage.strip()}\n\n"

    text += question_prefix + "\n" if "qn" in qa_format else question_prefix + " "
    text += question.strip() + "\n"

    if choices:
        for label, choice in zip(ascii_uppercase, choices):
            text += f"{label}. {choice.strip()}\n"

    answer_prefix = "Answer:"
    if answer or qa_format != "q":
        text += answer_prefix + "\n" if "an" in qa_format else answer_prefix
    if answer:
        if isinstance(answer, str):
            answer = answer.strip()
        answer = str(answer)
        text += answer if "an" in qa_format else " " + answer

    return text


def parse_mc_pred(output, num_options=4, qa_format="qnan"):
    """
    Parses the predicted MC option (e.g., "A") from the model output.
    Returns None if the output is not a valid MC option.
    """
    parsed_answer = None
    valid = True
    if qa_format == "q":
        if output.startswith("Answer:"):  # output answer should start with "Answer: "
            output = output.replace("Answer: ", "")
        else:
            valid = False
    elif qa_format in ["qa", "qna"]:
        if output.startswith(" "):  # output answer should start with leading space
            output = output.lstrip()
        else:
            valid = False

    if output and valid and (output[0] in ascii_uppercase[:num_options]):
        parsed_answer = output[0]

    return parsed_answer


def get_checkpoints(model_name):
    refs = HfApi().list_repo_refs(model_name)
    checkpoints = []
    for branch in refs.branches:
        checkpoints.append(branch.name)
    return checkpoints


def batched_generate(prompts, model, tokenizer, batch_size=1, **generation_kwargs):
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
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
            tokenizer=tokenizer,
            **generation_kwargs,
        )
        batch_generations = tokenizer.batch_decode(batch_outputs.sequences, skip_special_tokens=True)
        # remove the prompt from the generation
        batch_generations = [gen[len(prompt) :] for prompt, gen in zip(batch_prompts, batch_generations)]
        generations.extend(batch_generations)
        pbar.update(len(batch_prompts))
    return generations


def load_model_and_tokenizer(model_name_or_path, tokenizer_name_or_path=None, step=None, padding_side="left"):
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

    # when model is too small, need to limit the number of visible devices
    # for some reason the device mapping doesn't work for small models on lots of GPUs
    if "1B" in model_name_or_path:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        revision=revision if "allenai" in model_name_or_path else None,
    )
    model.eval()

    print(f"Loading tokenizer from {tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    tokenizer.backend_tokenizer.model.dropout = 0.0  # always use dropout p = 0.0 for inference
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = padding_side

    return model, tokenizer


def write_results(results, output_dir, metric="accuracy", print_metrics=False):
    metrics = {"num_examples": len(results), "accuracy": np.mean([r["correct"] for r in results])}

    if "valid" in results[0]:
        metrics["valid_answer"] = np.mean([r["valid"] for r in results])

    if "split" in results[0]:
        for split in sorted(set([r["split"] for r in results])):
            split_results = [r for r in results if r["split"] == split]
            metrics[f"{split}_accuracy"] = np.mean([r["correct"] for r in split_results])

    if print_metrics:
        for k, v in metrics.items():
            print(f"{k}: {v}")

    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    print(f"Saving results to {output_dir}")

    with open(output_dir / "metrics.json", "w") as fo:
        json.dump(metrics, fo, indent=4)
    with open(output_dir / "example_prompt.txt", "w") as fo:
        fo.write(results[0]["prompt"])
    pd.DataFrame(results).to_json(output_dir / "predictions.jsonl", orient="records", lines=True)
