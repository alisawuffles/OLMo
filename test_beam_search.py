from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import (
    LogitsProcessorList,
    StoppingCriteriaList,
)
from olmo.torch_util import seed_all

seed_all(42)

model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-1124-7B")
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B")
input_text = tokenizer.eos_token
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
print(input_ids)
output_ids = model.generate(input_ids, max_new_tokens=100, do_sample=False)
print(tokenizer.decode(output_ids[0]))


def generate(
    model,
    tokenizer,
    input_ids: Optional[torch.Tensor] = None,
    constraint: str = None,
    max_new_tokens: Optional[int] = 100,
    do_sample: bool = False,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    **kwargs,
):
    """
    constraint: the sequence of text that the model should match exactly
    """
    input_ids = input_ids.to(input_ids.device)
    vocab = tokenizer.get_vocab()

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    eos_token_id_tensor = torch.tensor([tokenizer.eos_token_id]).to(input_ids.device)
    generation_config, model_kwargs = model._prepare_generation_config(model.generation_config, **kwargs)
    inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(
        input_ids, generation_config.bos_token_id, model_kwargs
    )

    batch_size, cur_len = input_ids.shape
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = model._get_initial_cache_position(input_ids, model_kwargs)

    for step in range(max_new_tokens):
        inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = model.forward(**inputs, return_dict=True)
        model_kwargs = model._update_model_kwargs_for_generation(outputs, model_kwargs)
        next_token_logits = outputs.logits[:, -1, :].clone().float()

        # pre-process logits
        if logits_processor:
            next_token_logits = logits_processor(input_ids, next_token_logits)

        # can only consider tokens
        decoded_outputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        for i, decoded_output in enumerate(decoded_outputs):
            assert constraint.startswith(decoded_output), f"{constraint} does not start with {decoded_output}"
            if constraint == decoded_output:
                next_token_logits[i][tokenizer.eos_token_id] = float("inf")
                continue

            valid_next_token_ids = [
                token_id
                for token, token_id in vocab.items()
                if constraint.replace(decoded_output, "").startswith(tokenizer.decode(token_id))
            ]
            print(f"Valid choices for next token: {tokenizer.convert_ids_to_tokens(valid_next_token_ids)}")
            mask = torch.zeros_like(next_token_logits[i], dtype=torch.bool)
            mask[valid_next_token_ids] = True
            next_token_logits[i][~mask] = float("-inf")

        # decode
        if do_sample:
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_logits, dim=-1)

        next_tokens = next_tokens * unfinished_sequences + tokenizer.pad_token_id * (1 - unfinished_sequences)

        # update model inputs for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        # stopping criteria
        if stopping_criteria and stopping_criteria(input_ids, None):
            break

        # if eos_token was found in one sentence, set sentence to finished
        unfinished_sequences = unfinished_sequences.mul(
            next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
        )

        # stop when each sentence is finished
        if unfinished_sequences.max() == 0:
            break

        del outputs

    return input_ids


constraint = (
    "Lexical analysis is the conversion of a text into meaningful lexical tokens based on a lexical grammar."
)
output_ids = generate(
    model,
    tokenizer,
    input_ids,
    constraint=constraint,
    max_new_tokens=100,
    do_sample=False,
)
print(tokenizer.convert_ids_to_tokens(output_ids[0]))
# print(tokenizer.decode(output_ids[0]))
print(tokenizer.convert_ids_to_tokens(tokenizer.encode(tokenizer.eos_token + constraint)))
