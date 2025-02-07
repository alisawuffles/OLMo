import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)
from olmo.torch_util import seed_all

seed_all(42)

model_name_or_path = "models/hf_models/OLMo2-7B-pts200k-t180k-ctx2995/step95972"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


class SurfaceFormConstraintLogitsProcessor(LogitsProcessor):
    """
    [`LogitsProcessor`] which forces the generated sequence of tokens to decode to a specific surface form.

    Args:
        constraint (`str`):
            The surface form that the model should generate
        eos_token_id (`Union[int, List[int], torch.Tensor]`):
            The id(s) of the *end-of-sequence* token.
        device (`str`, *optional*, defaults to `"cpu"`):
            The device to allocate the tensors.
    """

    def __init__(self, constraint: str, eos_token_id: int, device: str = "cpu"):
        self.constraint = constraint
        self.eos_token_id = eos_token_id
        self.vocab = tokenizer.get_vocab()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        decoded_outputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        for i, decoded_output in enumerate(decoded_outputs):
            print(f"Seq {i}: current sequence is {tokenizer.convert_ids_to_tokens(input_ids[i])}")

            # when constraint is satisfied, force the next token to be eos
            if self.constraint == decoded_output:
                print("    Done!")
                mask = torch.zeros_like(scores[i], dtype=torch.bool)
                mask[self.eos_token_id] = True
                scores[i][~mask] = float("-inf")
                continue

            # set logits for all invalid next-tokens to -inf
            valid_next_token_ids = [
                token_id
                for token, token_id in self.vocab.items()
                if self.constraint.replace(decoded_output, "").startswith(tokenizer.decode(token_id))
            ]
            print(f"    Valid next-token choices are {tokenizer.convert_ids_to_tokens(valid_next_token_ids)}")
            mask = torch.zeros_like(scores[i], dtype=torch.bool)
            mask[valid_next_token_ids] = True
            scores[i][~mask] = float("-inf")
            # divide all scores by 2
            scores[i] /= 2.0
        return scores


input_text = tokenizer.eos_token
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
print(input_ids)

constraint = "Lexical analysis is the conversion of a text into meaningful lexical tokens based on a lexical grammar. Learn about the stages, categories, and examples of lexical tokens, and the difference between lexical analysis and large language models."
logits_processor = LogitsProcessorList([SurfaceFormConstraintLogitsProcessor(constraint, tokenizer.eos_token_id)])
model.generation_config.early_stopping = True
output_ids = model.generate(
    input_ids,
    logits_processor=logits_processor,
    num_beams=4,
    num_return_sequences=4,
    max_new_tokens=100,
    do_sample=False,
    synced_gpus=False,
    # temperature=10.0,
)
for o_ids in output_ids:
    print(tokenizer.convert_ids_to_tokens(o_ids))

print(tokenizer.convert_ids_to_tokens(tokenizer.encode(tokenizer.eos_token + constraint)))
