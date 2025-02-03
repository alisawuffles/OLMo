from eval.arc.run_eval import evaluate_arc
from eval.boolq.run_eval import evaluate_boolq
from eval.copa.run_eval import evaluate_copa
from eval.coqa.run_eval import evaluate_coqa
from eval.drop.run_eval import evaluate_drop
from eval.hellaswag.run_eval import evaluate_hellaswag
from eval.hotpotqa.run_eval import evaluate_hotpotqa
from eval.jeopardy.run_eval import evaluate_jeopardy
from eval.lambada.run_eval import evaluate_lambada
from eval.mmlu.run_eval import evaluate_mmlu
from eval.openbookqa.run_eval import evaluate_openbookqa
from eval.squad.run_eval import evaluate_squad
from eval.tofu.run_eval import evaluate_tofu
from eval.triviaqa.run_eval import evaluate_triviaqa
from eval.wikidataqa.run_eval import evaluate_wikidataqa
from eval.winogrande.run_eval import evaluate_winogrande


EVALUATION_CONFIGS = {
    "ARC-Easy": {
        "path": "olmo_data/eval/arc-easy/test.jsonl",
        "eval_func": evaluate_arc,
        "num_incontext_examples": 5,
        "eval_batch_size": 64,
        "max_num_examples": 1000,
    },
    "ARC-Challenge": {
        "path": "olmo_data/eval/arc-challenge/test.jsonl",
        "eval_func": evaluate_arc,
        "num_incontext_examples": 5,
        "eval_batch_size": 64,
        "max_num_examples": 1000,
    },
    "BoolQ": {
        "path": "olmo_data/eval/boolq/validation.jsonl",
        "eval_func": evaluate_boolq,
        "num_incontext_examples": 5,
        "eval_batch_size": 32,
        "max_num_examples": 1000,
    },
    "COPA": {
        "path": "olmo_data/eval/copa/test.jsonl",
        "eval_func": evaluate_copa,
        "num_incontext_examples": 5,
        "eval_batch_size": 64,
        "max_num_examples": 1000,
    },
    "CoQA": {
        "path": "olmo_data/eval/coqa/coqa-dev-v1.0.json",
        "eval_func": evaluate_coqa,
        "num_incontext_examples": None,
        "eval_batch_size": 32,
        "max_num_examples": 300,
    },
    "DROP": {
        "path": "olmo_data/eval/drop/validation.jsonl",
        "eval_func": evaluate_drop,
        "num_incontext_examples": 5,
        "eval_batch_size": 16,
        "max_num_examples": 1000,
    },
    "HellaSwag": {
        "path": "olmo_data/eval/hellaswag/validation.jsonl",
        "eval_func": evaluate_hellaswag,
        "num_incontext_examples": 5,
        "eval_batch_size": 32,
        "max_num_examples": 1000,
    },
    "HotpotQA": {
        "path": "olmo_data/eval/hotpotqa/hotpot_dev_distractor_v1.json",
        "eval_func": evaluate_hotpotqa,
        "num_incontext_examples": 5,
        "eval_batch_size": 32,
        "max_num_examples": 1000,
    },
    "Jeopardy": {
        "path": "olmo_data/eval/jeopardy/test.jsonl",
        "eval_func": evaluate_jeopardy,
        "num_incontext_examples": 5,
        "eval_batch_size": 128,
        "max_num_examples": 1000,
    },
    "lambada": {
        "path": "olmo_data/eval/lambada/test.jsonl",
        "eval_func": evaluate_lambada,
        "num_incontext_examples": 5,
        "eval_batch_size": 64,
        "max_num_examples": 1000,
    },
    "mmlu": {
        "path": "olmo_data/eval/mmlu/test.jsonl",
        "eval_func": evaluate_mmlu,
        "num_incontext_examples": 5,
        "eval_batch_size": 64,
        "max_num_examples": 1000,
    },
    "openbookqa": {
        "path": "olmo_data/eval/openbookqa/test.jsonl",
        "eval_func": evaluate_openbookqa,
        "num_incontext_examples": 5,
        "eval_batch_size": 64,
        "max_num_examples": 1000,
    },
    "squad": {
        "path": "olmo_data/eval/squad/validation.jsonl",
        "eval_func": evaluate_squad,
        "num_incontext_examples": 5,
        "eval_batch_size": 32,
        "max_num_examples": 1000,
    },
    "tofu": {
        "path": "olmo_data/eval/tofu/world_facts.jsonl",
        "eval_func": evaluate_tofu,
        "num_incontext_examples": 5,
        "eval_batch_size": 128,
        "max_num_examples": 1000,
    },
    "triviaqa": {
        "path": "olmo_data/eval/triviaqa/validation.jsonl",
        "eval_func": evaluate_triviaqa,
        "num_incontext_examples": 5,
        "eval_batch_size": 128,
        "max_num_examples": 1000,
    },
    "wikidataqa": {
        "path": "olmo_data/eval/wikidataqa/task.json",
        "eval_func": evaluate_wikidataqa,
        "num_incontext_examples": 5,
        "eval_batch_size": 128,
        "max_num_examples": 1000,
    },
    "winogrande": {
        "path": "olmo_data/eval/winogrande/dev.jsonl",
        "eval_func": evaluate_winogrande,
        "num_incontext_examples": 5,
        "eval_batch_size": 128,
        "max_num_examples": 1000,
    },
}
