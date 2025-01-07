"""
by Domino we really mean stage 1 pretraining data for Olmo 2
"""

from pathlib import Path
import os
import pandas as pd
from tqdm import tqdm


corpus_dir = Path("olmo_data/dolmino_shuffle")
eval_file_ids = [int(l) for l in open(corpus_dir / "eval_file_ids.txt").read().splitlines()]
all_files = [f for f in os.listdir(corpus_dir) if f.endswith(".jsonl.zst")]

for filename in tqdm(all_files):
    df = pd.read_json(corpus_dir / filename, lines=True, compression="zstd")
    file_id = int(filename[: -len(".jsonl.zst")])
    if file_id in eval_file_ids:
        # eval files as jsonl for bpb evals
        df.to_json(corpus_dir / f"eval/{file_id}.jsonl", orient="records", lines=True)
    else:
        # train files as txt for training tokenizers
        text = "\n".join(df.text)
        with open(corpus_dir / f"train/{file_id}.txt", "w") as f:
            f.write(text)
