import io
import json
from pathlib import Path
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor

import zstandard as zstd
from tqdm import tqdm
from functools import partial
from olmo.util import ensure_dir


def count_lines_in_file(file_path):
    if file_path.endswith(".json.zst"):
        with open(file_path, "rb") as fh:
            dctx = zstd.ZstdDecompressor()
            stream_reader = dctx.stream_reader(fh)
            text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")
            line_count = 0
            for _ in text_stream:
                line_count += 1
            return line_count
    elif file_path.endswith(".jsonl"):
        with open(file_path, "r") as fh:
            line_count = 0
            for _ in fh:
                line_count += 1
            return line_count


def process_file(file_path, data_dir):
    data_dir = Path(data_dir)
    file_path = Path(file_path)
    relative_path = file_path.relative_to(data_dir)
    return (str(relative_path), count_lines_in_file(str(file_path)))


def main():
    data_dir = Path("/lustre/share/llmservice_nlp_fm/adlr-nlp-sharing/sprabhumoye")
    # data_dir = Path("olmo_data/dolmino_dclm")
    target_data_size = 5 * 10**10
    ext = ".jsonl"
    # ext = ".json.zst"
    output_dir = Path("olmo_data/nemo_shuffle")
    # output_dir = Path("olmo_data/temp")
    ensure_dir(output_dir)
    files = [str(p) for p in data_dir.rglob(f"*{ext}")]
    file_lengths_path = output_dir / "file_lengths.json"
    # files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".json.zst")]

    # count lines in each file
    if not os.path.exists(file_lengths_path):
        file_lengths = {}
        process_file_with_dir = partial(process_file, data_dir=str(data_dir))
        with ProcessPoolExecutor() as executor:
            tasks = list(
                tqdm(executor.map(process_file_with_dir, files), total=len(files), desc="Counting lines in files")
            )

        for relative_path, count in tasks:
            if count > 0:
                file_lengths[relative_path] = count

        with open(file_lengths_path, "w") as fh:
            json.dump(file_lengths, fh)
    else:
        with open(file_lengths_path, "r") as fh:
            file_lengths = json.load(fh)

    total_lines = sum(file_lengths.values())
    file_probabilities = {f: c / total_lines for f, c in file_lengths.items()}

    # collect document locations
    filenames = list(file_probabilities.keys())
    probs = list(file_probabilities.values())
    n_samples = target_data_size // 100  # assuming each doc has >100 bytes on average is very safe
    file_counts = np.random.multinomial(n_samples, probs)

    print("Collecting document locations")
    data_locations = {}
    for file_idx, count in enumerate(file_counts):
        if count == 0:
            continue
        rel_path = filenames[file_idx]
        line_indices = np.random.choice(file_lengths[rel_path], size=count, replace=False)
        data_locations[rel_path] = line_indices.tolist()

    # read data at those locations
    data = []
    data_size = 0
    pbar = tqdm(total=target_data_size, desc="Reading data")
    for rel_path, line_idxs in data_locations.items():
        line_idxs.sort()
        if ext == ".json.zst":
            fin = open(data_dir / rel_path, "rb")
            dctx = zstd.ZstdDecompressor()
            stream_reader = dctx.stream_reader(fin)
            lines = io.TextIOWrapper(stream_reader, encoding="utf-8")
        elif ext == ".jsonl":
            fin = open(data_dir / rel_path, "r")
            lines = fin

        for i, line in enumerate(lines):
            if i in line_idxs:
                text = json.loads(line)["text"]
                data.append(text)
                data_size += len(text)
                pbar.update(len(text))
                if data_size >= target_data_size:
                    break
        else:
            fin.close()
            continue

        break

    # write subset files to new directory
    np.random.shuffle(data)
    num_files = 10
    for i in range(num_files):
        chunk = data[i * len(data) // num_files : (i + 1) * len(data) // num_files]
        with open(output_dir / f"{i:04d}.jsonl.zstd", "wb") as fo:
            with zstd.ZstdCompressor().stream_writer(fo) as compressor:
                for text in chunk:
                    compressor.write(json.dumps({"text": text}).encode("utf-8") + b"\n")


if __name__ == "__main__":
    main()
