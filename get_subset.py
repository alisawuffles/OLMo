import io
import json
import orjson
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
    ordered_files = list(file_lengths.keys())

    # Collect document idxs, grouped by file
    num_docs = sum(file_lengths.values())
    print(f"Total number of documents: {num_docs}")
    num_docs_to_sample = target_data_size // 1000  # assuming each doc has > 5000 bytes on average
    global_doc_idxs = np.random.choice(num_docs, size=num_docs_to_sample, replace=False)

    cumulative_M = 0
    file_info = []
    for rel_path in tqdm(ordered_files, desc="Grouping global document indices by file"):
        file_len = file_lengths[rel_path]
        file_start, file_end = cumulative_M, cumulative_M + file_len
        global_doc_idxs_for_file = {idx for idx in global_doc_idxs if file_start <= idx < file_end}
        local_idxs = sorted(idx - file_start for idx in global_doc_idxs_for_file)
        file_info.append((rel_path, local_idxs))
        cumulative_M += file_len

    # read data at those locations
    data = []
    data_size = 0
    pbar = tqdm(total=target_data_size, desc="Reading data")
    for rel_path, local_idxs in file_info:
        if not local_idxs:
            continue
        max_i = local_idxs[-1]  # idx of last document needed in file
        pointer = 0  # track position in local_idxs

        if ext == ".json.zst":
            fin = open(data_dir / rel_path, "rb")
            dctx = zstd.ZstdDecompressor()
            stream_reader = dctx.stream_reader(fin)
            lines = io.TextIOWrapper(stream_reader, encoding="utf-8")
        elif ext == ".jsonl":
            fin = open(data_dir / rel_path, "r")
            lines = fin

        for i, line in enumerate(lines):
            if i > max_i or pointer >= len(local_idxs):  # done reading this file
                break
            if i < local_idxs[pointer]:  # skip until we reach the next document we need
                continue
            if pointer < len(local_idxs) and i == local_idxs[pointer]:
                text = orjson.loads(line).get("text", "")
                data.append(text)
                data_size += len(text)
                pbar.update(len(text))
                pointer += 1
                if data_size >= target_data_size:
                    break
        else:
            fin.close()
            continue

        break

    print(f"Total bytes of data: {data_size}")

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
