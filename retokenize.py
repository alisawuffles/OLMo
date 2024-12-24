import sys
from pathlib import Path

import os
import tempfile
from contextlib import nullcontext
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tqdm.auto as tqdm
from numba import njit
from tokenizers import Tokenizer
from transformers import AutoTokenizer
from olmo.util import ensure_dir

SOURCE_FILE = Path(sys.argv[1])
DEST_FILE = Path(sys.argv[2])
print(f"{SOURCE_FILE} -> {DEST_FILE}")

input_dtype = np.uint32
output_dtype = np.uint32


@njit
def find_first(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx
    return None


def make_memory_tmpfile():
    shm = Path("/dev/shm")
    # file is about to be memory-mapped so using a tmpfs
    # saves us a copy if it is not local to begin with
    return tempfile.NamedTemporaryFile("w+b", prefix="resharder-", **({"dir": shm} if shm.exists() else {}))


def get_file_len(f):
    cur = f.tell()
    f.seek(0, os.SEEK_END)
    out = f.tell()
    f.seek(cur)
    return out


print("Checking target object")
ensure_dir(os.path.dirname(DEST_FILE))
if DEST_FILE.exists():
    print("Target object already exists!")
    sys.exit()

print("Loading tokenizer...")
olmo_tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B")
tokenizer_file = sys.argv[3]
print(f"tokenizer: {tokenizer_file}")
tokenizer = Tokenizer.from_file(tokenizer_file)

vsize = tokenizer.get_vocab_size()
print(f"Vocab size: {vsize}")

data = np.memmap(SOURCE_FILE, dtype=input_dtype)


def split_on_eos(data, progress=True):
    i = 0
    ctx = tqdm.tqdm(total=len(data)) if progress else nullcontext()
    with ctx as pbar:
        while True:
            try:
                (j,) = find_first(data[i:], olmo_tokenizer.eos_token_id)
            except:
                j = len(data) - i - 1
            yield (i, i + j + 1)
            i += j + 1
            pbar.update(j)
            if i >= len(data):
                break


def smart_coalesce(chunks, target_size=4096 * 10):
    cur_a, cur_b = 0, 0
    for a, b in chunks:
        assert cur_b == a
        if b - cur_a >= target_size:
            target_b = cur_a + target_size
            if abs(cur_b - target_b) < abs(b - target_b):
                yield cur_a, cur_b
                cur_a = cur_b
            else:
                yield cur_a, b
                cur_a = b

        cur_b = b
    yield cur_a, cur_b


def process_chunk(chunk):
    a, b = chunk
    decoded = olmo_tokenizer.decode(data[a:b])
    result = np.array(
        tokenizer.encode(decoded).ids,
        dtype=output_dtype,
    )
    return len(decoded.encode()), result


def full_retokenize(data):
    return np.array(tokenizer.encode(olmo_tokenizer.decode(data)).ids, dtype=output_dtype)


def incremental_retokenize(data, progress=True):
    fout = tempfile.TemporaryFile()
    total_bytes = 0
    for chunk in smart_coalesce(split_on_eos(data)):
        byte_count, result_chunk = process_chunk(chunk)
        result_chunk.tofile(fout)
        total_bytes += byte_count
    fout.seek(0)
    return total_bytes, fout


def incremental_retokenize_parallel(data, fout, progress=True):
    total_bytes = 0
    with Pool(4) as p:
        imap = p.imap(process_chunk, smart_coalesce(split_on_eos(data)))
        for byte_count, result_chunk in imap:
            result_chunk.tofile(fout)
            total_bytes += byte_count
    fout.seek(0)
    return total_bytes


print("Retokenizing file...")
with open(DEST_FILE, "wb") as fout:
    total_bytes = incremental_retokenize_parallel(data, fout)
    out_bytes = get_file_len(fout)
    out_tokens = int(out_bytes / output_dtype().nbytes)

print(f"input tokens    : {len(data)}")
print(f"total bytes     : {total_bytes}")
print(f"output tokens   : {out_tokens}")
print(f"bytes per token : {total_bytes / out_tokens}")
print("Done!")
