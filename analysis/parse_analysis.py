import json
import os
import pandas as pd
import stanza
from transformers import AutoTokenizer
from analysis.constituency_parsing import (
    tokenize,
    parse_constituency_tree,
    collect_words,
    get_path,
    find_lca,
)
from tqdm.auto import tqdm

ngram_size = 4

nlp = stanza.Pipeline(lang="en", processors="tokenize,pos,constituency")

df = pd.read_json(
    "olmo_data/olmo2_shuffle/0000.jsonl.zstd",
    lines=True,
    compression="zstd",
)

tokenizer_path = "models/hf_models/OLMo2-7B-pts200k-t80k-ctx2756-colon/step1000"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

t = os.path.dirname(tokenizer_path).split("-")[-3]
fo = open(f"analysis/constituency_results_{t}_{ngram_size}gram.jsonl", "w")

for i, row in tqdm(df.iterrows(), total=len(df)):
    doc = nlp(row["text"])
    for sentence in doc.sentences:
        sentence_str = sentence.text

        # collect token boundaries to determine whether n-grams cross token boundaries
        tokenized = tokenizer.convert_ids_to_tokens(tokenizer.encode(sentence_str))
        token_boundaries = [0]
        for token in tokenized:
            token_boundaries.append(token_boundaries[-1] + len(token))

        # constituency parse
        tree_str = str(sentence.constituency)
        tokens = tokenize(tree_str)
        root = parse_constituency_tree(tokens)
        word_list = []
        collect_words(root, word_list)
        words = [word.label for word in word_list]
        num_words = len(word_list)

        if num_words != len(sentence.tokens):
            print("Mismatched number of words and tokens")
            print(sentence_str)
            print(f"words ({len(words)}): {words}")
            print(f"tokens ({len(sentence.tokens)}): {[token.text for token in sentence.tokens]}")
            continue

        # collect all paths
        paths = []
        for idx in range(num_words):
            paths.append(get_path(word_list[idx]))

        # find the min level for the first and last word in every n-gram
        min_levels = []
        sentence_start_idx = sentence.tokens[0].start_char
        for idx in range(num_words - (ngram_size - 1)):
            lca = find_lca(paths[idx], paths[idx + (ngram_size - 1)])
            min_level = max(paths[idx].index(lca), paths[idx + (ngram_size - 1)].index(lca))
            a, b = (
                sentence.tokens[idx].start_char - sentence_start_idx,
                sentence.tokens[idx + (ngram_size - 1)].end_char - sentence_start_idx,
            )
            within_single_token = True
            if any([a < boundary < b for boundary in token_boundaries]):
                within_single_token = False
            words = tuple([word_list[idx + i].label for i in range(ngram_size)])
            min_levels.append((words, min_level, within_single_token))

        ex = {
            "sentence": sentence_str,
            "tokenized": tokenized,
            "tree_str": tree_str,
            "min_levels": min_levels,
        }

        fo.write(json.dumps(ex) + "\n")

    if i % 100 == 0:
        fo.flush()
