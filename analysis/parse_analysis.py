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


nlp = stanza.Pipeline(lang="en", processors="tokenize,pos,constituency")

df = pd.read_json(
    "olmo_data/olmo2_shuffle/0000.jsonl.zstd",
    lines=True,
    compression="zstd",
)

tokenizer_path = "models/hf_models/OLMo2-7B-pts200k-t80k-ctx2756-colon/step1000"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

fo = open("analysis/constituency_results.jsonl", "w")

for i, row in tqdm(df.iterrows()):
    doc = nlp(row["text"])
    for sentence in doc.sentences:
        sentence_str = sentence.text
        tokenized = tokenizer.convert_ids_to_tokens(tokenizer.encode(sentence_str))
        token_boundaries = [0]
        for token in tokenized:
            token_boundaries.append(token_boundaries[-1] + len(token))

        tree_str = str(sentence.constituency)
        tokens = tokenize(tree_str)
        root = parse_constituency_tree(tokens)
        word_list = []
        collect_words(root, word_list)
        words = [word.label for word in word_list]

        # collect all paths
        paths = []
        for idx in range(len(word_list)):
            paths.append(get_path(word_list[idx]))

        # find the min level for every pair of neighboring words
        min_levels = []
        sentence_start_idx = sentence.tokens[0].start_char
        for idx in range(len(word_list) - 1):
            lca = find_lca(paths[idx], paths[idx + 1])
            min_level = max(paths[idx].index(lca), paths[idx + 1].index(lca))
            a, b = (
                sentence.tokens[idx].start_char - sentence_start_idx,
                sentence.tokens[idx + 1].end_char - sentence_start_idx,
            )
            within_single_token = True
            if any([a < boundary < b for boundary in token_boundaries]):
                within_single_token = False
            min_levels.append((word_list[idx].label, word_list[idx + 1].label, min_level, within_single_token))

        ex = {
            "sentence": sentence_str,
            "tokenized": tokenized,
            "tree_str": tree_str,
            "min_levels": min_levels,
        }

        fo.write(f"{ex}\n")

    if i % 100 == 0:
        fo.flush()
