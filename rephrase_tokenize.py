"""
Tokenize human eval dataset.
"""
import argparse
import json
import os

import numpy as np
from transformers import AutoTokenizer

from train.mmap_dataset import MmapDataset


def tokenize_one_sample(sample):
    prompt, solution = sample

    prompt_ids = tokenizer(prompt, return_tensors="np").input_ids[0]

    if args.partial:
        prompt_ids = [prompt_ids[i] for i in range(len(prompt_ids)) if i % 2 == 0]

    solution_ids = tokenizer(solution, return_tensors="np").input_ids[0]

    # Remove BOS, add EOS
    solution_ids = np.concatenate((solution_ids[1:], [tokenizer.eos_token_id]))

    token_ids = np.concatenate((prompt_ids, solution_ids))
    is_ignore = np.zeros(token_ids.shape, dtype=np.int8)
    #is_ignore[:prompt_ids.shape[0]] = 1

    return token_ids, is_ignore


def tokenize_dataset(rows):
    token_ids = []
    is_ignore = []
    for row in rows:
        tmp_token_ids, tmp_is_ignore = tokenize_one_sample(row)
        token_ids.append(tmp_token_ids)
        is_ignore.append(tmp_is_ignore)

    return token_ids, is_ignore


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, default=
        os.path.expanduser("~/datasets/humaneval/HumanEval.jsonl"))
    parser.add_argument("--out-file", type=str)
    parser.add_argument(
        "--model-name-or-path", type=str, default="meta-llama/Llama-2-7b-hf"
    )
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    out_file = args.out_file or args.in_file.replace(".jsonl", ".tok")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)

    # Load the dataset
    #dataset = load_dataset("openai_humaneval")["test"]
    dataset = [json.loads(l) for l in open(args.in_file)]
    rows = []
    for row in dataset:
        if "prompt" in row:
            prompt, solution = row["prompt"], row["canonical_solution"]
        else:
            prompt, solution = "", row["text"]
        rows.append((prompt, solution))

    # Tokenize
    token_ids, is_ignore = tokenize_dataset(rows)
    max_seq_len = max(len(x) for x in token_ids)
    print(f"#seq: {len(token_ids)}")
    print(f"max_seq_len: {max_seq_len}")

    mmap_dataset = MmapDataset.create_by_pad_truncate(
        token_ids, is_ignore, args.max_seq_len, padding_value=tokenizer.unk_token_id)
    mmap_dataset.save(out_file)
    print(f"Save to {out_file}")

    if args.debug:
        item = mmap_dataset[0]
        token_ids = item["token_ids"]
        is_ignore = item["is_ignore"]
        padding_mask = item["padding_mask"]
        token_ids[is_ignore] = tokenizer.unk_token_id
        print(tokenizer.decode(token_ids))
