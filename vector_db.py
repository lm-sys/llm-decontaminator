import argparse
import json
import torch
import os
import random

import openai
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    AutoModel,
)



def read_dataset(r_path):
    print(f"Reading {r_path}...")
    dataset = [json.loads(l)["text"] for l in open(r_path, "r")]
    print("Done.")
    return dataset

def bert_encode(model, data, batch_size=32, device=None):
    return model.encode(data, batch_size=batch_size, show_progress_bar=True, device=device)

def top_k_similarity(train_embs, test_embs, top_k):
    # Compute cosine-similarities
    cosine_scores = util.cos_sim(test_embs, train_embs)
    # Find the top-k most similar train_embs for each test_emb
    top_k_indices = torch.topk(cosine_scores, k=top_k, dim=1).indices

    return top_k_indices


def build_database(model, train_path, test_path, output_path,  top_k=1, batch_size=32, device=None):

    train_cases = read_dataset(train_path)
    test_cases = read_dataset(test_path)
    train_embs = bert_encode(model, train_cases, batch_size=batch_size, device=device)
    test_embs = bert_encode(model, test_cases, batch_size=batch_size, device=device)
    top_k_indices = top_k_similarity(train_embs, test_embs, top_k)

    db = []

    for i, test_case in enumerate(test_cases):
        top_k_cases = [train_cases[index] for index in top_k_indices[i]]
        db.append({"test": test_case, "train": top_k_cases})      

    with open(output_path, "w") as f:
        for each in db:
            f.write(json.dumps(each) + "\n")

    return db





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Build database of top-k similar cases')
    parser.add_argument('--train_path', type=str, required=True, help='Path to train cases')
    parser.add_argument('--test_path', type=str, required=True, help='Path to test cases')
    parser.add_argument('--output_path', type=str, required=True, help='Path to output database')
    parser.add_argument('--bert-model', type=str, default='multi-qa-MiniLM-L6-cos-v1', help='Path to sentence transformer model')
    parser.add_argument('--top_k', type=int, default=1, help='Number of top-k similar cases to retrieve')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for encoding')
    parser.add_argument('--device', type=str, default=None, help='Device to use for encoding (e.g. "cuda:0")')
    args = parser.parse_args()

    model = SentenceTransformer(args.bert_model)
    build_database(model, args.train_path, args.test_path, args.output_path, args.top_k, args.batch_size, args.device)
