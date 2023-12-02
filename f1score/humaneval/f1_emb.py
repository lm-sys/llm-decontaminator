import json
import torch
import random

from sentence_transformers import SentenceTransformer


def compute_f1score(TP, FP, FN):
    # Compute precision and recall
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    
    # Compute F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    return f1


def get_embedding(model, text):
    query_embedding = torch.tensor(model.encode(text)).cuda().unsqueeze(0).unsqueeze(0)
    return query_embedding


def get_programs(r_path):
    dataset = [json.loads(l) for l in open(r_path, "r")]
    programs = [each["text"] for each in dataset]
    return programs


languages = ["python", "c", "js"]
model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

threshold = 0.6

original_programs = get_programs("data/test/HumanEval.jsonl")
original_programs = original_programs[:100]


origin_embs = []

for i in range(len(original_programs)):
    origin_embs.append(torch.mean(get_embedding(model, original_programs[i])[0], dim=0))

FP = 0
cnt = 0
rand_embs = random.sample(origin_embs, 15)

for i in range(len(rand_embs)):
    if cnt >= 100:
        break
    for j in range(i + 1, len(rand_embs)):
        cnt += 1
        if torch.cosine_similarity(rand_embs[i], rand_embs[j], dim=0).item() > threshold:
            FP += 1
        if cnt >= 100:
            break

print(FP)

te_f1 = compute_f1score(100, FP, 0)
print(f"Test set F1 score: {te_f1}")

for language in languages:
    rephrased_programs = get_programs(f"data/rephrase/humaneval_{language}.jsonl")

    rephrase_embs = []

    for i in range(len(original_programs)):
        rephrase_embs.append(torch.mean(get_embedding(model, rephrased_programs[i])[0], dim=0))


    re_TP = 0
    re_FN = 0

    for i in range(len(origin_embs)):
        if torch.cosine_similarity(origin_embs[i], rephrase_embs[i], dim=0).item() > threshold:
            re_TP += 1
        else:
            re_FN += 1

    print(re_TP)
    print(re_FN)

    re_f1 = compute_f1score(re_TP, FP, re_FN)

    print(f"Rephrase {language} F1 score: {re_f1}")
