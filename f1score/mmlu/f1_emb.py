import json
import torch
import pandas as pd
import os
import random

from sentence_transformers import SentenceTransformer



def get_rephrase_english_questions(r_path):
    # get all the questions from the rephrase file
    questions = []
    with open(r_path, "r") as fin:
        for line in fin:
            tmp_dict = json.loads(line)
            tmp_dict["text"] = tmp_dict["text"] if tmp_dict["text"] is not None else ""
            question = tmp_dict["text"].split("\nAnswer:")[0]
            questions.append(question)
    return questions

def get_chinese_questions(r_path):
    # get all the questions from the rephrase file
    questions = []
    with open(r_path, "r") as fin:
        for line in fin:
            tmp_dict = json.loads(line)
            question = tmp_dict["text"].split("\n答")[0]
            questions.append(question)
    return questions

def get_original_questions(r_path):
    test_df = pd.read_csv(r_path, header=None)
    questions = [test_df.iloc[idx, 0] for idx in range(test_df.shape[0])]
    return questions


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




subjects = ["abstract_algebra", "sociology", "high_school_us_history"]

model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

threshold = 0.5

for subject in subjects:
    original_questions = get_original_questions(f"data/rephrase/{subject}_test.csv")
    rephrase_questions = get_rephrase_english_questions(f"data/rephrase/{subject}_test_rephrase_english_filtered_question.jsonl")
    chinese_questions = get_chinese_questions(f"data/rephrase/{subject}_test_chinese_filtered_question.jsonl")

    original_questions = original_questions[:100]


    origin_embs = []
    rephrase_embs = []
    chinese_embs = []

    for i in range(len(original_questions)):
        origin_embs.append(torch.mean(get_embedding(model, original_questions[i])[0], dim=0))
        rephrase_embs.append(torch.mean(get_embedding(model, rephrase_questions[i])[0], dim=0))
        chinese_embs.append(torch.mean(get_embedding(model, chinese_questions[i])[0], dim=0))
        

    re_TP = 0
    ch_TP = 0
    re_FN = 0
    ch_FN = 0
    FP = 0

    rand_embs = random.sample(origin_embs, 15)

    cnt = 0
    for i in range(len(rand_embs)):
        if cnt >= 100:
            break
        for j in range(i + 1, len(rand_embs)):
            cnt += 1
            if torch.cosine_similarity(rand_embs[i], rand_embs[j], dim=0).item() > threshold:
                FP += 1
            if cnt >= 100:
                break

    for i in range(len(origin_embs)):
        re_sim = torch.cosine_similarity(origin_embs[i], rephrase_embs[i], dim=0).item()
        if re_sim > threshold or rephrase_questions[i] == "":
            re_TP += 1
        else:
            re_FN += 1

        ch_sim = torch.cosine_similarity(origin_embs[i], chinese_embs[i], dim=0).item()
        if ch_sim > threshold or chinese_questions[i] == "":
            ch_TP += 1
        else:
            ch_FN += 1


    print(FP, re_TP, ch_TP, re_FN, ch_FN)

    re_f1 = compute_f1score(re_TP, FP, re_FN)
    ch_f1 = compute_f1score(ch_TP, FP, ch_FN)
    te_f1 = compute_f1score(100, FP, 0)


    print(f"{subject} test set f1: {te_f1}")
    print(f"{subject} rephrase f1: {re_f1}")
    print(f"{subject} chinese f1: {ch_f1}")
