import json
import torch
import pandas as pd
import random
import time

from openai import OpenAI

client = OpenAI()

MAXRETRY = 30

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


def detect_contamination(model, question1, question2, instruct):

    retries = 0
    while retries < MAXRETRY:
        try:
            prompt = "part1: \{\n" + question1 + "\n\}\npart2: \{\n" + question2 + "\n\}"

            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": instruct},
                    {"role": "user", "content": prompt}
                ],
                timeout=3,
                temperature=0.3,
            )

            pred = completion.choices[0].message.content

            if pred == "True":
                return True
            elif pred == "False":
                return False

            raise Exception("Invalid prediction: {}".format(pred))
        except Exception as e:
            print(f"Retrying...{e}")
            
        time.sleep(1)
        retries += 1

    print(f"Failed to get prediction after {retries} retries.")
    return False


model = "gpt-4"

subjects = ["abstract_algebra", "sociology", "high_school_us_history"]
instruct ="I will now give you two questions which might be written in different languages. Please help me determine if these two questions are the same. If they are, please answer 'True', otherwise answer 'False'. Do not respond with anything else."


for subject in subjects:
    original_questions = get_original_questions(f"data/rephrase/{subject}_test.csv")
    rephrase_questions = get_rephrase_english_questions(f"data/rephrase/{subject}_test_rephrase_english_filtered_question.jsonl")
    chinese_questions = get_chinese_questions(f"data/rephrase/{subject}_test_chinese.jsonl")

    original_questions = original_questions[:100]

    re_TP = 0
    ch_TP = 0
    re_FN = 0
    ch_FN = 0
    FP = 0

    rand_questions = random.sample(original_questions, 15)

    cnt = 0
    for i in range(len(rand_questions)):
        if cnt >= 100:
            break
        for j in range(i + 1, len(rand_questions)):
            cnt += 1
            if detect_contamination(model, rand_questions[i], rand_questions[j], instruct):
                FP += 1
            if cnt >= 100:
                break

    for i in range(len(original_questions)):
        if detect_contamination(model, original_questions[i], rephrase_questions[i], instruct) or rephrase_questions[i] == "":
            re_TP += 1
        else:
            re_FN += 1

        if detect_contamination(model, original_questions[i], chinese_questions[i], instruct) or chinese_questions[i] == "":
            ch_TP += 1
        else:
            ch_FN += 1

    re_f1 = compute_f1score(re_TP, FP, re_FN)
    ch_f1 = compute_f1score(ch_TP, FP, ch_FN)
    te_f1 = compute_f1score(100, FP, 0)


    print(f"{subject} test set f1: {te_f1}")
    print(f"{subject} rephrase f1: {re_f1}")
    print(f"{subject} chinese f1: {ch_f1}")
