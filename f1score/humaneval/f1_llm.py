import json
import torch
import random
import time

from openai import OpenAI

client = OpenAI()

MAXRETRY = 30

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


def get_programs(r_path):
    dataset = [json.loads(l) for l in open(r_path, "r")]
    programs = [each["text"] for each in dataset]
    return programs

languages = ["python", "c", "js"]

model = "gpt-4"

instruct ="""I will now give you two programs which might be written in different languages. 
Please help me determine if these two questions are the same. 
If they are, please answer 'True', otherwise answer 'False'. Do not respond with anything else.
"""

FP = 0


original_programs = get_programs("data/test/HumanEval.jsonl")
original_programs = original_programs[:100]

rand_programs = random.sample(original_programs, 15)

cnt = 0

for i in range(len(rand_programs)):
    if cnt >= 100:
        break
    for j in range(i + 1, len(rand_programs)):
        cnt += 1
        if detect_contamination(model, rand_programs[i], rand_programs[j], instruct):
            FP += 1
        if cnt >= 100:
            break


te_f1 = compute_f1score(100, FP, 0)

print(f"Test set F1 score: {te_f1}")


for language in languages:

    re_TP = 0
    re_FN = 0

    rephrased_programs = get_programs(f"data/rephrase/humaneval_{language}.jsonl")

    for i in range(len(original_programs)):
        if rephrased_programs[i] == "":
            re_TP += 1
            continue
        if detect_contamination(model, original_programs[i], rephrased_programs[i], instruct):
            re_TP += 1
        else:
            re_FN += 1


    re_f1 = compute_f1score(re_TP, FP, re_FN)
    print(f"Rephrase {language} F1 score: {re_f1}")
