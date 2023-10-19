import argparse
import concurrent.futures
import json
import torch
import pandas as pd
import random
import time
from detect_instruct import datatype_to_instruct

import openai
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    AutoModel,
)


def detect_contamination(model, question1, question2, instruct):
        
    retries = 0
    while retries < 30:
        try:
            prompt = "part1: \{\n" + question1 + "\n\}\npart2: \{\n" + question2 + "\n\}"

            completion = openai.ChatCompletion.create(
                engine=model,
                messages=[
                    {"role": "system", "content": instruct},
                    {"role": "user", "content": prompt}
                ],
                timeout=3,
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


def llm_detect(model, database, output_path, instruct, max_workers=32):
    
    results = []
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, pairs in enumerate(database):
            test_case = pairs["test"]
            case_results = []
            for train_case in pairs["train"]:
                future = executor.submit(detect_contamination, model, test_case, train_case, instruct)
                case_results.append(future)
            futures.append(case_results)

        for case_results in futures:
            results.append([future.result() for future in case_results])

    rephrase_test_num = 0
    for i in range(len(database)):
        database[i]["results"] = results[i]
        rephrase_test_num += 1 if True in results[i] else 0

    with open(output_path, "w") as fout:
        for each in database:
            fout.write(json.dumps(each) + "\n")

    return rephrase_test_num




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="LLM Decontaminator")
    parser.add_argument("--model", type=str, help="The name of the OpenAI model to use")
    parser.add_argument("--database_path", type=str, help="The path to the JSONL database file")
    parser.add_argument("--output_path", type=str, help="The path to the output JSONL file")
    parser.add_argument("--data-type", type=str, help="The name of the instruction function to use")
    parser.add_argument("--max-workers", type=int, default=32, help="The maximum number of worker threads to use")

    args = parser.parse_args()
    model = args.model
    database = args.database
    output_path = args.output
    data_type = args.data_type
    max_workers = args.max_workers

    instruct = datatype_to_instruct(data_type)

    # call the llm_detect function with the parsed arguments
    rephrase_test_num = llm_detect(model, database, output_path, instruct, max_workers)

    print("Rephrased {} test cases.".format(rephrase_test_num))

