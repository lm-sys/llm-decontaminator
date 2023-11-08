import argparse
import concurrent.futures
import json
import os
import time

from openai import OpenAI

from detect_instruct import datatype_to_instruct

client = OpenAI()

def check_openai_key():
    if not "OPENAI_API_KEY" in os.environ:
        raise Exception("Please set your OPENAI_API_KEY environment variable.")


def detect_contamination(model, question1, question2, instruct):

    retries = 0
    while retries < 30:
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

    for i in range(len(database)):
        database[i]["results"] = results[i]

    with open(output_path, "w") as fout:
        for each in database:
            fout.write(json.dumps(each) + "\n")

    return database




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="LLM Decontaminator")
    parser.add_argument("--model", type=str, default="gpt-4", help="The name of the OpenAI model to use")
    parser.add_argument("--database_path", type=str, required=True, help="The path to the JSONL database file")
    parser.add_argument("--output_path", type=str, required=True, help="The path to the output JSONL file")
    parser.add_argument("--data-type", type=str, default="code", help="The name of the instruction function to use")
    parser.add_argument("--max-workers", type=int, default=4, help="The maximum number of worker threads to use")

    args = parser.parse_args()

    check_openai_key()

    model = args.model
    database_path = args.database_path
    output_path = args.output_path
    data_type = args.data_type
    max_workers = args.max_workers

    instruct = datatype_to_instruct(data_type)

    with open(database_path, "r") as fin:
        database = [json.loads(l) for l in fin]

    # call the llm_detect function with the parsed arguments
    database = llm_detect(model, database, output_path, instruct, max_workers)
    rephrase_num = sum([1 if True in each["results"] else 0 for each in database])

    print("Rephrased {} test cases.".format(rephrase_num))

