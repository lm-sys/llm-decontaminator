import argparse
import json


def show(database, mode="all"):

    for each in database:
        test_case = each["test"]
        for i, train_case in enumerate(each["train"]):
            if each["results"][i]:
                print(f"Test case:\n{test_case}\n")
                print(f"Train case:\n{train_case}\n")

    rephrase_num = sum([1 if True in each["results"] else 0 for each in database])
    print(f"Rephrase num: {rephrase_num}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Decontaminator")
    parser.add_argument("--database_path", type=str, required=True, help="The path to the JSONL database file")
    parser.add_argument("--mode", type=str, default="all", help="The mode to show")
    args = parser.parse_args()
    database_path = args.database_path
    mode = args.mode

    with open(database_path, "r") as fin:
        database = [json.loads(l) for l in fin]

    show(database, mode)