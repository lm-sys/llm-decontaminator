import argparse

from sentence_transformers import SentenceTransformer

from detect_instruct import datatype_to_instruct
from llm_detect import llm_detect, check_openai_key
from vector_db import build_database
from show_samples import show

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Build database of top-k similar cases')
    parser.add_argument('--train_path', type=str, required=True, help='Path to train cases')
    parser.add_argument('--test_path', type=str, required=True, help='Path to test cases')
    parser.add_argument('--output_path', type=str, required=True, help='Path to output database')
    parser.add_argument('--bert-model', type=str, default='multi-qa-MiniLM-L6-cos-v1', help='Path to sentence transformer model')
    parser.add_argument('--top_k', type=int, default=1, help='Number of top-k similar cases to retrieve')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for encoding')
    parser.add_argument('--device', type=str, default=None, help='Device to use for encoding (e.g. "cuda:0")')

    parser.add_argument("--model", type=str, default="gpt-4", help="The name of the OpenAI model to use")
    parser.add_argument("--data-type", type=str, default="code", help="The name of the instruction function to use")
    parser.add_argument("--max-workers", type=int, default=2, help="The maximum number of worker threads to use")

    args = parser.parse_args()

    check_openai_key()

    bert_model = SentenceTransformer(args.bert_model)
    database = build_database(bert_model, args.train_path, args.test_path, args.output_path, args.top_k, args.batch_size, args.device)

    instruct = datatype_to_instruct(args.data_type)
    print("Starting LLM detection...")
    database = llm_detect(args.model, database, args.output_path, instruct, max_workers=args.max_workers)

    show(database, mode="all")

    