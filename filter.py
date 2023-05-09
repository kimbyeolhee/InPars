import json
from tqdm import tqdm
import argparse

def read_synthetic_data(args):
    rows = []
    with open(args.input, "r") as f:
        for line in tqdm(f):
            row = json.loads(line.strip())
            if len(row["log_probs"]) < args.min_tokens:
                continue
            if len(row["log_probs"]) > args.max_tokens:
                continue
            if args.skip_questions_copied_from_context:
                if row["question"].lower() in row["doc_text"].lower():
                    continue
            rows.append(row)
    return rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="trec-covid-queries.jsonl",
                        help="Input jsonl file with the synthetic queries to be filtered.")
    parser.add_argument("--dataset", default="trec-covid", type=str,
                        help="Dataset name from BEIR collection.")
    parser.add_argument('--dataset_source', default='ir_datasets',
                        help="The dataset source: ir_datasets or pyserini")
    parser.add_argument("--filter_strategy", type=str, default="scores",
                        help="Filtering strategy: scores or reranker.")
    parser.add_argument('--keep_top_k', type=int, default=10_000,
                        help='Write only top_k best scored query-doc pairs.')
    parser.add_argument("--output", type=str, default="trec-covid-queries-filtered.jsonl",
                        help="Path to save the filtered queries.")
    parser.add_argument("--model_name_or_path", type=str,
                        default='castorini/monot5-3b-msmarco-10k', required=False,
                        help="Reranker model to be used in case of filtering_strategy=reranker.")
    parser.add_argument('--min_tokens', type=int, default=3,
                        help='Skip question that have fewer than this number of words.')
    parser.add_argument('--max_tokens', type=int, default=1000,
                        help='Skip question that have more than this number of words.')
    parser.add_argument('--skip_questions_copied_from_context', action='store_true',
                        help='If passed, skip questions that were copied from the passage.')
    parser.add_argument("--device", default=None, type=str,
                        help="CPU or CUDA device.")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use FP16 weights during inference.")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size for inference.")

    args = parser.parse_args()
    assert args.filter_strategy in ['scores', 'reranker']

    dataset = read_synthetic_data(args)

    print(dataset[0].keys())  
    print("😎 doc_id : ", dataset[0]["doc_id"])
    print("😎 text : ", dataset[0]["text"])
    print("🙄 query: ", dataset[0]["query"])
    print("😎 log_probs : ", dataset[0]["log_probs"])
    print("😎 prompt_text : ", dataset[0]["prompt_text"])