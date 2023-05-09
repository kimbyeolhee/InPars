import os
import pandas as pd
from dataset import load_corpus
from inpars import InPars
import argparse
from transformers import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default='EleutherAI/gpt-j-6B')
    parser.add_argument('--prompt', type=str, default="inpars",
                        help="Prompt type to be used during query generation: \
                        inpars, promptagator or custom")
    parser.add_argument('--dataset', default="trec-covid",
                        help="Dataset from BEIR or custom corpus file (CSV or JSONL)")
    parser.add_argument('--dataset_source', default='ir_datasets',
                        help="The dataset source: ir_datasets")
    parser.add_argument('--n_fewshot_examples', type=int, default=3)
    parser.add_argument('--max_doc_length', default=256, type=int, required=False)
    parser.add_argument('--max_query_length', default=200, type=int, required=False)
    parser.add_argument('--max_prompt_length', default=2048, type=int, required=False)
    parser.add_argument('--max_new_tokens', type=int, default=64)
    parser.add_argument('--max_generations', type=int, default=100_000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--revision', type=str, default=None)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--torch_compile', action='store_true')
    parser.add_argument('--tf', action='store_true')
    parser.add_argument('--output', type=str, default="trec-covid-queries.jsonl")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None)
    # parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    set_seed(args.seed)

    dataset = load_corpus(args.dataset, args.dataset_source)
    print(len(dataset))

