import os
import csv
import re
import json
import random
import argparse
from tqdm import tqdm
import subprocess
from transformers import set_seed
from dataset import load_corpus
from pathlib import Path
from pyserini.search.lucene import LuceneSearcher # Traditional lexical search


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="trec-covid-queries-filtered-reranker.jsonl")
    parser.add_argument('--output', type=str, default="trec-covid-triples.tsv")
    parser.add_argument('--dataset', type=str, default="trec-covid")
    parser.add_argument('--dataset_source', default='ir_datasets',
                        help="The dataset source: ir_datasets or pyserini")
    parser.add_argument('--max_hits', type=int, default=1000)
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size retrieval.")
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)

    corpus = load_corpus(args.dataset, args.dataset_source)
    index = f'beir-v1.0.0-{args.dataset}.flat'
    index = "trec-covid-r1-full-text"

    # only using 10000 datasets for testing
    # corpus = corpus[:10000]

    # convert to {"doc_id" : "text"} format
    corpus = dict(zip(corpus["doc_id"], corpus["text"]))

    # if os.path.isdir(index):
    #     searcher = LuceneSearcher(index_dir=index)
    # else:
    #     searcher = LuceneSearcher.from_prebuilt_index(index)
    searcher = LuceneSearcher.from_prebuilt_index(index)
    
    n_no_query = 0
    n_docs_not_found = 0
    n_examples = 0
    queries = []

    with open(args.input) as f, open(f'{Path(args.output).parent}/topics-{args.dataset}.tsv', 'w') as out:
        tsv_writer = csv.writer(out, delimiter='\t', lineterminator='\n')
        for (i, line) in enumerate(f):
            row = json.loads(line.strip())

            if not row["query"]: # skip empty queries
                n_no_query += 1
                continue

            query = " ".join(row["query"].split()) 
            queries.append((query, None, row["doc_id"])) # (query, log_probs, pos_doc_id) # ('What is the genetic risk of KD?', None, '8rxaih7e')
            tsv_writer.writerow([i, query]) # 0	Is Toxocara exposure associated with suicide attempts in psychiatric patients?

    tmp_run = f"{Path(args.output).parent}/tmp-run-{args.dataset}.txt"
    print(f"😰, {Path(args.output).parent}/tmp-run-{args.dataset}.txt")
    if not os.path.exists(tmp_run):
        subprocess.run([
            'python3', '-m', 'pyserini.search.lucene',
                '--threads', '8',
                '--batch-size', str(args.batch_size),
                '--index', index,
                '--topics', f'{Path(args.output).parent}/topics-{args.dataset}.tsv',
                '--output', tmp_run,
                '--bm25',
        ])
    
    results = {}
    with open(tmp_run) as f:
        for line in f:
            qid, _, doc_id, rank, score, ranker =  re.split(r"\s+", line.strip()) # 0 Q0 8120287 1 17.324377 Anserini
            if qid not in results:
                results[qid] = []
            results[qid].append(doc_id)

    with open(args.output, "w") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        for qid in tqdm(results, desc="sampling"):
            hits = results[qid] # list of doc_ids # 1,000 hits per query
            query, log_probs, pos_doc_id = queries[int(qid)] # log_probs is None for now
            n_examples += 1 # 1,000 examples per query
            sampled_ranks = random.sample(range(len(hits)), min(len(hits), args.n_samples + 1)) # sample 100 ranks from 1,000 ranks
            n_samples_so_far = 0

            for (rank, neg_doc_id) in enumerate(hits): # neg_doc_id : 7958030, pos_doc_id : 471hzpyf 
                if rank not in sampled_ranks: 
                    continue

                if neg_doc_id not in corpus:
                    n_docs_not_found += 1
                    continue

                pos_doc_text = corpus[pos_doc_id].replace('\n', ' ').strip()
                neg_doc_text = corpus[neg_doc_id].replace('\n', ' ').strip()

                writer.writerow([query, pos_doc_text, neg_doc_text])
                n_samples_so_far += 1
                if n_samples_so_far >= args.n_samples:
                    break            

    if n_no_query > 0:
        print(f'{n_no_query} lines without queries.')

    if n_docs_not_found > 0:
        print(f'{n_docs_not_found} docs returned by the search engine but not found in the corpus.')
    
    # index 매칭 문제 해결 안될 시 pyserini 라이브러리 말고 그냥 BM25로 직접 계산해보기