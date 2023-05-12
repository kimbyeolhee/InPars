import os
import ftfy
import pandas as pd
from tqdm.auto import tqdm

def load_corpus(dataset_name, source='ir_datasets'):
    texts = []
    docs_ids = []

    if not os.path.exists("data"):
        os.makedirs("data")

    if os.path.isfile(f"data/{dataset_name}.csv"):
        return pd.read_csv(f"data/{dataset_name}.csv")
    else:
        if source == 'ir_datasets':
            import ir_datasets

            identifier = f'beir/{dataset_name}'
            if identifier in ir_datasets.registry._registered:
                dataset = ir_datasets.load(identifier)
            else:
                dataset = ir_datasets.load(dataset_name)
            
            for doc in tqdm(dataset.docs_iter(), total=dataset.docs_count(), desc="Loading docs from ir-datasets"):
                texts.append(
                    ftfy.fix_text(
                        f"{doc.title} {doc.text}" if "title" in dataset.docs_cls()._fields else doc.text
                    )
                )
                docs_ids.append(doc.doc_id)
        
        else:
            raise NotImplementedError(f"Source {source} not implemented")
        df = pd.DataFrame({"doc_id" : docs_ids, "text": texts})
        df.to_csv(f"data/{dataset_name}.csv", index=False)

        return df


def load_queries(dataset_name, source="ir_datsets"):
    queries = {}

    if source == "ir_datasets":
        import ir_datasets
        dataset = ir_datasets.load(f"beir/{dataset_name}")

        for query in dataset.queries_iter():
            queries[query.query_id] = ftfy.fix_text(query.text)
    else:
        from pyserini.search import get_topics

        for (qid, data) in get_topics(f"beir-v1.0.0-{dataset_name}-test").items():
            queries[str(qid)] = ftfy.fix_text(data["title"])
    
    return queries


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="msmarco-passage")
    parser.add_argument("--source", type=str, default="ir_datasets")
    parser.add_argument("--output", type=str, default="data/msmarco-passage.csv")
    args = parser.parse_args()

    df = load_corpus(args.dataset, args.source)
    df.to_csv(args.output, index=False)