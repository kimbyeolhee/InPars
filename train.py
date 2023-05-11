import torch
from dataset import load_dataset
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    set_seed,
)


def split_triples(triples):
    examples = {"label": [], "text": []}
    for i in range(len(triples["query"])):
        examples["text"].append(
            f'Query: {triples["query"][i]} Document: {triples["positive"][i]} Relevant:'
        )
        examples["label"].append("true")
        examples["text"].append(
            f'Query: {triples["query"][i]} Document: {triples["negative"][i]} Relevant:'
        )
        examples["label"].append("false")
    return examples


if __name__ == "__main__":
    # args

    # load model, tokenizer

    # load dataset and preprocessing for model

    # load trainer

    # train

    print("Done")
