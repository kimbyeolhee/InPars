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

if __name__ == "__main__":
    # args

    # load model, tokenizer

    # load dataset and preprocessing for model

    # load trainer

    # train

    print("Done")
