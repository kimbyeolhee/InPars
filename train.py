import torch
from datasets import load_dataset
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

@dataclass
class ExtraArguments:
    triples: str = field(
        default="trec-covid-triples.tsv",
        metadata={"help": "Triples file contains query, positive and negative documents in TSV format."}
    )
    pairs: str = field(
        default=None,
        metadata={"help": "File containing pairs of query, passage, label(pos/neg) in TSV format."}
    )
    base_model: str = field(
        default="t5-base",
        metadata={"help" : "Base model for fine-tuning"}

    )
    max_doc_length: int = field(
        default=300,
        metadata={"help": "Maximum doc length. if doc length is greater than this, it will be truncated."}
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

def tokenize(batch):
    tokenized = tokenizer(
        batch['text'],
        padding=True,
        truncation=True,
        max_length=args.max_doc_length,
    )
    tokenized["labels"] = tokenizer(batch["label"])['input_ids']
    return tokenized

if __name__ == "__main__":
    parser = HfArgumentParser((Seq2SeqTrainingArguments, ExtraArguments))
    training_args, args = parser.parse_args_into_dataclasses()
    training_args.evaluation_strategy = "no"
    training_args.do_eval = False
    training_args.output_dir = "./reranker/"
    training_args.max_steps = 156
    set_seed(training_args.seed)

    total_examples = None
    if training_args.max_steps > 0:
        total_examples = training_args.gradient_accumulation_steps * training_args.per_device_train_batch_size * training_args.max_steps * torch.cuda.device_count()


    # load model, tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # load dataset and preprocessing for model
    if args.triples:
        dataset = load_dataset(
            "csv",
            data_files=args.triples,
            sep="\t",
            names=("query", "positive", "negative"),
        )
        dataset = dataset.map(
            split_triples,
            remove_columns=("query", "positive", "negative"),
            batched=True,
        )
    else:
        raise Exception("triple file is required")

    if total_examples:
        dataset["train"] = dataset["train"].shuffle().select(range(total_examples))
    dataset = dataset.map(tokenize, remove_columns=("text", "label"), batched=True, desc="tokenizing")

    # load trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=DataCollatorForSeq2Seq(tokenizer),
    )

    # train
    train_metrics = trainer.train()
    trainer.save_model(training_args.output_dir)
    trainer.save_state()
    trainer.save_metrics("train", train_metrics.metrics)

    print("Done")
