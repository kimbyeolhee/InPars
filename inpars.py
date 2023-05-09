import os
import random
import pandas as pd
import urllib.error
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_examples(corpus, n_fewshot_examples):
    try:
        df = pd.read_json(
            f'https://huggingface.co/datasets/inpars/fewshot-examples/resolve/main/data/{corpus}.json',
            lines=True
        )

        df = df[['query_id', 'doc_id', 'query', 'document']].values.tolist()
        random_examples = random.sample(df, n_fewshot_examples)
        with open('query_ids_to_remove_from_eval.tsv', 'w') as fout:
            for item in random_examples:
                fout.write(f'{item[0]}\t{item[2]}\n')
        return random_examples
    
    except urllib.error.HTTPError:
        return []

class InPars:
    def __init__(
            self,
            base_model='EleutherAI/gpt-j-6B',
            revision=None, # The specific model version to use(branch name, a tag name, or a commit id...)
            corpus='msmarco',
            prompt=None,
            n_fewshot_example=None,
            max_doc_length=None,
            max_query_length=None,
            max_prompt_length=None,
            max_new_tokens=64,
            fp16=False, # Use float16 precision to speed up inference
            int8=False, # Use int8 precision to speed up inference
            device =None,
            tf=False,
            torch_compile=False, # Compile the model with torch.jit.script to speed up inference
            verbose=False # Print out the model input and output
    ):
        self.corpus = corpus
        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length
        self.max_prompt_length = max_prompt_length
        self.max_new_tokens = max_new_tokens
        self.n_fewshot_example = n_fewshot_example
        self.device = device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"revision": revision}
        if fp16:
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["low_cpu_mem_usage"] = True
        if int8:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"

        if fp16 and base_model == "EleutherAI/gpt-j-6B":
            model_kwargs["revision"] = "float16"


        self.model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        if torch_compile:
            self.model = torch.compile(self.model)
        self.model.to(self.device)
        self.model.eval()

        self.fewshot_examples = load_examples(corpus, self.n_fewshot_examples)