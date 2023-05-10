from typing import List
from tqdm.auto import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from math import ceil, exp
from utils import chunk

# Based on https://github.com/castorini/pygaggle/blob/f54ae53d6183c1b66444fa5a0542301e0d1090f5/pygaggle/rerank/base.py#L63
prediction_tokens = {
    'castorini/monot5-small-msmarco-10k':   ['▁false', '▁true'],
    'castorini/monot5-small-msmarco-100k':  ['▁false', '▁true'],
    'castorini/monot5-base-msmarco':        ['▁false', '▁true'],
    'castorini/monot5-base-msmarco-10k':    ['▁false', '▁true'],
    'castorini/monot5-large-msmarco':       ['▁false', '▁true'],
    'castorini/monot5-large-msmarco-10k':   ['▁false', '▁true'],
    'castorini/monot5-base-med-msmarco':    ['▁false', '▁true'],
    'castorini/monot5-3b-med-msmarco':      ['▁false', '▁true'],
    'castorini/monot5-3b-msmarco-10k':      ['▁false', '▁true'],
    'castorini/monot5-3b-msmarco':          ['▁false', '▁true'],
    'unicamp-dl/mt5-base-en-msmarco':       ['▁no'   , '▁yes'],
    'unicamp-dl/mt5-base-mmarco-v2':        ['▁no'   , '▁yes'],
    'unicamp-dl/mt5-base-mmarco-v1':        ['▁no'   , '▁yes'],
}


class Reranker:
    def __init__(self, silent=False, batch_size=8, fp16=False, torchscript=False, device=None):
        self.silent = silent
        self.batch_size = batch_size
        self.fp16 = fp16
        self.torchscript = torchscript
        self.device = device
    
    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        return MonoT5Reranker(model_name_or_path, **kwargs)
    
class MonoT5Reranker(Reranker):
    name: str = "MonoT5"

    def __init__(self, 
                 model_name_or_path='castorini/monot5-base-msmarco-10k',
                 token_false = None,
                 token_true = True,
                 torch_compile=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        if not self.device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_args = {}
        if self.fp16:
            model_args["torch_dtype"] = torch.float16
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, **model_args)
        self.torch_compile = torch_compile
        if torch_compile:
            self.model = torch.compile(self.model)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.token_false_id, self.token_true_id = self.get_prediction_tokens(
            model_name_or_path, self.tokenizer, token_false, token_true
        )
    
    def get_prediction_tokens(self, model_name_or_path, tokenizer, token_false=None, token_true=None):
        if not (token_false and token_true):
            if model_name_or_path in prediction_tokens:
                token_false, token_true = prediction_tokens[model_name_or_path]
                token_false_id = tokenizer.get_vocab()[token_false]
                token_true_id = tokenizer.get_vocab()[token_true]
                return token_false_id, token_true_id
            else:
                return self.get_prediction_tokens("castorini/monot5-base-msmarco", self.tokenizer)
        else:
            token_false_id = tokenizer.get_vocab()[token_false]
            token_true_id = tokenizer.get_vocab()[token_true]
            return token_false_id, token_true_id
        
    @torch.no_grad()
    def rescore(self, pairs: List[List[str]]):
        """
        pairs: List of (query, text) pairs
        """
        scores = []

        for batch in tqdm(chunk(pairs, self.batch_size), disable=self.silent, desc="Rescoring", total=ceil(len(pairs) / self.batch_size)):
            prompts = [f"Query: {query} Document: {text} Relevant:" for (query, text) in batch]
            tokens = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.tokenizer.model_max_length,
                pad_to_multiple_of=(8 if self.torch_compile else None)
            ).to(self.device)

            output = self.model.generate(
                **tokens, max_new_tokens=1, return_dict_in_generate=True, output_scores=True
            )
            
            # decode output
            preds = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
            print("😀")
            print(preds)


            batch_scores = output.scores[0]
            batch_scores = batch_scores[:, [self.token_false_id, self.token_true_id]]
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores += batch_scores[:, 1].exp().tolist()
        
        return scores


