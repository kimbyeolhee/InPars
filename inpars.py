import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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
            fp16=False,
            int8=False,
            device =None,
            tf=False,
            torch_compile=False,
            verbose=False
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
        