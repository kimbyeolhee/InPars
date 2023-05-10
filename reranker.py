import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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
            
