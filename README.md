<h1 align="center">InPars</h1>
<div align="center">
  <strong>Inquisitive Parrots for Search</strong>
</div>

Unofficial PyTorch Reimplementation of InPars. Most of codes are from [zetaalphavector/InPars](https://github.com/zetaalphavector/InPars).


## Install 
```
pip install -r requirements.txt
```

## Usage

### generate data from BEIR dataset
```
python InPars.generate.py
```

### filter queries
```
python InPars.filter.py
```

### generate negative examples by retrieving candidate docs with BM25
```
python InPars.generate_triples.py
```

### fine-tuning the model(reranker) using triples file
```
python InPars.train.py
```

### rerank again with using fine-tuned reranker
```
python InPars.rerank.py
```

### evaluate the result
```
python InPars.evaluate.py
```


## References
* InPars : [Paper](https://arxiv.org/abs/2202.05144)