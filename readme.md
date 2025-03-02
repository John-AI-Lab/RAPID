# RAPID: Long-Context Inference with Retrieval-Augmented Speculative Decoding

This repo provides the official implementation of our paper "Long-Context Inference with Retrieval-Augmented Speculative Decoding".





## Updates

- [2024.2.28]  🚀 Release the paper and code of RAPID. 







## Highlights

- Using RAG to speculate (and then accelerate) the generation of long-context LLMs.
- Using retrieval-augmented target distribution to incorporate benefits from both RAG and long-context LLMs.
- Got 2x speedup with performance improvements when using self-speculation.







## Quick Start

1. Retrieval relevant chunks from a long context for a query. We provide a simple and effective RAG pipeline based on [BGE-M3](https://huggingface.co/BAAI/bge-m3) embedding model.

```python
from src.rag import get_rag_context
import math


long_context = ""
query = ""

target_rag_length = 8192
token_per_chunk = 512
num_to_retrieve = math.ceil(target_rag_length / token_per_chunk)

rag_context, num_rag_chunks = get_rag_context(
  long_context,
  query,
  embed_model_path="BAAI/bge-m3",
  tokenizer=tokenizer,
  num_to_retrieve=num_to_retrieve,
  max_chunk_token=token_per_chunk,
  rag_threshold=0.3
)
```

2. We implemented RAPID by wrapping the huggingface transformers generate with a monkey patch:

```python
from src.utils import generate, _get_candidate_generator, _assisted_decoding
import transformers


transformers.generation.utils.GenerationMixin.generate = generate
transformers.generation.utils.GenerationMixin._get_candidate_generator = _get_candidate_generator
transformers.generation.utils.GenerationMixin._assisted_decoding = _assisted_decoding
```



3. Now RAPID can be utilized by calling model.generate:

```python
from src.utils import load_model_tokenizer

target_model, target_tok = load_model_tokenizer(
  "meta-llama/Llama-3.1-8B-Instruct",
  device_list="0"
)
draft_model, draft_tok = target_model, target_tok # using self-sepcualtion

# using upward-speculation
"""
draft_model, draft_tok = load_model_tokenizer(
  "meta-llama/Llama-3.1-70B-Instruct",
  device_list="1,2,3,4,5,6,7" # depend on your GPUs
)
"""
long_input = long_context + "\n" + query # also make your instructions here
rag_input = rag_context + "\n" + query # also make your instructions here

input_ids = self.tokenizer([long_input], return_tensors="pt", add_special_tokens=False).input_ids.to(target_model.device)
draft_input_ids = self.assistant_tokenizer([rag_input], return_tensors="pt", add_special_tokens=False).input_ids.to(draft_model.device)
            
outputs = target_model.generate(
    input_ids, 
  	assistant_input_ids=draft_input_ids, 
    eos_token_id=target_tok.eos_token_id,  
    assistant_model=draft_model, 
    speculative_margin=10, # \eta in Eq. (6) 
    use_cache=True,
  	max_new_tokens=1024,
    **generation_kwargs
    # do_sample=False, 
    # top_p=1, 
    # top_k=-1, 
    # temperature=1,
)
responses = self.tokenizer.batch_decode(outputs[:,input_ids.shape[1]:], skip_special_tokens=True)[0]
print(responses)
```







## Evaluation



### LongBench V2

- Evaluate with temperature=0.1 following official settings. To avoid the [randomness issue](https://github.com/THUDM/LongBench/issues/94) when evaluating with vllm in official repo, we provide an evaluation script based on transformers generate.

 



##### LLaMA-3.1 Series

| Target Model           | Draft Model            | η    | CoT  | Overal | Easy | Hard | Short | Medium | Long |
| ---------------------- | ---------------------- | ---- | ---- | ------ | ---- | ---- | ----- | ------ | ---- |
| LLaMA-3.1-8B-Instruct  | -                      | -    | ❌    | 28.0   | 29.2 | 27.3 | 33.3  | 25.1   | 25.0 |
| LLaMA-3.1-8B-Instruct  | LLaMA-3.1-8B-Instruct  | 10   | ❌    | 32.4   | 34.9 | 30.9 | 37.8  | 29.8   | 28.7 |
| LLaMA-3.1-8B-Instruct  | -                      | -    | ✅    | 30.4   | 35.4 | 27.3 | 36.7  | 27.9   | 25.0 |
| LLaMA-3.1-8B-Instruct  | LLaMA-3.1-8B-Instruct  | 10   | ✅    | 34.2   | 39.1 | 31.2 | 41.1  | 31.2   | 28.7 |
| LLaMA-3.1-70B-Instruct | -                      | -    | ❌    | 31.6   | 32.3 | 31.2 | 41.1  | 27.4   | 24.1 |
| LLaMA-3.1-8B-Instruct  | LLaMA-3.1-70B-Instruct | 50   | ❌    | 38.8   | 40.6 | 37.6 | 37.8  | 38.1   | 41.7 |
| LLaMA-3.1-70B-Instruct | LLaMA-3.1-70B-Instruct | 20   | ❌    | 40.2   | 42.2 | 38.9 | 42.8  | 37.2   | 41.7 |
| LLaMA-3.1-70B-Instruct | -                      | -    | ✅    | 36.2   | 35.9 | 36.3 | 45.0  | 34.0   | 25.9 |
| LLaMA-3.1-8B-Instruct  | LLaMA-3.1-70B-Instruct | 40   | ✅    | 40.2   | 44.3 | 37.6 | 41.1  | 37.7   | 43.5 |
| LLaMA-3.1-70B-Instruct | LLaMA-3.1-70B-Instruct | 20   | ✅    | 40.2   | 45.3 | 37.0 | 44.4  | 36.3   | 40.7 |



##### Qwen2.5 Series

| Target Model         | Draft Model          | η    | CoT  | Overal   | Easy     | Hard     | Short    | Medium   | Long     |
| -------------------- | -------------------- | ---- | ---- | -------- | -------- | -------- | -------- | -------- | -------- |
| Qwen2.5-7B-Instruct  | -                    | -    | ❌    | 30.2     | 31.2     | 29.6     | **41.7** | 24.7     | 22.2     |
| Qwen2.5-7B-Instruct  | Qwen2.5-7B-Instruct  | 20   | ❌    | **32.0** | **35.4** | **29.9** | 40.6     | **27.4** | **26.9** |
| Qwen2.5-7B-Instruct  | -                    | -    | ✅    | 33.2     | 36.5     | 31.2     | **46.7** | 24.2     | **28.7** |
| Qwen2.5-7B-Instruct  | Qwen2.5-7B-Instruct  | 5    | ✅    | **35.4** | **40.6** | **32.2** | 42.2     | **33.0** | **28.7** |
| Qwen2.5-72B-Instruct | -                    | -    | ❌    | 40.0     | 41.7     | 38.9     | 42.2     | 37.2     | 41.7     |
| Qwen2.5-7B-Instruct  | Qwen2.5-72B-Instruct | 50   | ❌    | 35.6     | 38.5     | 33.8     | 42.2     | 30.2     | 35.2     |
| Qwen2.5-72B-Instruct | Qwen2.5-72B-Instruct | 20   | ❌    | **42.9** | **44.3** | **42.1** | **48.9** | **37.7** | **43.5** |
| Qwen2.5-72B-Instruct | -                    | -    | ✅    | 43.9     | **49.5** | 40.5     | 46.7     | 40.5     | **46.3** |
| Qwen2.5-7B-Instruct  | Qwen2.5-72B-Instruct | 50   | ✅    | 41.2     | 41.7     | 40.8     | 43.3     | 36.7     | **46.3** |
| Qwen2.5-72B-Instruct | Qwen2.5-72B-Instruct | 20   | ✅    | **44.1** | 45.3     | **43.4** | **47.2** | **42.8** | 41.7     |





### InfiniteBench

- Evaluate with top_p=1, temperature=1.

##### LLaMA-3.1 Series

| Target Model           | Draft Model            | η    | En.QA     | En.MC     | En.Sum    | AVG.      |
| ---------------------- | ---------------------- | ---- | --------- | --------- | --------- | --------- |
| LLaMA-3.1-8B-Instruct  | -                      | -    | 34.58     | 53.28     | 30.14     | 39.33     |
| LLaMA-3.1-8B-Instruct  | LLaMA-3.1-8B-Instruct  | 10   | **34.90** | **63.32** | **30.27** | **42.83** |
| LLaMA-3.1-70B-Instruct | -                      | -    | 36.48     | 68.56     | **30.18** | 45.07     |
| LLaMA-3.1-8B-Instruct  | LLaMA-3.1-70B-Instruct | 10   | **40.94** | 79.04     | 29.96     | 49.98     |
| LLaMA-3.1-70B-Instruct | LLaMA-3.1-70B-Instruct | 10   | 40.56     | **81.66** | 29.64     | **50.62** |

##### Qwen2.5 Series

| Target Model         | Draft Model          | η    | En.QA     | En.MC     | En.Sum    | AVG.      |
| -------------------- | -------------------- | ---- | --------- | --------- | --------- | --------- |
| Qwen2.5-7B-Instruct  | -                    | -    | 16.93     | 66.81     | 30.62     | 38.12     |
| Qwen2.5-7B-Instruct  | Qwen2.5-7B-Instruct  | 20   | **19.81** | **75.98** | **31.64** | **42.48** |
| Qwen2.5-72B-Instruct | -                    | -    | 39.21     | 81.66     | 32.45     | 51.11     |
| Qwen2.5-7B-Instruct  | Qwen2.5-72B-Instruct | 20   | 30.10     | 83.84     | 32.21     | 48.72     |
| Qwen2.5-72B-Instruct | Qwen2.5-72B-Instruct | 10   | **40.52** | **85.59** | **32.94** | **53.02** |





## Citation

If you find our paper useful, hope you can star our repo and cite our paper as follows:

```
@article{chen2025longcontextinferenceretrievalaugmentedspeculative,
      title={Long-Context Inference with Retrieval-Augmented Speculative Decoding}, 
      author={Guanzheng Chen and Qilong Feng and Jinjie Ni and Xin Li and Michael Qizhe Shieh},
      year={2025},
      eprint={2502.20330},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.20330}, 
}
```



