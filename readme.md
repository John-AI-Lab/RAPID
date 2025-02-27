# RAPID: Long-Context Inference with Retrieval-Augmented Speculative Decoding

This repo provides the official implementation of our paper "Long-Context Inference with Retrieval-Augmented Speculative Decoding".







## Updates

- [2024.2.28]  🚀 Release the code of RAPID. 







## Quick Start



```python
from src.model_wrap import LongSpecEngine

target_model, target_tokenizer = # init your target HF model and tokenizer here
assistant_model, assistant_tokenizer = # init your draft HF model and tokenizer here
full_prompt = # long context
rag_prompt = # RAG context

with torch.no_grad():
    longspec_engine = LongSpecEngine(model, tokenizer, assistant_model, assistant_tokenizer)
    outputs, latency = longspec_engine.generate(
                full_prompt,
                rag_prompt,
                speculative_margin=10,
                do_speculative=True,
                max_new_tokens=max_new_tokens,
                top_p=1, 
                temperature=temperature,
                assistant_confidence_threshold=0.4,
                num_assistant_tokens=10,
                spec_alpha=0.1
            )
```









## Evaluation
