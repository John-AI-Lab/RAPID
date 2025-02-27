


from transformers import AutoTokenizer, AutoModelForCausalLM, SinkCache
import time
import torch


from .rag import get_rag_context
from .engine import self_spec, autoregressive


def get_max_gpu_memory():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device)
        return gpu_properties.total_memory
    else:
        return None

def load_model(
    model_name: str = "../../../yarn-mistral-7b-128k",
    device_list="0"
):
    print("Loading tokenizer")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # tok.pad_token = tok.eos_token
    print("Loading model")
    start_time = time.time()

    # max_gpu_memory = get_max_gpu_memory() / (1024 ** 3)
    device_list = [int(x) for x in device_list.split(",")]
    # max_memory = {i: f"{max_gpu_memory}GiB" for i in device_list}
    llm =  AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2",
        device_map="auto",
        # max_memory=max_memory,
        trust_remote_code=True)
    
    llm.eval()
    # llm = LLM(model=model_name, trust_remote_code=True, gpu_memory_utilization=0.95)#, tensor_parallel_size=ngpu)
    print("Time taken:", round(time.time() - start_time))
    return llm, tok  # type: ignore


class SpecEngine:
    def __init__(self, target_model, target_tokenizer, assistant_model, assistant_tokenizer):
        self.model = target_model
        self.tokenizer = target_tokenizer
        self.assistant_model = assistant_model
        self.assistant_tokenizer = assistant_tokenizer

    def get_prefill_outputs(
        self,
        model,
        input_ids,
        past_key_values=None,
    ):
        model_kwargs = {}
        model_kwargs["use_cache"] = True
        model_kwargs["output_attentions"] = False
        model_kwargs["output_hidden_states"] = False
        model_kwargs["return_dict"] = True
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        model_inputs["num_logits_to_keep"] = 1
        with torch.no_grad():
            outputs = model(**model_inputs)
        return outputs

class LongSpecEngine(SpecEngine):
    def __init__(self, target_model, target_tokenizer, assistant_model, assistant_tokenizer, embedding_model, num_to_retrieve=32, max_chunk_token = 128):
        super().__init__(target_model, target_tokenizer, assistant_model, assistant_tokenizer)
        self.embedding_model = embedding_model
        self.num_to_retrieve = num_to_retrieve
        self.max_chunk_token = max_chunk_token

    def generate(
            self,
            query: str,
            long_context: str,
            speculative_margin: float=0.1,
            **generation_kwargs
        ):
        with torch.no_grad():
            retrieved_context = get_rag_context(long_context, query, self.embedding_model, self.tokenizer, self.num_to_retrieve, self.max_chunk_token)
            input_ids = self.tokenizer([long_context + "\n\n" + query], return_tensors="pt", add_special_tokens=False).input_ids.to(self.model.device)
            assistant_input_ids = self.assistant_tokenizer([retrieved_context+ "\n\n" + query], return_tensors="pt", add_special_tokens=False).input_ids.to(self.assistant_model.device)
            
            print(input_ids.shape)
            print(assistant_input_ids.shape)
            
            prefill_start_time = time.time()
            target_past_key_values = self.get_prefill_outputs(self.model, input_ids[:, :-1]).past_key_values
            assistant_past_key_values = self.get_prefill_outputs(self.assistant_model, assistant_input_ids[:, :-1]).past_key_values
            prefill_time = time.time() - prefill_start_time
            
            outputs, latency = self.model.generate(
                input_ids.to(self.model.device), 
                assistant_input_ids=assistant_input_ids.to(self.assistant_model.device), 
                eos_token_id=self.tokenizer.eos_token_id,  
                assistant_model=self.assistant_model, 
                speculative_margin=speculative_margin,
                output_latency=True,
                past_key_values=target_past_key_values,
                assistant_past_key_values=assistant_past_key_values,
                use_cache=True,
                max_new_tokens=100,
                **generation_kwargs
            )
            latency['prefill_time'] = prefill_time
            num_tokens_generated = outputs.shape[1] - input_ids.shape[1]
            tokens_per_second = num_tokens_generated / latency['decoding_time']
            latency['num_tokens_generated'] = num_tokens_generated
            latency['tokens_per_second'] = tokens_per_second
            responses = self.tokenizer.batch_decode(outputs[:,input_ids.shape[1]:], skip_special_tokens=True)[0]
            return responses, latency
        

class StreamingLLMEngine(SpecEngine):
    def __init__(self, target_model, target_tokenizer, assistant_model, assistant_tokenizer):
        super().__init__(target_model, target_tokenizer, assistant_model, assistant_tokenizer)
    
    def generate(self, query: str, use_chat_template=True, max_new_tokens=100, **gen_args):
        with torch.no_grad():
            if use_chat_template:
                messages = [
                    {"role": "system", "content": "You are a knowledgeable person."},
                    {"role": "user", "content": query},
                ]
                
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                text = query

            input_ids = self.tokenizer(
                [text], 
                return_tensors="pt", 
                add_special_tokens=False
            ).input_ids.to(self.model.device)
            
            # if input_ids.shape[1] > 120000:
            #     input_ids = torch.cat([input_ids[:, :60000], input_ids[:, -60000:]], dim=1) 
                
            outputs, latency = self_spec(input_ids, self.model, self.tokenizer, max_new_tokens, gamma=10, **gen_args)
        
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return responses, latency
    
    
class AutoregressiveEngine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def generate(self, query: str, use_chat_template=True, max_new_tokens=100):
        with torch.no_grad():
            if use_chat_template:
                messages = [
                    {"role": "user", "content": query},
                ]
                
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                text = query

            input_ids = self.tokenizer(
                [text], 
                return_tensors="pt", 
                add_special_tokens=False
            ).input_ids.to(self.model.device)
            
            if input_ids.shape[1] > 120000:
                input_ids = torch.cat([input_ids[:, :60000], input_ids[:, -60000:]], dim=1)
                
            outputs, latency = autoregressive(input_ids, self.model, self.tokenizer, max_new_tokens)
        
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return responses, latency