


from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import torch


from .rag import get_rag_context


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

    max_gpu_memory = get_max_gpu_memory() / (1024 ** 3)
    device_list = [int(x) for x in device_list.split(",")]
    max_memory = {i: f"{max_gpu_memory}GiB" for i in device_list}
    llm =  AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        use_flash_attention_2=True, 
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True)
    # llm = LLM(model=model_name, trust_remote_code=True, gpu_memory_utilization=0.95)#, tensor_parallel_size=ngpu)
    print("Time taken:", round(time.time() - start_time))
    return llm, tok  # type: ignore






class LongSpecEngine:
    def __init__(self, target_model, target_tokenizer, assistant_model, assistant_tokenizer):
        self.model = target_model
        self.tokenizer = target_tokenizer
        self.assistant_model = assistant_model
        self.assistant_tokenizer = assistant_tokenizer
        

    def get_prefill_outputs(
        self,
        model,
        input_ids,
        
    ):
        model_kwargs = {}
        model_kwargs["use_cache"] = True
        model_kwargs["output_attentions"] = False
        model_kwargs["output_hidden_states"] = False
        model_kwargs["return_dict"] = True
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        # if "num_logits_to_keep" in model_inputs:
        model_inputs["num_logits_to_keep"] = 1
        outputs = model(**model_inputs)
        return outputs




    @torch.inference_mode()
    def prefill_first_iteration(
        self,
        input_ids,
        assistant_input_ids,
    ):
        # do prefix first
        target_logits = None
        if self.model.device != self.assistant_model.device:
            stream_0 = torch.cuda.Stream(device=self.model.device)
            stream_1 = torch.cuda.Stream(device=self.assistant_model.device)
            with torch.cuda.stream(stream_0):
                # candidate_future = executor.submit(candidate_generator.get_candidates, input_ids, assistant_input_ids)
                
                outputs = self.get_prefill_outputs(self.model, input_ids)
                past_key_values = outputs.past_key_values
                # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
                # model_kwargs = model._update_model_kwargs_for_generation(
                #     outputs,
                #     model_kwargs,
                #     is_encoder_decoder=model.config.is_encoder_decoder,
                #     num_new_tokens=0,
                # )
            with torch.cuda.stream(stream_1):
                assistant_outputs = self.get_prefill_outputs(self.assistant_model, assistant_input_ids)
                past_key_values_assistant = assistant_outputs.past_key_values
                # Candidate generation on GPU 1
                # candidate_input_ids, candidate_logits = candidate_generator.get_candidates(input_ids, assistant_input_ids.to(candidate_generator.assistant_model.device))
            stream_1.synchronize()
            stream_0.synchronize()
            # stream_1.synchronize()
            del outputs, assistant_outputs
        else:
            # concatenate the input_ids and assistant_input_ids
            # Create the position IDs

            # concatenated_input_ids = torch.cat((input_ids, assistant_input_ids), dim=1)
            # print(input_ids.shape)
            model_inputs = self.model.prepare_inputs_for_generation(input_ids, num_logit_to_keep=0)
            # if "num_logits_to_keep" in model_inputs:
            model_inputs["num_logits_to_keep"] = 1
            # position_ids_input = torch.arange(0, input_ids.size(1), dtype=torch.long, device=input_ids.device)
            # position_ids_assistant = torch.arange(0, assistant_input_ids.size(1), dtype=torch.long, device=input_ids.device)
            # concatenated_position_ids = torch.cat((position_ids_input, position_ids_assistant), dim=0).unsqueeze(0)
            # model_inputs.update({"position_ids": position_ids_input})
            model_inputs.update({"use_cache": True})
            outputs = self.model(**model_inputs)
            from transformers import DynamicCache

            overlap_length = 0
            for i in range(min(len(assistant_input_ids[0]), len(input_ids[0]))):
                if assistant_input_ids[0][i] == input_ids[0][i]:
                    overlap_length += 1
                else:
                    break

            # Extract past_key_values
            past_key_values = outputs.past_key_values
            print(f"Overlap length: {overlap_length}, Assistant input length: {len(assistant_input_ids[0])}, Input length: {len(input_ids[0])}")
            def split_past_key_values(all_past, split_position):
                all_keys, all_values = all_past.key_cache, all_past.value_cache
                assistant_keys_values = []
                for key, value in zip(all_keys, all_values):
                    assistant_keys_values.append((key[:, :, :split_position, :], value[:, :, :split_position, :]))
                # past_key_values_target = DynamicCache.from_legacy_cache(target_keys_values)
                past_key_values_assistant = DynamicCache.from_legacy_cache(assistant_keys_values)
                return past_key_values_assistant

            # Split the concatenated past_key_values
            past_key_values_assistant = split_past_key_values(DynamicCache.from_legacy_cache(past_key_values), overlap_length)
            return past_key_values, past_key_values_assistant
        return past_key_values, past_key_values_assistant




    def generate(
            self,
            long_input: str,
            rag_input: str,
            speculative_margin: float=0.1,
            do_speculative: bool=False,
            rag_only: bool=False,
            **generation_kwargs
        ):
        with torch.no_grad():
            input_ids = self.tokenizer([long_input], return_tensors="pt", add_special_tokens=False).input_ids.to(self.model.device)
            assistant_input_ids = self.assistant_tokenizer([rag_input], return_tensors="pt", add_special_tokens=False).input_ids.to(self.assistant_model.device)
            prefill_start_time = time.time()
            if do_speculative:
                # target_past_key_values, assistant_past_key_values = self.prefill_first_iteration(input_ids[:, :-1], assistant_input_ids)
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
                    **generation_kwargs
                    # max_new_tokens=100,
                    # do_sample=False, 
                    # top_p=1, 
                    # top_k=-1, 
                    # temperature=1,
                    # speculative_margin=speculative_margin
                )
                latency['prefill_time'] = prefill_time
                latency['accept_tokens'] = latency['accept_tokens'].item()
                latency['candidates_tokens'] = latency['candidates_tokens']
            else:
                if rag_only:
                    input_ids = assistant_input_ids
                prefill_outputs = self.get_prefill_outputs(self.model, input_ids[:, :-1])
                prefill_time = time.time() - prefill_start_time
                past_key_values = prefill_outputs.past_key_values
                decoding_start_time = time.time()
                outputs = self.model.generate(
                    input_ids.to(self.model.device), 
                    eos_token_id=self.tokenizer.eos_token_id,
                    past_key_values=past_key_values,
                    output_latency=False,
                    **generation_kwargs
                    # max_new_tokens=100,
                    # do_sample=False, 
                    # top_p=1, 
                    # top_k=-1, 
                    # temperature=1,
                    # speculative_margin=speculative_margin
                )
                latency = {
                    "prefill_time": prefill_time,
                    "decoding_time": time.time() - decoding_start_time
                }
                # print(outputs.shape)
                
                # accepted_tokens: number of tokens accepted by the model
                # judge if accepted tokens is a tensor
                # if isinstance(accepted_tokens, torch.Tensor):
                #     accepted_tokens = accepted_tokens.item()
            # print(outputs)
            num_tokens_generated = outputs.shape[1] - input_ids.shape[1]
            tokens_per_second = num_tokens_generated / latency['decoding_time']
            latency['num_tokens_generated'] = num_tokens_generated
            latency['tokens_per_second'] = tokens_per_second
            responses = self.tokenizer.batch_decode(outputs[:,input_ids.shape[1]:], skip_special_tokens=True)[0]
            return responses, latency
