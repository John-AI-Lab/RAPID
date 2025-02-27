import torch
from torch.nn import functional as F
from typing import Optional
from transformers import SinkCache, DynamicCache
from tqdm import tqdm
import time

# copy from https://github.com/LeeSinLiang/microGPT/blob/ed40cf9780dbeb180adfe94c227d4aa97e69250e/gpt.py
def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """

    Args:
        logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
        top_k (int, optional): top_k. Defaults to 0.
        top_p (float, optional): top_p. Defaults to 0.0.

    Returns:
        torch.Tensor: a renormalized logits
    """
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[:, [-1]]] = float('-inf')
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float('-inf')
    return logits


def norm_logits(logits : torch.Tensor, temperature : float, top_k : float, top_p : float) -> torch.Tensor:
    """

    Args:
        logits (torch.Tensor): shape (1, vocab)
        temperature (float): temperature
        top_k (float): top_k
        top_p (float): top_p

    Returns:
        torch.Tensor: next token with shape as (batch,  1)
    """
    assert logits.dim() == 2
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=1)
    return probs


def sample(probs : torch.Tensor, num_samples: int = 1):
    idx_next = torch.multinomial(probs, num_samples=num_samples)
    # if (idx_next.item() == 0):
    #     raise RuntimeError
    
    # idx_next = torch.argmax(probs, dim=1).unsqueeze(0)
    return idx_next


def max_fn(x):
    """
        norm(max (x, 0))
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=1, keepdim=True) 
    return x_max / x_max_sum


class SelfSpecModel():
    def __init__(self, model : torch.nn.Module, temperature : float = 1, top_k : int = 0, top_p : float = 0) -> None:
        self._model = model
        self._past_key_values = None
        self._draft_past_key_values = None

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p

    def _forward_with_kvcache(self, input_ids : torch.Tensor) -> torch.Tensor:
        if self._past_key_values is None:
            # prefill stage
            cache = DynamicCache()
            outputs = self._model(input_ids, past_key_values=cache, use_cache=True, num_logits_to_keep=1)
            cache = outputs.past_key_values
                
            self._past_key_values = cache
            last_q = norm_logits(outputs.logits[:, -1, :], self._temperature, self._top_k, self._top_p)
            not_cached_q = None
        else:
            outputs = self._model(input_ids, past_key_values=self._past_key_values, use_cache=True, num_logits_to_keep=input_ids.shape[1])
            
            not_cached_q = outputs.logits
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)
                
            for i in range(not_cached_q.shape[-2]):
                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)
            
            last_q = not_cached_q[:, -1, :]
            self._past_key_values = outputs.past_key_values
        
        return last_q, not_cached_q
    
    def _draft_forward_with_kvcache(self, input_ids : torch.Tensor) -> torch.Tensor:
        if self._draft_past_key_values is None:
            cache = SinkCache(window_length=4096, num_sink_tokens=4)
            
            outputs = self._model(input_ids, past_key_values=cache, use_cache=True, num_logits_to_keep=1)
            cache = outputs.past_key_values
            
            self._draft_past_key_values = cache
            last_q = norm_logits(outputs.logits[:, -1, :], self._temperature, self._top_k, self._top_p)
            not_cached_q = None
            cache_prev = None
        else:
            cache_prev = self._clone_sinkcache(self._draft_past_key_values)
            # cache_prev = None
            outputs = self._model(input_ids, past_key_values=self._draft_past_key_values, use_cache=True, num_logits_to_keep=input_ids.shape[1])
            
            not_cached_q = outputs.logits
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)
                
            for i in range(not_cached_q.shape[-2]):
                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)
            
            last_q = not_cached_q[:, -1, :]
            self._draft_past_key_values = outputs.past_key_values
            # cache_curr = self._clone_sinkcache(self._draft_past_key_values)
        
        return last_q, not_cached_q, cache_prev
    
    def _clone_sinkcache(self, cache):
        new_cache = SinkCache(window_length=4096, num_sink_tokens=4)
        new_cache.key_cache = [k.clone() for k in cache.key_cache]
        new_cache.value_cache = [v.clone() for v in cache.value_cache]
        new_cache.cos_sin_rerotation_cache = {k: (v[0].clone(), v[1].clone()) for k, v in cache.cos_sin_rerotation_cache.items()}
        new_cache._cos_cache = cache._cos_cache.clone()
        new_cache._sin_cache = cache._sin_cache.clone()
        new_cache._seen_tokens = cache._seen_tokens
        return new_cache
    
    @torch.no_grad()  
    def prefill(self, input):
        q, _ = self._forward_with_kvcache(input)
        return sample(q)
    
    @torch.no_grad()
    def draft_prefill(self, input):
        q, _, _ = self._draft_forward_with_kvcache(input)
        return sample(q)

    @torch.no_grad()
    def _generate_with_kvcache(self, input_ids : torch.Tensor, 
                                    gamma : int, 
                                    mode : str) -> torch.Tensor:
        """ forward the model gamma times

        Args:
            prefix (torch.Tensor): the prefix
            gamma (int): how many times approx guesses

        Returns:
            Torch.Tensor: prefix+generated tokens
        """
        input = input_ids.clone()
        x = input_ids.clone()
        logits_out = None
        caches = []

        for _ in range(gamma):
            if mode == "target":
                q, logits = self._forward_with_kvcache(input)
            else:
                q, logits, cache = self._draft_forward_with_kvcache(input)
                caches.append(cache)
                
            if logits_out is None:
                logits_out = logits
            else:
                logits_out = torch.cat((logits_out, logits), dim=1)   
            
            next_tok = sample(q)
            x = torch.cat((x, next_tok), dim=1)
            input = next_tok
        return x, logits_out, caches

    @torch.no_grad()
    def generate(self, input : torch.Tensor, gamma : int, mode = "target") -> torch.Tensor:
        output, logits, caches = self._generate_with_kvcache(input, gamma, mode)
        return output, logits, caches
    
    @torch.no_grad()
    def rollback(self, end_pos : int):
        assert self._past_key_values
        
        self._past_key_values._seen_tokens = end_pos
        for i in range(len(self._past_key_values.key_cache)):
            self._past_key_values.key_cache[i] = self._past_key_values.key_cache[i][:, :, :end_pos, :]
            self._past_key_values.value_cache[i] = self._past_key_values.value_cache[i][:, :, :end_pos, :]
        
    @torch.no_grad()
    def draft_rollback(self, cache):
        # self._draft_past_key_values._seen_tokens += end_pos
        # for i in range(len(self._past_key_values.key_cache)):
        #     self._draft_past_key_values.key_cache[i] = self._draft_past_key_values.key_cache[i][:, :, :end_pos, :]
        #     self._draft_past_key_values.value_cache[i] = self._draft_past_key_values.value_cache[i][:, :, :end_pos, :]
        self._draft_past_key_values = cache
        
    def clear(self):
        self._past_key_values = None
        self._draft_past_key_values = None
        
@torch.no_grad()
def autoregressive(prefix: torch.Tensor, model: torch.nn.Module, tokenizer, max_len: int, temperature: float = 1.0, top_k: int = 0, top_p: float = 0, verbose: bool = False, random_seed: int = None) -> torch.Tensor:
    """
    Autoregressive sampling.
    
    Args:
        prefix (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        model (torch.nn.Module): model
        max_len (int): the max overall generated tokens number.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    if random_seed:
        torch.manual_seed(random_seed)
    
    seq_len = prefix.shape[1]
    T = max_len
    
    eot_1 = tokenizer.eos_token_id
    if tokenizer.unk_token_id is not None:
        eot_2 = tokenizer.unk_token_id
    else:
        eot_2 = tokenizer.encode("<|eot_id|>")[-1]
    
    device = model.device
    
    engine = SelfSpecModel(model, temperature, top_k, top_p)
    
    # prefill
    prefill_start_time = time.time()
    nxt = engine.prefill(prefix)
    prefill_time = time.time() - prefill_start_time
    
    out = nxt.clone()
    tok = nxt.clone()
    decoding_start_time = time.time()
    
    while out.shape[1] < T:
        x, _, _ = engine.generate(tok, 1)        
        out = torch.cat((out, x[:, -1].unsqueeze(-1)), dim=1)
        tok = out[:, -1].unsqueeze(0)
        
        if out[0, -1] == eot_1 or out[0, -1] == eot_2:
            break
    
    latency = {
        "prefill_time": prefill_time,
        "decoding_time": time.time() - decoding_start_time
    }
    
    tokens_per_second = out.shape[1] / latency['decoding_time']
    latency['num_tokens_generated'] = out.shape[1]
    latency['tokens_per_second'] = tokens_per_second
    print(f"generated tokens numbers {out.shape[1]}, tokens_per_second {tokens_per_second}")
    
    engine.clear()
    torch.cuda.empty_cache()
    
    return out, latency
        

@torch.no_grad()
def self_spec(prefix : torch.Tensor, model : torch.nn.Module, tokenizer,
                         max_len : int , gamma : int = 4,
                         temperature : float = 1.0, top_k : int = 0, top_p : float = 0, verbose : bool = False, random_seed : int = None) -> torch.Tensor:
    """
    Google version Speculative Sampling.
    https://arxiv.org/pdf/2211.17192.pdf
        
    Adapted with KV Cache Optimization.
        
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    if random_seed:
        torch.manual_seed(random_seed)
    
    seq_len = prefix.shape[1]
    T = max_len
    
    eot_1 = tokenizer.eos_token_id
    if tokenizer.unk_token_id is not None:
        eot_2 = tokenizer.unk_token_id
    else:
        eot_2 = tokenizer.encode("<|eot_id|>")[-1]
    
    assert prefix.shape[0] == 1, "input batch size must be 1"
    
    device = model.device
    
    engine = SelfSpecModel(model, temperature, top_k, top_p)
    
    # prefill
    prefill_start_time = time.time()
    engine.draft_prefill(prefix)
    nxt = engine.prefill(prefix)
    prefill_time = time.time() - prefill_start_time
    
    resample_count = 0
    target_sample_count = 0
    accepted_count = 0
    candidates_count = 0
    out = nxt.clone()
    tok = nxt.clone()
    decoding_start_time = time.time()
    
    while out.shape[1] < T:
        x, draft_probs, caches = engine.generate(tok, gamma, mode="approx")        
        if_all, target_probs, _ = engine.generate(x, 1)
        
        draft_tokens = x[:, -gamma:]
        q = draft_probs[:, -gamma:, :]
        q_i = q[:, torch.arange(gamma), draft_tokens].squeeze(0, 1)
        
        p = target_probs
        p_i = p[:, torch.arange(gamma), draft_tokens].squeeze(0, 1)
        probability_ratio = p_i / q_i
        r_i = torch.rand_like(probability_ratio)
        flag_accept_matrix = (r_i <= probability_ratio)
        
        # print("out: ", tokenizer.batch_decode(out)[0])
        # print("x: ", tokenizer.batch_decode(x)[0])
        # print("accept_nums: ", accept_nums)
        
        # # flag_accept_matrix = (target_tokens == draft_tokens)
        eot_condition = ((draft_tokens == eot_1) | (draft_tokens == eot_2))
        
        accept_flags_int = (flag_accept_matrix & (~eot_condition)).int()
        accept_flags_cumprod = torch.cumprod(accept_flags_int, dim=1)
        accept_flags_matrix = accept_flags_cumprod.bool()
        accept_nums = accept_flags_matrix.sum(dim=1, keepdim=True)[0][0].item()
        
        out = torch.cat((out, x[:, 1:accept_nums+1]), dim=1)
        accepted_count += accept_nums
        candidates_count += gamma
        condition = (eot_condition & flag_accept_matrix).any(dim=1, keepdim=True)
        # if condition.any():
        #     break
        # if eot_condition

        # if accept_nums < gamma:
            # engine.rollback(seq_len + out.shape[1])
            # out = torch.cat((out, draft_tokens[:, :accept_nums]), dim=1)
            # # resample_count += 1
            # engine.draft_rollback(caches[accept_nums])
        p_n_plus_1 = p[:, accept_nums, :]
        if accept_nums < gamma:
            engine.rollback(seq_len + out.shape[1])
            engine.draft_rollback(caches[accept_nums])
            q_n_plus_1 = q[:, accept_nums, :]
            p_prime = torch.clamp((p_n_plus_1 - q_n_plus_1), min=0)
            p_prime.div_(p_prime.sum())
            resample_count += 1
            # print("from here 1")
        else:
            # engine.rollback(seq_len + out.shape[1])
            # engine.draft_rollback(caches[accept_nums])
            p_prime = p_n_plus_1
            target_sample_count += 1
        # if condition.any():
        #     t = torch.tensor([eot_1]).expand(out.size(0), 1).to(out.device)
        # else:
        # print(accept_nums < gamma)
        # print(p_prime)
        if torch.isnan(p_prime).any():
            t = torch.tensor([eot_1]).expand(out.size(0), 1).to(out.device)
        else:
            t = torch.multinomial(p_prime, num_samples=1).squeeze(1)[None, :]
        out = torch.cat((out, t), dim=1)
            # if (t == eot_1).any() or (t == eot_2).any():
            #     break
        # else:
        #     # out = torch.cat((out, draft_tokens[:, :accept_nums]), dim=1)
        #     # target_sample_count += 1
        #     out = torch.cat((out, if_all[:,-1].unsqueeze(-1)), dim=1)
        #     target_sample_count += 1
            
        tok = t
        # print("=" * 100)
        
        if t[0, 0] == eot_1 or t[0, 0] == eot_2:
            break
    
    latency = {
        "prefill_time": prefill_time,
        "decoding_time": time.time() - decoding_start_time,
        "accept_tokens": accepted_count,
        "candidates_tokens": candidates_count
    }
    
    tokens_per_second = out.shape[1] / latency['decoding_time']
    latency['num_tokens_generated'] = out.shape[1]
    latency['tokens_per_second'] = tokens_per_second
    print(f"generated tokens numbers {out.shape[1]}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}, tokens_per_second {tokens_per_second}")
    
    engine.clear()
    torch.cuda.empty_cache()
    
    return out, latency