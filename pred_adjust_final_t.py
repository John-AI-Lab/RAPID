import os, csv, json
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset
import re
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
import tiktoken
import torch.multiprocessing as mp

import sys
sys.path.append("/mnt/workspace/Projects/LongSpec/")
sys.path.append("/mnt/workspace/Projects/LongSpec/LongBench")
from src_adjust_final_t.rag import get_input_prompts, get_rag_context
from src_adjust_final_t.model_wrap import LongSpecEngine
import torch


import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Set the seed value all over the place to make this reproducible.
seed_val = 42
set_seed(seed_val)




model_map = json.loads(open('LongBench/config/model2path.json', encoding='utf-8').read())
maxlen_map = json.loads(open('LongBench/config/model2maxlen.json', encoding='utf-8').read())

URL = "http://127.0.0.1:8000/v1"
API_KEY = "token-abc123"
template_rag = open('LongBench/prompts/0shot_rag.txt', encoding='utf-8').read()
template_no_context = open('LongBench/prompts/0shot_no_context.txt', encoding='utf-8').read()
template_0shot = open('LongBench/prompts/0shot.txt', encoding='utf-8').read()
template_0shot_cot = open('LongBench/prompts/0shot_cot.txt', encoding='utf-8').read()
template_0shot_cot_ans = open('LongBench/prompts/0shot_cot_ans.txt', encoding='utf-8').read()



def apply_chat_template(tok, input_text):
    messages = [
            # {
            #     "role": "system",
            #     "content": "You are a friendly chatbot who always remembers all details of chat history which would be precisely used later and responds in the style of a pirate.",
            # },
            {
                "role": "user", "content": input_text
            }
        ]
    input_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return input_text

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
    # if len(device_list) == 4:
    #     max_memory[3] = "76GiB"
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

def query_llm(
        prompt, 
        model, 
        tokenizer,
        assistant_model,
        assistant_tokenizer, 
        max_len, 
        client=None, 
        rag_prompt="",  
        temperature=0.5, 
        max_new_tokens=128, 
        stop=None, 
        do_speculative=False,
        rag_only=False,
        speculative_margin=0.1,
        assistant_confidence_threshold=0.1,
        num_assistant_tokens=10,
        alpha=0.1
    ):


    # truncate
    # max_len = maxlen_map[model]
    # max_len
    # if model in model_map:
    input_ids = tokenizer.encode(prompt)
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
        prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    # else:
    #     input_ids = tokenizer.encode(prompt, disallowed_special=())
    #     if len(input_ids) > max_len:
    #         input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
    #         prompt = tokenizer.decode(input_ids)
    # if model in model_map:
    #     model = model_map[model] 
    
    full_prompt = apply_chat_template(tokenizer, prompt)
    if rag_prompt:
        rag_prompt = apply_chat_template(tokenizer, rag_prompt)

    with torch.no_grad():
        # assistant_model, assistant_tokenizer = model, tokenizer
        longspec_engine = LongSpecEngine(model, tokenizer, assistant_model, assistant_tokenizer)
        input_ids = tokenizer([full_prompt], return_tensors="pt", add_special_tokens=False).input_ids
        print(input_ids.shape)
        if do_speculative:
            assistant_input_ids = assistant_tokenizer([rag_prompt], return_tensors="pt", add_special_tokens=False).input_ids
            # pritn(retrieved_context[-1000])
            print(assistant_input_ids.shape)
            outputs, latency = longspec_engine.generate(
                full_prompt,
                rag_prompt,
                # inpud_ids.to(model.device),
                # assistant_input_ids.to(assistant_model.device),
                speculative_margin=speculative_margin,
                do_speculative=True,
                max_new_tokens=max_new_tokens,
                top_p=1, 
                # top_k=-1, 
                temperature=temperature,
                assistant_confidence_threshold=assistant_confidence_threshold,
                num_assistant_tokens=num_assistant_tokens,
                spec_alpha=alpha
            )
            # accepted_tokens = latency['accept_tokens'].item()
            # accepted_tokens = accepted_tokens.item()
        else:
            # inpud_ids = torch.cat([assistant_input_ids, inpud_ids], dim=1)
            outputs, latency = longspec_engine.generate(
                full_prompt,
                rag_prompt,
                do_speculative=False,
                # inpud_ids.to(model.device), 
                # eos_token_id=tok.eos_token_id, 
                max_new_tokens=max_new_tokens, 
                # do_sample=False, 
                top_p=1, 
                # top_k=-1, 
                temperature=temperature,
                rag_only=rag_only
            )
    return outputs, latency
    



def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None

def get_pred(
        data,
        args, 
        fout,
        # max_chunk_len=128,
        # num_to_recall=32,
        # rag_threshold=0.1,
    ):
    model = args.model
    max_len = maxlen_map[model]
    model = model_map[model] 
    model, tokenizer = load_model(model, args.device_list)
    if args.assistant_model == args.model and args.assistant_device_list == args.device_list:
        assistant_model = model
        assistant_tokenizer = tokenizer
    else:
        assistant_model = model_map[args.assistant_model] 
        assistant_model, assistant_tokenizer = load_model(assistant_model, args.assistant_device_list)
    # if "gpt" in model or "o1" in model:
    #     tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained(model_map[model], trust_remote_code=True)
    client = OpenAI(
        base_url=URL,
        api_key=API_KEY
    )
    for item in tqdm(data):
        context = item['context']
        if args.rag > 0:
            template = template_rag
            retrieved = item["retrieved_context"][:args.rag]
            retrieved = sorted(retrieved, key=lambda x: x['c_idx'])
            context = '\n\n'.join([f"Retrieved chunk {idx+1}: {x['content']}" for idx, x in enumerate(retrieved)])
        elif args.no_context:
            template = template_no_context
        elif args.cot:
            template = template_0shot_cot
        else:
            template = template_0shot
        prompt = template.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip())
        
        len_prompt = len(tokenizer.encode(prompt))
        len_prompt = len_prompt if len_prompt < max_len else max_len
        num_to_retrieve = len_prompt / (args.max_chunk_len * args.num_to_shorten)
        minimum_chunks = min(args.minimum_rag_len / args.max_chunk_len, len_prompt / args.max_chunk_len)
        import math
        num_to_retrieve = math.ceil(max(num_to_retrieve, minimum_chunks))
        print(f"Number of Chunks to Retrieve: {num_to_retrieve}")
        rag_prompt = ""
        # if args.do_speculative:
        query = prompt.split("</text>")[-1].strip().split("\n\n")[0]
        rag_context, num_rag_chunks = get_rag_context(
            context.strip(),
            query.replace("What is the correct answer to this question:", "").strip(),
            "/mnt/workspace/ckpts/RAG/bge-m3",
            # "/mnt/workspace/ckpts/RAG/bge-large-en-v1.5",
            num_to_retrieve=num_to_retrieve,
            max_chunk_token=args.max_chunk_len,
            tokenizer=tokenizer,
            rag_threshold=args.rag_threshold,
        )
        rag_template = template.replace("Please read the following text", "Please read the following retrieved text chunks")
        rag_prompt = rag_template.replace("$DOC$", rag_context).replace("$Q$", item['question'].strip()).replace("$C_A$", item['choice_A'].strip()).replace("$C_B$", item['choice_B'].strip()).replace("$C_C$", item['choice_C'].strip()).replace("$C_D$", item['choice_D'].strip())

        if args.cot:
            max_new_tokens = 1024
        else:
            max_new_tokens = 128
        output, latency = query_llm(
            prompt, 
            model, 
            tokenizer,
            assistant_model,
            assistant_tokenizer,
            max_len, 
            client, 
            rag_prompt=rag_prompt, 
            temperature=0.1, 
            max_new_tokens=max_new_tokens, 
            do_speculative=args.do_speculative, 
            speculative_margin=args.speculative_margin,
            rag_only=args.rag_only,
            assistant_confidence_threshold=args.assistant_confidence_threshold,
            num_assistant_tokens=args.num_assistant_tokens,
            alpha=args.alpha
            
        )
        if output == '':
            continue
        if args.cot: # extract answer
            response = output.strip()
            item['response_cot'] = response
            prompt = template_0shot_cot_ans.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip()).replace('$COT$', response)
            # output = query_llm(prompt, model, tokenizer, client, temperature=0.1, max_new_tokens=128)
            output, _ = query_llm(
                prompt, 
                model, 
                tokenizer,
                assistant_model,
                assistant_tokenizer,
                max_len, 
                client, 
                rag_prompt=rag_prompt, 
                temperature=0.1, 
                max_new_tokens=128, 
                do_speculative=False, 
                speculative_margin=args.speculative_margin
            )
            if output == '':
                continue
        response = output.strip()
        item['response'] = response
        item['pred'] = extract_answer(response)
        item['judge'] = item['pred'] == item['answer']
        item['context'] = context[:1000]
        item['rag_config'] = {
            'max_chunk_len': args.max_chunk_len,
            'num_rag_chunks': num_rag_chunks,
            'rag_threshold': args.rag_threshold
        }
        item['latency'] = latency
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        fout.flush()

def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    if args.rag > 0:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + f"_rag_{str(args.rag)}.jsonl")
    elif args.no_context:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + "_no_context.jsonl")
    elif args.cot:
        # out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + "_cot.jsonl")
        out_file = os.path.join(args.save_dir, args.save_name + "_cot.jsonl")
    else:
        # out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + ".jsonl")
        out_file = os.path.join(args.save_dir, args.save_name + ".jsonl")

    dataset = load_dataset('/mnt/workspace/data/THUDM/LongBench-v2', split='train') # dataset = json.load(open('data.json', 'r', encoding='utf-8'))
    data_all = [{"_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"], "difficulty": item["difficulty"], "length": item["length"], "question": item["question"], "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]} for item in dataset]

    # cache
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}
    fout = open(out_file, 'a', encoding='utf-8')
    data = []
    for item in data_all:
        if item["_id"] not in has_data:
            data.append(item)

    # data_subsets = [data[i::args.n_proc] for i in range(args.n_proc)]
    data_subsets = data[args.rank::args.n_proc]
    get_pred(data_subsets, args, fout)
    # processes = []
    # for rank in range(args.n_proc):
    #     p = mp.Process(target=get_pred, args=(data_subsets[rank], args, fout))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--save_name", type=str, default="llama31")
    parser.add_argument("--model", "-m", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--assistant_model", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--cot", "-cot", action='store_true') # set to True if using COT
    parser.add_argument("--no_context", "-nc", action='store_true') # set to True if using no context (directly measuring memorization)
    parser.add_argument("--rag", "-rag", type=int, default=0) # set to 0 if RAG is not used, otherwise set to N when using top-N retrieved context
    parser.add_argument("--n_proc", "-n", type=int, default=4)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--do_speculative", action='store_true')
    parser.add_argument("--rag_only", action='store_true')
    parser.add_argument("--speculative_margin", type=float, default=0.1)
    parser.add_argument("--max_chunk_len", type=int, default=128)
    parser.add_argument("--minimum_rag_len", type=int, default=4096)
    parser.add_argument("--num_to_shorten", type=int, default=32)
    parser.add_argument("--rag_threshold", type=float, default=0.1)
    parser.add_argument("--assistant_confidence_threshold", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--num_assistant_tokens", type=int, default=10)
    parser.add_argument("--device_list", type=str, default="0")
    parser.add_argument("--assistant_device_list", type=str, default="0")
    args = parser.parse_args()
    main()