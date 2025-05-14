from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
import argparse
import time
import json
import random
import warnings
import os
BASE_DIR = './data'
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-v0.1')
    parser.add_argument('--world_size', type=int, default=4)                                        # controls the number of gpus vLLM is allowed to use
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--input_dir', type=str, default='HuggingFaceH4/ultrafeedback_binarized')   # The SFT dataset
    parser.add_argument('--split', type=str, default='train_sft')
    parser.add_argument('--revision', type=str, default=None)                                       # model revision
    parser.add_argument('--chat', action='store_true')                                              # apply chat_template
    parser.add_argument('--system_message', type=str, default=None)                                   # user specific system message, only used when chat_template applied
    parser.add_argument('--temp', type=float, default=0.7)                      
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--output_prefix', type=str, default="output")
    parser.add_argument('--compute_len', action='store_true')
    return parser.parse_args()

def maybe_insert_system_message(messages, tokenizer, sysms):
    if messages[0]["role"] == "system":
        return
    if sysms:
        messages.insert(0, {"role": "system", "content": sysms})

def process_length(example):
    ids = tokenizer.encode(example["real"], add_special_tokens=False)
    example["len"] = len(ids)
    return example

def process_default(example):
    example['prompt'] = example['messages'][0]['content']
    example['real'] = example['messages'][1]['content']
    if args.chat:
        prompt_messages = example['messages'][:1]
        maybe_insert_system_message(prompt_messages, tokenizer, args.system_message)
        example['_prompt'] = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt = True)
    else:
        example['_prompt'] = example['prompt']
    return example

def process_metamathqa(example):
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )
    example['prompt'] = example['query']
    example['_prompt'] = problem_prompt.format(instruction = example['query'])
    example['real'] = example['response']
    return example

def process_ultrafeedback_binarized_cleaned(example):
    example['prompt'] = example['chosen'][0]['content']
    example['real'] = example['chosen'][1]['content']
    if args.chat:
        prompt_messages = example['chosen'][:1]
        maybe_insert_system_message(prompt_messages, tokenizer, args.system_message)
        example['_prompt'] = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt = True)
    else:
        example['_prompt'] = example['prompt']
    return example

def process_open_thoughts(example):
    example["prompt"] = example['conversations'][0]["value"]
    example["real"] = example['conversations'][1]["value"]
    system = example['system']
    if args.chat:
        prompt_messages = [{"role" : "user", "content" : example["prompt"]}]
        maybe_insert_system_message(prompt_messages, tokenizer, system)
        example["_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt = True)
    else:
        example['_prompt'] = example['prompt']
    return example

if __name__ == "__main__":
    args = parse_arguments()
    model_path = args.model
    world_size = args.world_size
    NAME = args.output_prefix
    TEMP = args.temp
    TOP_P = args.top_p
    TOP_K = args.top_k
    MAX_TOKENS = args.max_new_tokens
    OUT_NAME = f'{NAME}_{TEMP}_{TOP_P}_{TOP_K}_{MAX_TOKENS}_{args.seed}.json'

    # load a base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.chat_template is None:                         # add default chat template
        warnings.warn("No chat_template found, using default chat_template.")
        tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

    llm = LLM(
        model=model_path,
        tensor_parallel_size=world_size,
        seed = args.seed,
        revision = args.revision
    )

    sampling_params = SamplingParams(temperature=TEMP, top_p=TOP_P, top_k=TOP_K, max_tokens=MAX_TOKENS)

    # load data
    data = load_dataset(args.input_dir, split=args.split)
    if args.input_dir == 'meta-math/MetaMathQA':
        data = data.map(process_metamathqa)
    elif args.input_dir == 'argilla/ultrafeedback-binarized-preferences-cleaned':
        data = data.map(process_ultrafeedback_binarized_cleaned)
    elif args.input_dir == 'open-thoughts/OpenThoughts-114k':
        data = data.map(process_open_thoughts)
    else:
        data = data.map(process_default)
    
    if args.compute_len:
        data = data.map(process_length)
        print("Avg Length: ", sum(data["len"]) / len(data["len"]))

    prompts_old = data['prompt']
    prompts_all = data['_prompt']
    corrects_all = data['real']
    for index in random.sample(range(len(prompts_all)), 3):
        print(prompts_all[index])

    start=time.time()

    #run vllm
    results_gathered = list(map(lambda x: x.outputs[0].text, 
                                llm.generate(prompts_all, sampling_params)))

    results = [r.replace(tokenizer.eos_token, "").lstrip() for r in results_gathered]

    timediff=time.time()-start
    print(f"time elapsed: {timediff}")
    data = []
    for idx in range(len(corrects_all)):
        d = {
            "prompt": prompts_old[idx],
            "real": corrects_all[idx],
            "generated": results[idx],
            "id": idx
        }
        data.append(d)

    os.makedirs(BASE_DIR, exist_ok=True)
    with open(os.path.join(BASE_DIR, OUT_NAME), 'w') as f:
        json.dump(data, f, indent=4)