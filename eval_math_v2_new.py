from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Optional
import json
import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import HfArgumentParser, AutoTokenizer
from typing import List
import numpy as np
import pandas as pd
# from math_utils import to_answer, prompt_styles
from grader import math_equal
import os

from math_verify import parse, verify


prompt_styles = {
    "llama3": '<|start_header_id|>user<|end_header_id|>\n\nSolve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{{answer}}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.\n\nProblem: {problem}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n',
    'qwen': '<|im_start|>system\nPlease reason step by step, and put your final answer within \boxed{{}}.<|im_end|>\n<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n',
    'llama3-vanilla':'<|start_header_id|>user<|end_header_id|>\n\nPlease reason step by step, and put your final answer within \boxed{{}}.\n\nProblem: {problem}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'}

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    model_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        metadata={"help": "the model name or path"},
    )
    eos_ids: List[int] = field(default_factory=lambda: [], metadata={"help": "the ids of the end of sentence tokens"})
    dataset_name_or_path: Optional[str] = field(
        default="hendrydong/hendrycks_math",
        metadata={"help": "the location of the dataset name or path"},
    )
    split: Optional[str] = field(
        default="math500",
        metadata={"help": "the split of the dataset"},
    )
    output_dir: Optional[str] = field(
        default="llama-8b-math.jsonl",
        metadata={"help": "the location of the output file"},
    )
    max_new_tokens: Optional[int] = field(
        default=8000,
        metadata={"help": "the maximum length of the new tokens"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "the random seed"},
    )
    temperature: Optional[float] = field(
        default=0.,
        metadata={"help": "the temperature"},
    )
    dataset_key: Optional[str] = field(
        default="problem",
        metadata={"help": "the key of the dataset"},
    )
    num_of_workers: Optional[int] = field(
        default=2048,
        metadata={"help": "the number of workers"},
    )
    prompt_styles: Optional[str] = field(
        default="qwen",
        metadata={"help": "the prompt style"},
    )
    ngpu: Optional[int] = field(
        default=8,
        metadata={"help": "the number of gpus"},
    )



# load hyperparameters
parser = HfArgumentParser((ScriptArguments,))
args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)


# load the dataset
dataset = load_dataset(args.dataset_name_or_path, split = args.split)

# print(dataset)
# print(dataset[0])

import subprocess

command = ["bash", "//export/home/projects/scripts/run_Ngpus.sh", args.model_name_or_path, str(args.ngpu)]
if args.max_new_tokens < 8000:
    command = ["bash", "//export/home/projects/scripts/run_Ngpus_4K.sh", args.model_name_or_path, str(args.ngpu)]


subprocess.run(command)
print("Servers are running")

# identify whether port 8000 is available
import time
while True:
    # lsof 
    time.sleep(5)
    print("Checking if port 8000 is available")
    command = '''lsof -i:8000'''
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    # if not available, break
    if result.stdout != "":
        break

# wait other ports
time.sleep(10)

import requests
def query_model(prompt, args, port):
    json = {
        **args,
        "prompt": prompt,
    }
    response = requests.post(url="http://localhost:" + str(port) + "/generate", json=json)
    response_json = response.json()
    return [response_json["text"][i][len(prompt) :] for i in range(len(response_json["text"]))]

default_args = {
    "n": 1,
    "temperature": args.temperature,
    "max_tokens": args.max_new_tokens,
    "top_p": 1.0,
    "top_k": -1,
    "stop_token_ids": [tokenizer.eos_token_id],
}


if __name__ == "__main__":
    num_of_workers = args.num_of_workers
    prompts = []
    for i in range(len(dataset)):
        t = prompt_styles[args.prompt_styles].format(problem = dataset[i][args.dataset_key])
        #t = tokenizer.apply_chat_template([{"role":"user", "content": dataset[i][args.dataset_key]}], tokenize = False, add_generation_prompt= True)
        prompts.append(t)
    ports = [8000+i for i in range(args.ngpu)]
    

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    with ThreadPoolExecutor(max_workers=args.num_of_workers) as executor:
        result = [
            executor.submit(query_model, prompts[i], default_args, ports[i % len(ports)]) for i in range(len(prompts))
        ]
        # use tqdm to show progress
        for _ in tqdm(as_completed(result), total=len(result)):
            pass

    generated_texts = [r.result()[0] for r in result]

    # Flatten the list of results (each worker returns a list of generated texts)

    
    c = 0
    for i in range(len(dataset)):
        ans = parse(generated_texts[i])
        #if ans:
        #    ans = parse(ans)
        sol = parse(dataset[i]["solution"])
        if verify(ans, sol):
            c+=1
    


    # kill the servers
    command = '''pkill -f "python -m vllm.entrypoints.api_server"'''
    subprocess.run(command, shell=True, text=True, capture_output=True)

    print(args.model_name_or_path, c/len(dataset))

    # save as jsonl
    with open(args.output_dir, "w") as f:
        for i in range(len(dataset)):
            f.write(json.dumps({"problem": dataset[i][args.dataset_key], "response": generated_texts[i]}) + "\n")
    print(args.model_name_or_path, c/len(dataset))


