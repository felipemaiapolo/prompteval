import json
import os
from tqdm.auto import tqdm

N_TEMPLATES = 100

evaluators_configs_path = "/home/skunk/miniconda3/envs/arena/lib/python3.12/site-packages/alpaca_eval/evaluators_configs"

generation_paths = ["../data/AlpacaEval_outputs/Mistral-7B-Instruct-v0.2/model_outputs.json",
                    "../data/AlpacaEval_outputs/llama-2-70b-chat-hf/model_outputs.json",
                    "../data/AlpacaEval_outputs/cohere/model_outputs.json",
                    "../data/AlpacaEval_outputs/Qwen1.5-7B-Chat/model_outputs.json"]

for generation_path in generation_paths:
    with open(generation_path) as f:
        d = json.load(f)
    for i in range(len(d)):
        d[i]['generator'] = d[i]['generator'].replace('-new','')+'-new' #if we do not do this, AE will use the cache evals
    with open(generation_path, 'w') as f:
        json.dump(d, f)

    for s in ['templates','configs']:
        command = f"cp -r '../data/templates/AlpacaEval/{s}' '{evaluators_configs_path}'"
        os.system(command)

    path = "/".join(generation_path.split('/')[:-1])
    
    for i in tqdm(range(N_TEMPLATES)):
        command = f"alpaca_eval --model_outputs '{generation_path}' --annotators_config 'configs/config_{i}.yaml'"
        os.system(command)
        
        command = f"rm '{evaluators_configs_path}/configs/annotations_seed0_config_{i}.json'"
        os.system(command)
        
        command = f"rm -rf '{path}/leaderboard.csv'"
        os.system(command)
        
        old_file_name = path+'/annotations.json'
        os.rename(old_file_name, old_file_name.replace('.json','')+f"_{i}.json")