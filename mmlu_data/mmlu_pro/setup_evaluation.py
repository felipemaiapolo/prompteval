import json
import os
from tqdm import tqdm

from dotenv import load_dotenv

import constants

if __name__ == '__main__':
    
    with open('template_to_index.json', 'r') as file:
        indices = [v for _, v in json.load(file).items()]

    commands = []

    for i in tqdm(indices, desc='building'):
        
        output_path = os.path.join('output', f't_{i}')
        if os.path.exists(output_path) and len(os.listdir(output_path)) > 0:
            continue
        
        command = f"lm_eval --model ibm_genai --model_args model_id={constants.model_path} --tasks mmlu_pro.{i} --batch_size auto --output_path {output_path} --log_samples"

        commands.append(command)

    load_dotenv()

    print(len(commands))

    api_keys = []
    for i in [0]:
        api_keys.append(os.getenv(f"KEY_{i}"))
    
    key_dict = {api_key: [] for api_key in api_keys}
    for i, command in tqdm(enumerate(commands), desc='matching'):
        key_dict[api_keys[i%len(api_keys)]].append(command)

    with open(f'{constants.prefix}_assignment.json', 'w') as file:
        json.dump(key_dict, file, indent=2)