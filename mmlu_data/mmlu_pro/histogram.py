
import constants
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

if __name__ == "__main__":
    folder_dir = f'./{constants.prefix}_output'

    with open('template_to_index.json', 'r') as file:
        indices = [v for _, v in json.load(file).items()]

    scores_list = []
    for i in tqdm(indices, desc='scoring'):
        template_dir = os.path.join(folder_dir, f't_{i}')
        
        for j_file in os.listdir(template_dir):

            if os.path.isfile(os.path.join(template_dir, j_file)) and j_file.startswith("result") and j_file.endswith(".json"):

                with open(os.path.join(template_dir, j_file), 'r') as file:
                    scores_list.append(json.load(file)["results"][f"mmlu_pro.{i}"]["unitxt_accuracy,none"])

    
    plt.hist(scores_list, bins=30, edgecolor='black')  # Adjust the number of bins as needed
    plt.xlabel('Accuracies')
    plt.ylabel('Frequency')
    # plt.title(f'{constants.prefix} Accuracy Spread')
    plt.savefig(f'{constants.prefix}_histogram.pdf')


