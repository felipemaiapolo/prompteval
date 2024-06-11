import json
import os
import pickle

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from prompteval.get_features import create_discrete_features, create_sentence_transformers, create_finetuned_features
from prompteval.utils.utils import check_multicolinearity, pca_filter


def extract_mmlu():
    # fmt: off
    mmlu_tasks = [
        'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
        'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
        'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics',
        'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology',
        'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography',
        'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics'
        'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics',
        'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law',
        'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics',
        'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting',
        'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies',
        'sociology', 'us_foreign_policy', 'virology', 'world_religions'
    ]
    # fmt: on
    data = {}
    for task in tqdm(mmlu_tasks):
        mmlu_subset = load_dataset("felipemaiapolo/PromptEval_MMLU_correctness", task)
        data[task] = []

        for model in mmlu_subset.keys():
            Y = []
            J = mmlu_subset[model].shape[1]
            for j in range(J):
                Y.append(mmlu_subset[model][f"example_{j}"])
            Y = np.array(Y).T
            data[task].append(Y)

    return data


def get_directories_in_folder(folder_path):
    """
    Lists all directories in the given folder path.

    Parameters:
    folder_path (str): The path to the folder.

    Returns:
    list: A list of directory names in the folder.
    """
    all_entries = os.listdir(folder_path)
    directories = [entry for entry in all_entries if os.path.isdir(os.path.join(folder_path, entry))]
    return directories


def create_Ys(data_path="data/"):

    ### Creating Ys

    Ys = {}

    ## BBH, LMentry
    data_path_y = data_path + "raw open-source model responses with gold and auto validation values/"
    for bench in ["BBH", "LMentry"]:
        print(f"Creating Y for {bench}")
        Ys[bench] = {}

        for task in tqdm(get_directories_in_folder(data_path_y + bench)):

            Ys[bench][task] = []

            path = data_path_y + bench + "/" + task

            models = [x[2] for x in os.walk(path)][0]

            for model in models:

                with open(path + "/" + model) as json_data:
                    data = json.load(json_data)
                    json_data.close()

                ####
                examples = [str(j) for j in np.sort([int(i) for i in list(data["samples"].keys())])]
                format_types = ["default", "rephrase", "cot", "gradual"]
                n_examples = len(examples)
                n_formats = np.sum([len(data["samples"][examples[0]][template]) for template in format_types])

                ####
                Y = np.zeros((n_formats, n_examples))
                count_formats = 0
                for format_type in format_types:
                    formats = [
                        str(j)
                        for j in np.sort([int(i) for i in list(data["samples"][examples[0]][format_type].keys())])
                    ]
                    for format in formats:
                        for i in range(n_examples):
                            Y[count_formats, i] = int(
                                data["samples"][examples[i]][format_type][format]["auto_validation"]
                            )

                        count_formats += 1
                Ys[bench][task].append(Y)
                # print(f"- Bench: {bench}, Task: {task}, Model: {model}, Templates/Examples: {Y.shape}")

    ## MMLU
    print("Creating Y for MMLU")
    Ys["MMLU"] = extract_mmlu()

    ## Saving
    for bench in Ys.keys():
        for task in Ys[bench].keys():
            for model in range(len(Ys[bench][task])):
                if np.mean(np.unique(Ys[bench][task][model]) == np.array([0, 1])) < 1:
                    print("a")
                Ys[bench][task][model] = np.int8(Ys[bench][task][model])

    with open(data_path + "Ys.pickle", "wb") as handle:
        pickle.dump(Ys, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_Xs(data_path="data/", emb_dim=25):

    ## Creating "mmlu_values" which will be used to generate discrete features for mmlu
    # fmt: off
    formats = [
        0, 7, 8, 16, 19, 20, 31, 32, 35, 37, 41, 42, 45, 46, 47, 48, 50, 51, 55, 59, 63, 66, 71,
        72, 75, 76, 87, 94, 95, 96, 97, 104, 110, 111, 112, 113, 120, 122, 123, 124, 128, 132, 133,
        138, 140, 141, 144, 147, 148, 149, 154, 155, 158, 161, 162, 163, 166, 169, 170, 171, 181,
        182, 183, 190, 197, 200, 204, 207, 214, 215, 222, 226, 227, 229, 230, 241, 243, 244, 248,
        249, 250, 252, 258, 260, 261, 266, 267, 268, 272, 276, 278, 280, 282, 286, 290, 294, 296,
        298, 300, 301
    ]
    # fmt: on

    mmlu_formats_path = "data/templates/mmlu_templates.json"
    with open(mmlu_formats_path) as json_data:
        mmlu_templates = json.load(json_data)
        json_data.close()
    mmlu_templates = [mmlu_templates[i] for i in formats]
    mmlu_formats_path = "data/templates/mmlu_templates_metadata.json"
    with open(mmlu_formats_path) as json_data:
        mmlu_metadata = json.load(json_data)
        json_data.close()
    df = pd.DataFrame([mmlu_metadata[template] for template in mmlu_templates])
    mmlu_values = {col: df[col].unique() for col in df.columns}

    ## Generating X
    Xs = {}

    with open("data/Ys.pickle", "rb") as handle:
        Ys = pickle.load(handle)

    for bench in ["BBH", "LMentry", "MMLU"]:

        print(f"Creating X for {bench}")

        Xs[bench] = {}
        tasks = [s.replace(".csv", "") for s in [x[2] for x in os.walk(f"data/templates/{bench}")][0]]

        for task in tqdm(tasks):
            n_llms = len(Ys[bench][task])
            formats = list(pd.read_csv(f"data/templates/{bench}/{task}.csv")["prompt template"])

            # sentence transformers embeddings
            emb = create_sentence_transformers(formats, emb_dim)
            emb = pca_filter(emb)
            check_multicolinearity(emb)

            # manual/discrete features
            if bench != "MMLU":
                disc = create_discrete_features(formats, benchmark=bench, dataset=task)
            else:
                disc = []
                for feat in mmlu_values.keys():
                    x = np.zeros((len(mmlu_templates), len(mmlu_values[feat])))
                    for i, template in enumerate(mmlu_templates):
                        x[i, np.argmax(mmlu_values[feat] == mmlu_metadata[template][feat])] = 1
                    disc.append(x)
                disc = np.hstack(disc)
            disc = pca_filter(disc)
            check_multicolinearity(disc)

            # ft features 
            emb_ft_odd = create_finetuned_features(formats, 
                                                    bench,  
                                                    split='odds',)
            emb_ft_even = create_finetuned_features(formats, 
                                                    bench,  
                                                    split='even',)

            Xs[bench][task.replace(".json", "")] = []

            for llm in range(n_llms):
                emb_ft = emb_ft_odd if llm % 2 == 0 else emb_ft_even

                # storing
                Xs[bench][task.replace(".json", "")].append([disc, emb, emb_ft]) 

        with open("data/Xs.pickle", "wb") as handle:
            pickle.dump(Xs, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    #create_Ys()
    create_Xs()
