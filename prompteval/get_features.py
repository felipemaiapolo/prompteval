import os
import numpy as np
import pandas as pd
import torch  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "CPU"
data_path = "../data/"

def load_templates(benchmark, dataset=None):
    template_dir = data_path+f"templates/{benchmark}"
    all_files = os.listdir(template_dir) if dataset is None else [f"{dataset}.csv"]
    templates = {file: pd.read_csv(os.path.join(template_dir, file)) for file in all_files}
    return templates


def get_framing_words(benchmark, dataset=None):
    templates = load_templates(benchmark, dataset=dataset)
    framing_words = []
    for task in templates.keys():
        prompts = templates[task]["prompt template"]
        for s in prompts:
            framing_words += [w for w in s.split() if ":" in w and (w[0].isupper() or w[0].isnumeric())]
    return list(set(framing_words))


def framing_words(s, benchmark=None, dataset=None):
    """Counts framing words (e.g. 'Answer: ') via heuristic"""
    words = s.split()
    if benchmark is not None:
        # getting framings for a specific dataset / benchmark
        framings = get_framing_words(benchmark, dataset=dataset)
    else:
        # getting framings for all benchmarks
        framings = list(set(word for benchmark in ("LMentry", "BBH", "MMLU") for word in get_framing_words(benchmark)))

    counts = []
    for framing in framings:
        counts += [sum(1 for word in words if word == framing)]
    return counts


def casing_features(s):
    """Counts uppercase, lowercase, and capitalized words."""
    words = s.split()
    uppercase = sum(1 for word in words if word.isupper())
    lowercase = sum(1 for word in words if word.islower())
    capitalized = sum(1 for word in words if word.istitle())
    return uppercase, lowercase, capitalized


def count_of_line_breaks(s):
    return s.count("\n")


def special_character_count(s):
    characters = [
        ":",
        "-",
        "||",
        "<sep>",
        "::",
        "",
        "(",
        ")",
        '"',
        "?",
    ]
    counts = [s.count(char) for char in characters]
    return counts


def count_spaces(s):
    return s.count(" ")


def vectorize_string(s, benchmark=None, dataset=None):
    return np.hstack(
        [
            casing_features(s),
            count_of_line_breaks(s),
            special_character_count(s),
            count_spaces(s),
            framing_words(s, benchmark=benchmark, dataset=dataset),
        ]
    )


def unnest_templates(templates_task):
    return [
        templates_task[type_template][template]
        for type_template in templates_task.keys()
        for template in templates_task[type_template]
    ]


def create_discrete_features(templates_task, benchmark=None, dataset=None):
    return np.vstack([vectorize_string(s, benchmark=benchmark, dataset=dataset) for s in templates_task])


def create_sentence_transformers(
    templates_task, pca_dim=20
):  # https://huggingface.co/facebook/dpr-question_encoder-multiset-base
    model = SentenceTransformer(
        "sentence-transformers/facebook-dpr-question_encoder-multiset-base"
    )  # 'paraphrase-MiniLM-L6-v2')
    embedding = model.encode(templates_task)
    pca = PCA(n_components=pca_dim)
    features_low_dim = pca.fit_transform(embedding)
    return features_low_dim


def create_finetuned_features(templates_task, 
                                bench, 
                                split,
                                ): 
    
    from utils import MultiLabelRaschModel_ID_tokens
                                           
    sentence_representations = []
        
    name = f'id_token_bert_{split}_{bench}'
    model = MultiLabelRaschModel_ID_tokens.from_pretrained(f'LucasWeber/{name}').to(device)

    for sentence in templates_task:
        inputs = model.tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
        sentence_representation = model.extract_representation(inputs.to(device)).cpu().squeeze().detach().numpy()
        sentence_representations.append(sentence_representation)
    
    sentence_representations = np.vstack(sentence_representations)
    return sentence_representations
