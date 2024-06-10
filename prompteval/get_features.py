import os

import numpy as np
import pandas as pd
import torch  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_templates(benchmark, dataset=None):
    template_dir = f"data/templates/{benchmark}"
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


def create_finetuned_features(
    templates_task,
    checkpoint_name,
    n_llms,
):
    """
    TODO:
    - change the saving of model state-dicts: we might want to upload those to HF and not store them in the repo
        - consequently, change the way that the model is loaded
    - refactor get_examples_per_task_mmlu + task_order_mmlu -> won't work in the package, so have to get it from somerwhere else
    """

    from prompteval.utils.utils_ft import (
        MultiLabelRaschModel_ID_tokens,
        get_examples_per_task_mmlu,
        n_max_tasks,
        task_order_mmlu,
    )

    ft_model_path = "./ft_models/used_models"

    model_base = "bert-base-uncased"
    bench = "BBH" if "BBH" in checkpoint_name else "LMentry" if "LMentry" in checkpoint_name else "MMLU"
    n_tasks = n_max_tasks[bench]

    mmlu_data_dir = "../data/mmlu/mmlu_output_codellama_codellama-34b-instruct/t_0"
    n_examples_mmlu = get_examples_per_task_mmlu(mmlu_data_dir, task_order_mmlu[:n_tasks])
    n_examples = 100 * n_tasks if bench != "MMLU" else sum(n_examples_mmlu)

    tokenizer = AutoTokenizer.from_pretrained(model_base)
    sentence_representations = []

    state_dict = torch.load(os.path.join(ft_model_path, checkpoint_name), map_location=device)

    example_tokens = [f"[Example_{i}]" for i in range(0, n_examples)]
    tokenizer.add_tokens(["[INPUT]", "[BR]", "[DATA_ID]"] + example_tokens)
    model = MultiLabelRaschModel_ID_tokens(model_base, n_examples, n_llms, {}, cls=False).to(device)
    model.model.resize_token_embeddings(len(tokenizer))

    model.load_state_dict(state_dict)

    for sentence in templates_task:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
        sentence_representation = model.extract_representation(inputs.to(device)).cpu().squeeze().detach().numpy()
        sentence_representations.append(sentence_representation)

    sentence_representations = np.vstack(sentence_representations)
    return sentence_representations
