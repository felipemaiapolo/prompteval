from sentence_transformers import SentenceTransformer
from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from sklearn.cluster import KMeans
import yaml
from tqdm.auto import tqdm

N_TEMPLATES = 100
D = 25
EMBEDDER='sentence-transformers/all-mpnet-base-v2'

part1 = [
    "As an exceptionally efficient assistant, you assess and choose the best large language models (LLMs) by evaluating the quality of their responses to specific instructions. This process will culminate in a leaderboard that highlights the most accurate and preferred answers by humans.",
    "You serve as a highly efficient assistant, evaluating and selecting the top large language models (LLMs) based on how well they respond to given instructions. This method will be used to generate a leaderboard featuring the most precise and human-favored answers.",
    "As a highly capable assistant, you evaluate and pick the best large language models (LLMs) by examining the quality of their responses to instructions. This evaluation process will result in a leaderboard that showcases the most accurate and preferred answers by people.",
    "In your role as an efficient assistant, you assess and select the premier large language models (LLMs) based on their response quality to specific instructions. This will create a leaderboard displaying the most accurate and human-preferred responses.",
    "You function as a highly effective assistant, evaluating and choosing the best large language models (LLMs) by analyzing the quality of their responses to given instructions. This evaluation will form a leaderboard reflecting the most precise and human-preferred answers.",
    "As a highly efficient assistant, your task is to evaluate and select the finest large language models (LLMs) based on the quality of their responses to given prompts. This will lead to the creation of a leaderboard that ranks the most accurate and favored answers by humans.",
    "You are an efficient assistant who evaluates and selects the top-performing large language models (LLMs) by examining the quality of their responses to specified instructions. This process will result in a leaderboard showcasing the most accurate and preferred answers by users.",
    "Serving as a highly efficient assistant, you assess and choose the best large language models (LLMs) by evaluating their response quality to particular instructions. This will generate a leaderboard that features the most accurate and preferred answers by humans.",
    "As a highly efficient assistant, you evaluate and pick the top large language models (LLMs) based on the quality of their responses to certain instructions. This process will create a leaderboard highlighting the most accurate and human-preferred responses.",
    "In your role as a highly efficient assistant, you assess and select the best large language models (LLMs) by analyzing the quality of their responses to given instructions. This will lead to a leaderboard that showcases the most accurate and favored answers by people."
]

part2 = [
    "I need a leaderboard for different large language models. I will give you prompts and their outputs from these models. Your job is to evaluate these responses and determine which model generates the best output from a human perspective.",
    "I require a ranking of various large language models. I'll provide the prompts given to these models along with their outputs. Your task is to assess these responses and select the model that offers the best human-like output.",
    "I need a leaderboard showcasing various large language models. I will supply you with prompts and the corresponding outputs from these models. Your role is to evaluate these responses and identify which model delivers the best human-appealing output.",
    "I am looking for a leaderboard of different large language models. You will receive prompts given to these models and their outputs. Your job is to evaluate these responses and choose the model that produces the best human-preferred output.",
    "I need a ranking system for several large language models. I will provide prompts and the resulting outputs from these models. Your task is to evaluate these responses and select the model that offers the best human-perspective output.",
    "I require a leaderboard for multiple large language models. I'll provide you with prompts and their respective outputs from these models. Your job is to assess these responses and determine which model generates the best output from a human point of view.",
    "I need a leaderboard for various large language models. I will supply prompts given to these models along with their outputs. Your task is to evaluate these responses and pick the model that delivers the best human-like output.",
    "I am looking for a ranking of different large language models. You'll receive the prompts and the resulting outputs from these models. Your role is to assess these responses and choose the model that produces the best human-preferred output.",
    "I require a leaderboard for various large language models. I'll provide you with prompts and the corresponding outputs from these models. Your job is to evaluate these responses and select the model that offers the best output from a human perspective.",
    "I need a ranking of multiple large language models. I will give you prompts and their outputs from these models. Your task is to assess these responses and determine which model delivers the best human-like output."
]

part3 = [
    "Presented here are the unordered outputs from various models. Each output is linked to a specific model, identified by a unique model ID.",
    "Here are the outputs from the models, listed in no particular order. Each output is connected to a distinct model, marked by a unique identifier.",
    "The unordered outputs from the models are shown below. Each one corresponds to a particular model, identified by a unique ID.",
    "Below are the unordered outputs from the models. Each output is associated with a specific model, recognized by its unique identifier.",
    "Displayed here are the unordered outputs from different models. Each output is tied to a particular model, identified by a unique model identifier.",
    "Here you have the unordered outputs from various models. Each output is linked to a distinct model, identified by its unique model ID.",
    "These are the unordered outputs from the models. Each output corresponds to a specific model, marked by a unique identifier.",
    "Here are the unordered outputs from several models. Each output is associated with a particular model, identified by a unique ID.",
    "The following are the unordered outputs from the models. Each output is tied to a specific model, identified by a unique identifier.",
    "Here are the unordered outputs generated by the models. Each output is linked to a unique model identifier."
]

part4 = [
    "Assess the models by examining the quality and relevance of their outputs, and choose the one that produced the best result. Respond with the model identifier of the best model. We will use your answer as the name of the best model, so ensure your output includes only one of the following identifiers: m or M.",
    "Evaluate the models based on the relevance and quality of their outputs, and select the model that generated the top output. Reply with the model identifier of the best one. Your output will serve as the name of the best model, so make sure it contains only one of these identifiers: m or M.",
    "Judge the models on the quality and relevance of their outputs, and choose the one that delivered the best result. Provide the model identifier of the best model in your response. We will use your output as the name of the best model, so ensure it contains only one of these identifiers: m or M.",
    "Examine the outputs for quality and relevance, and select the model that produced the best one. Answer by giving the identifier of the best model. Your output will be used as the name of the best model, so make sure it includes only one of these identifiers: m or M.",
    "Evaluate the models on the basis of the quality and relevance of their outputs, and choose the model with the best output. Reply with the identifier of the best model. Your output will be used as the name of the best model, so ensure it contains only one of these: m or M.",
    "Assess the models by the quality and relevance of their outputs, and select the one that provided the best output. Respond with the identifier of the best model. Your output will become the name of the best model, so it should only contain one of these identifiers: m or M.",
    "Evaluate the models by checking the quality and relevance of their outputs, and choose the one that created the best result. Reply with the identifier of the best model. We will use your answer as the name of the best model, so make sure your response contains only one of these: m or M.",
    "Judge the models on their output quality and relevance, and select the one that generated the best result. Provide the identifier of the best model in your response. Your answer will be used as the name of the best model, so ensure it contains only one of these identifiers: m or M.",
    "Review the outputs for quality and relevance, and pick the model that produced the best one. Reply with the identifier of the best model. Your response will be used as the name of the best model, so make sure it includes only one of these identifiers: m or M.",
    "Evaluate the models based on the relevance and quality of their outputs, and select the one that created the best result. Respond with the identifier of the best model. Your answer will serve as the name of the best model, so it should only include one of these: m or M."
]

if __name__=="__main__":
    variations = []
    discrete_cov = []
    for i1,p1 in tqdm(enumerate(part1)):
        for i2,p2 in enumerate(part2):
            for i3,p3 in enumerate(part3):
                for i4,p4 in enumerate(part4):
                    content = f"""<|im_start|>system
{p1}
<|im_end|>

<|im_start|>user
{p2}

## Instruction

{{
    "instruction": """ + '"""{instruction}"""' + f""",
}}

## Model Outputs

{p3}

{{
    {{
        "model_identifier": "m",
        "output": """ + '"""{output_1}"""' + """
    }},
    {{
        "model_identifier": "M",
        "output": """ + '"""{output_2}"""' + f"""
    }}
}}

## Task

{p4}

## Best Model Identifier
<|im_end|>"""
                                
                    variations.append(content)

                    ### discrete cov
                    x1 = np.zeros(len(part1))
                    x2 = np.zeros(len(part2))
                    x3 = np.zeros(len(part3))
                    x4 = np.zeros(len(part4))
                    x1[i1] = 1
                    x2[i2] = 1
                    x3[i3] = 1
                    x4[i4] = 1
                    x1 = x1[:-1]
                    x2 = x2[:-1]
                    x3 = x3[:-1]
                    x4 = x4[:-1]
                    discrete_cov.append(np.hstack([x1,x2,x3,x4]))

    #Clustering
    print("Embedding templates...")
    model = SentenceTransformer(EMBEDDER) 
    embeddings = model.encode(variations)
    print("Clustering templates...")
    simil = cosine_similarity(embeddings)
    X = SpectralEmbedding(n_components=N_TEMPLATES, random_state=0, affinity='precomputed').fit_transform(simil) 

    seed=0
    while True:
        kmeans = KMeans(n_clusters=N_TEMPLATES, random_state=seed, n_init="auto").fit(X)
        selected_templates = pairwise_distances(kmeans.cluster_centers_, X, metric='euclidean').argmin(axis=1)
        if np.unique(selected_templates).shape[0]==N_TEMPLATES:
            break
        else:
            seed+=1

    #Saving
    i=0
    for template in selected_templates:
        with open(f'../data/templates/AlpacaEval/templates/template_{i}.txt', "w") as file:
            file.write(variations[template])
        i+=1

    ### Generating configs
    for i in range(N_TEMPLATES):
        yaml_content =f"""weighted_alpaca_eval_gpt4o_mini:
          prompt_template: "templates/template_{i}.txt" #evaluators_configs/
          fn_completions: "openai_completions"
          completions_kwargs:
            model_name: "gpt-4o-mini-2024-07-18" # "gpt-4-1106-preview"
            max_tokens: 1
            temperature: 1 # temperature should be applied for sampling, so that should make no effect.
            logprobs: true
            top_logprobs: 5
          fn_completion_parser: "logprob_parser"
          completion_parser_kwargs:
            numerator_token: "m"
            denominator_tokens: ["m", "M"]
            is_binarize: false
          completion_key: "completions_all"
          batch_size: 1"""
        yaml_dict = yaml.safe_load(yaml_content)
        with open(f"../data/templates/AlpacaEval/configs/config_{i}.yaml", "w") as file:
            yaml.dump(yaml_dict, file, default_flow_style=False)

    ### Saving cov
    pca = PCA(n_components=D)
    embeddings = pca.fit_transform(embeddings[selected_templates])
    np.save('../data/embeddings_llm_judge.npy', embeddings)