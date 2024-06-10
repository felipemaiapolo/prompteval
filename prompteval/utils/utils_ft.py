import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

## GENERAL
BASE_PATH = "./"


def get_args():
    parser = argparse.ArgumentParser(description="Train representations")

    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Name of the model to train")
    parser.add_argument("--print_step", type=int, default=20, help="Number of steps between each print")
    parser.add_argument("--lr", type=float, default=5e-2, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay")
    parser.add_argument("--gamma", type=float, default=0.999, help="LR decay rate")
    parser.add_argument("--bs", type=int, default=64, help="Batch size")
    parser.add_argument("--bs_val", type=int, default=128, help="Batch size")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps")
    parser.add_argument("--exp", type=str, default="", help="additional description for experiment")
    parser.add_argument("--n_tasks", type=int, default="", help="number of tasks to train on")
    parser.add_argument("--bench", type=str, default="BBH", help="which bench to train on")

    return parser.parse_args()


def get_save_name(args, train_split_llms, type_training, epochs=None):
    return (
        f'{type_training}_{args.model_name.replace("/", "-")}_{train_split_llms}_{args.bench}_'
        f"epochs_{args.n_epochs if epochs is None else epochs}_lr_{args.lr}_BS_{args.bs}_warmup_{args.warmup_steps}_gamma_{args.gamma}_"
        f"weight_decay_{args.weight_decay}_{args.exp}_n_tasks_{args.n_tasks}"
    )


## TRAINING FUNCTIONS


def train_model(
    model,
    optimizer,
    train_loader,
    val_loader,
    formats_tokenized,
    args,
    TRAIN_SPLIT_LLMS,
    scheduler=None,
    n_epochs=10,
    print_step=5,
):
    val_losses = []
    val_esterrors = []

    for epoch in range(n_epochs):
        model.train()
        step = 1

        for x, format_ids, y in train_loader:
            x, y, format_ids = x.to(device), y.to(device), format_ids.to(device)
            probs = model.forward(x, format_ids, formats_tokenized=formats_tokenized)
            loss = F.binary_cross_entropy(probs.squeeze(), y.squeeze().float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if step % print_step == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    val_esterror = 0
                    max_evals = 20
                    for it, (x_val, format_ids_val, y_val) in enumerate(val_loader):
                        x_val, y_val = x_val.to(device), y_val.to(device)
                        probs = model.forward(x_val, format_ids_val, formats_tokenized=formats_tokenized)
                        val_loss += F.binary_cross_entropy(probs.squeeze(), y_val.squeeze().float()).cpu().item()
                        val_esterror += torch.abs(y_val - probs).mean().item()
                        if it > max_evals:
                            break
                    val_loss /= it
                    val_esterror /= it

                print(
                    f"Epoch [{epoch+1}/{n_epochs}], Step [{step + 1}/{len(train_loader)}], Training Loss: {np.round(loss.item(),6)}, Validation Loss: {np.round(val_loss,6)}, Validation error: {np.round(val_esterror,6)}"
                )

                model.train()
                val_losses.append(val_loss)
                val_esterrors.append(val_esterror)

            step += 1
        results_path = os.path.join(BASE_PATH, "results", "results_ft")
        os.makedirs(results_path, exist_ok=True)
        save_name = get_save_name(args, TRAIN_SPLIT_LLMS, "ID_token", epochs=epoch + 1)

        torch.save(model.state_dict(), os.path.join(results_path, f"{save_name}.pt"))

        val_losses_df = pd.DataFrame({"val_losses": val_losses, "val_esterrors": val_esterrors})
        val_losses_df.to_csv(os.path.join(results_path, f"{save_name}.csv"))

        plt.figure()
        plt.plot(np.array(val_losses))
        plt.savefig(os.path.join(results_path, f"{save_name}.pdf"))

        plt.figure()
        plt.plot(np.array(val_esterrors), color="red")
        plt.savefig(os.path.join(results_path, f"err_{save_name}.pdf"))

    return model, val_losses, val_esterrors


## MODELS


class MultiLabelRaschModel_ID_tokens(nn.Module):
    def __init__(self, model_name, n_examples, n_llms, example_token_lookup, d=25, cls=False, bias=False):
        super(MultiLabelRaschModel_ID_tokens, self).__init__()
        self.cls = cls
        self.lookup = example_token_lookup
        self.ID_token = example_token_lookup["[DATA_ID]"] if "[DATA_ID]" in example_token_lookup else None

        self.model = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.model.config.hidden_size, d, bias=bias)
        self.classifier = nn.Linear(d, n_llms, bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, examples_one_hot, format_indices, formats_tokenized=None):
        # convert to list and unnest
        format_indices = sum(format_indices.tolist(), [])
        input_ids, attention_mask = formats_tokenized["input_ids"][format_indices].to(device), formats_tokenized[
            "attention_mask"
        ][format_indices].to(device)

        # Replace the ID token with the respective identity of the example
        example_token_vector = torch.tensor(
            [self.lookup[ind.item()] for ind in torch.argmax(examples_one_hot, dim=1)]
        ).to(device)
        id_token_positions = input_ids == self.ID_token

        input_ids[id_token_positions] = example_token_vector

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        representation = (
            outputs.last_hidden_state[:, 0, :] if self.cls else outputs.last_hidden_state.mean(dim=1).squeeze()
        )
        projected_rep = self.projection(representation)

        logits = self.classifier(projected_rep)
        probs = self.sigmoid(logits)

        return probs.squeeze()

    def extract_representation(self, x):
        outputs = self.model(**x)
        representation = (
            outputs.last_hidden_state[:, 0, :] if self.cls else outputs.last_hidden_state.mean(dim=1).squeeze()
        )
        projected_rep = self.projection(representation)
        return projected_rep


## DATA PREPROCESSING


def process_text(text, id_token=False):
    pattern = r"\{[^}]*\}"
    replaced_text = re.sub(pattern, "[INPUT]", text)
    replaced_text = replaced_text.replace("\n", "[BR]")
    if id_token:
        replaced_text += "[DATA_ID]"
    return replaced_text


def get_one_hot(j, n_examples=None):
    x = torch.zeros(n_examples)
    x[j] = 1
    return x.float()


def build_feature_tensor(indices_list, n_examples):
    X = []
    format_ids = []
    for format_id, indices_format in enumerate(indices_list):
        X.append([])
        for index in indices_format:
            format_ids.append(format_id)
            one_hot = get_one_hot(index, n_examples)
            X[-1].append(one_hot)
        X[-1] = torch.stack(X[-1])

    X = torch.cat(X)
    return X, torch.tensor(format_ids).unsqueeze(dim=1)


def build_label_tensor(indices_list, Ys):
    return torch.tensor(np.hstack([Ys[:, i, indices] for i, indices in enumerate(indices_list)]).T)


def load_data(dir, bench):
    """Get data for full prompts"""

    def load_file(file_name):
        with open(file_name, "r") as f:
            data = json.load(f)
        return data

    all_data = {}
    if bench in ["BBH", "LMentry"]:
        tasks = os.listdir(dir)

        for task in tasks:
            all_data[task] = {}
            files = os.listdir(os.path.join(dir, task))
            for file in files:
                model_name = file.split("_")[-4]
                all_data[task][model_name] = load_file(os.path.join(dir, task, file))

    elif bench == "MMLU":
        raise NotImplementedError("Use dedicated function for mmlu (get_full_prompts_and_scores_mmlu)")
    else:
        raise NotImplementedError(f"Benchmark {bench} not implemented")

    return all_data


def get_full_prompts_and_scores(data, bench, tasks=["causal_judgement"]):
    """Get full prompts (template + datapoint) and scores of all models"""
    models = list(data[tasks[0]].keys())
    prompts = []
    correctnesses = []
    if bench in ["BBH", "LMentry"]:
        for k, model in enumerate(models):
            prompts.append([])
            correctnesses.append([])
            for task in tasks:
                for example in data[task][model]["samples"].keys():
                    for type_template in data[task][model]["samples"][example].keys():
                        for template_datapoint in data[task][model]["samples"][example][type_template].keys():
                            prompt = data[task][model]["samples"][example][type_template][template_datapoint]["prompt"]
                            if k == 0:
                                prompts[-1].append(prompt)
                            correctness = data[task][model]["samples"][example][type_template][template_datapoint][
                                "auto_validation"
                            ]
                            correctnesses[-1].append(torch.tensor([correctness]))
    elif bench == "MMLU":
        raise NotImplementedError("Use dedicated function for mmlu (get_full_prompts_and_scores_mmlu)")
    else:
        raise NotImplementedError(f"Benchmark {bench} not implemented")

    return prompts, torch.tensor(correctnesses), models


def get_tasks_mmlu(data_dir):
    tasks = []
    model_folder = os.listdir(data_dir)[0]
    template_folder = os.listdir(os.path.join(data_dir, model_folder))[0]
    tasks = [
        f.split(".")[1]
        for f in os.listdir(os.path.join(data_dir, model_folder, template_folder))
        if f != "results.json"
    ]
    return tasks


def get_examples_per_task_mmlu(data_dir, tasks):
    n_examples = []
    files = os.listdir(data_dir)

    for task in tasks:
        file = [f for f in files if task in f][0]
        if task in tasks:
            data = json.load(open(os.path.join(data_dir, file), "r"))
            n_examples.append(len(data))

    return n_examples


n_max_tasks = {"BBH": 15, "LMentry": 9, "MMLU": 57}

task_order_mmlu = [
    "global_facts",
    "abstract_algebra",
    "high_school_world_history",
    "professional_law",
    "high_school_physics",
    "management",
    "high_school_microeconomics",
    "conceptual_physics",
    "logical_fallacies",
    "electrical_engineering",
    "us_foreign_policy",
    "high_school_geography",
    "clinical_knowledge",
    "computer_security",
    "miscellaneous",
    "high_school_us_history",
    "prehistory",
    "jurisprudence",
    "high_school_macroeconomics",
    "formal_logic",
    "high_school_chemistry",
    "high_school_psychology",
    "public_relations",
    "moral_scenarios",
    "anatomy",
    "high_school_statistics",
    "college_mathematics",
    "econometrics",
    "sociology",
    "college_computer_science",
    "moral_disputes",
    "nutrition",
    "college_medicine",
    "high_school_government_and_politics",
    "virology",
    "high_school_biology",
    "college_chemistry",
    "high_school_mathematics",
    "college_biology",
    "marketing",
    "philosophy",
    "international_law",
    "human_sexuality",
    "machine_learning",
    "college_physics",
    "human_aging",
    "professional_psychology",
    "medical_genetics",
    "security_studies",
    "professional_accounting",
    "astronomy",
    "business_ethics",
    "world_religions",
    "elementary_mathematics",
    "high_school_computer_science",
    "high_school_european_history",
    "professional_medicine",
]

llms = [
    "mmlu_output_meta-llama_llama-3-8b",
    "mmlu_output_meta-llama_llama-3-8b-instruct",
    "mmlu_output_meta-llama_llama-3-70b-instruct",
    "mmlu_output_codellama_codellama-34b-instruct",
    "mmlu_output_google_flan-t5-xl",
    "mmlu_output_google_flan-t5-xxl",
    "mmlu_output_google_flan-ul2",
    "mmlu_output_ibm-mistralai_merlinite-7b",
    "mmlu_output_mistralai_mixtral-8x7b-instruct-v01",
    "mmlu_output_mistralai_mistral-7b-instruct-v0-2",
    "mmlu_output_google_gemma-7b",
    "mmlu_output_google_gemma-7b-it",
    "mmlu_output_tiiuae_falcon-40b",
    "mmlu_output_mistralai_mistral-7b-v0-1",
    "mmlu_output_tiiuae_falcon-180b",
]
