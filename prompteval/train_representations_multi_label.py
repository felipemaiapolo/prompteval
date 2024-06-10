import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from prompteval.utils.utils_ft import (
    MultiLabelRaschModel,
    MultiLabelRaschModel_ID_tokens,
    OneHotRaschModel,
    build_feature_tensor,
    build_label_tensor,
    get_args,
    get_examples_per_task_mmlu,
    get_save_name,
    process_text,
    task_order_mmlu,
    train_model,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(42)

# TODO:
# 1. change save names according to create_data.py
# 2. change the way that the model is saved
# 3. generalise training to potential new benches -> standardise
# 4. refactor code

BASE_PATH = "./"

# Data hyperparameters
TRAIN_SPLIT_LLMS = "odds"
TRAIN_SPLIT_SIZE = 0.8
MAX_LENGTH = 200  # max length of tokenizer output

ALT_SCHEDULER = False
CLS = False
ID_TOKEN = True

args = get_args()
print(args)

mmlu = args.bench == "MMLU"

# Selecting task to train
if not mmlu:
    tasks = [
        s.replace(".csv", "") for s in [x[2] for x in os.walk(os.path.join(BASE_PATH, "templates", args.bench))][0]
    ]
    tasks.sort()
else:
    tasks = task_order_mmlu

tasks = tasks[: args.n_tasks]
print(tasks)

mmlu_data_dir = "../data/mmlu/mmlu_output_codellama_codellama-34b-instruct/t_0"

n_examples_mmlu = get_examples_per_task_mmlu(mmlu_data_dir, tasks)
n_example_tokens = 100 * args.n_tasks if not mmlu else sum(n_examples_mmlu)

if args.model_name != "OneHotRaschModel":
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if ID_TOKEN:
        example_tokens = [f"[Example_{i}]" for i in range(0, n_example_tokens)]
        num_added_tokens = tokenizer.add_tokens(["[INPUT]", "[BR]", "[DATA_ID]"] + example_tokens)
        example_token_lookup = {
            n_example: token
            for n_example, token in zip(range(0, n_example_tokens), tokenizer.convert_tokens_to_ids(example_tokens))
        }
        example_token_lookup.update({"[DATA_ID]": tokenizer.convert_tokens_to_ids("[DATA_ID]")})
    else:
        num_added_tokens = tokenizer.add_tokens(["[INPUT]", "[BR]"])
        example_token_lookup = None
else:
    tokenizer = None


# Loading and encoding format templates
formats = []
n_examples = []
n_formats = []
for i, task in enumerate(tasks):
    formats_df = pd.read_csv(os.path.join(BASE_PATH, "templates", args.bench, f"{task}.csv"))
    formats += [process_text(text, id_token=ID_TOKEN) for text in formats_df["prompt template"].tolist()]
    if task != "word_not_containing":
        n_examples.append(formats_df.samples[0] if not mmlu else n_examples_mmlu[i])
    else:
        # this one does not have the full Ys. Check at some point why.
        n_examples.append(26)
    n_formats.append(len(formats_df))

if args.model_name == "OneHotRaschModel":
    formats_tokenized = None
else:
    formats_tokenized = tokenizer(
        formats, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH
    ).to(device)

# Loading data and model splitting
with open(os.path.join(BASE_PATH, "data", "Ys.pickle"), "rb") as handle:
    Ys = pickle.load(handle)

used_llms = (
    list(range(0, len(Ys[args.bench][task]), 2))
    if TRAIN_SPLIT_LLMS == "even"
    else list(range(1, len(Ys[args.bench][task]), 2))
)

# Data splitting
n_total_examples = sum(n_examples)
n_llms = len(used_llms)
train_indices = []
val_indices = []
used_Ys = []
n_previous_examples = 0
for n_examples_task, n_formats_task, task in zip(n_examples, n_formats, tasks):
    current_range = range(n_previous_examples, n_previous_examples + n_examples_task)
    # init empty tensor with separate labels for each task
    temp = np.zeros((n_llms, n_formats_task, n_total_examples))
    # add labels at corresponding indices
    temp[:, :, current_range] = np.concatenate([np.expand_dims(Ys[args.bench][task][i], 0) for i in used_llms])

    used_Ys.append(temp)

    for j in range(n_formats_task):
        train_indices.append(
            np.random.choice(n_examples_task, int(TRAIN_SPLIT_SIZE * n_examples_task), replace=False)
            + n_previous_examples
        )
        val_indices.append([k for k in current_range if k not in train_indices[-1]])
    n_previous_examples += n_examples_task
used_Ys = np.concatenate(used_Ys, axis=1)


# Prepare data
X_train, format_IDs_train = build_feature_tensor(train_indices, sum(n_examples))
X_val, format_IDs_val = build_feature_tensor(val_indices, sum(n_examples))

Y_train = build_label_tensor(train_indices, used_Ys)
Y_val = build_label_tensor(val_indices, used_Ys)

dataset = TensorDataset(X_train, format_IDs_train, Y_train)
val_dataset = TensorDataset(X_val, format_IDs_val, Y_val)

train_loader = DataLoader(dataset, batch_size=args.bs, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=int(args.bs_val), shuffle=False)

# Initialize model
if args.model_name != "OneHotRaschModel":
    model = (
        MultiLabelRaschModel(args.model_name, n_examples, n_llms, cls=CLS).to(device)
        if not ID_TOKEN
        else MultiLabelRaschModel_ID_tokens(args.model_name, n_examples, n_llms, example_token_lookup, cls=CLS).to(
            device
        )
    )
    model.model.resize_token_embeddings(len(tokenizer))
else:
    model = OneHotRaschModel(n_examples, n_llms).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if ALT_SCHEDULER:
    factor = 0.5
    patience = 3
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=factor, patience=patience)
else:
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: (
            min(step / args.warmup_steps, 1) if step < args.warmup_steps else args.gamma ** (step - args.warmup_steps)
        ),
    )

# Run training
model, val_losses, val_esterrors = train_model(
    model,
    optimizer,
    train_loader,
    val_loader,
    formats_tokenized,
    scheduler=scheduler,
    n_epochs=args.n_epochs,
    print_step=args.print_step,
)

# Save results
results_path = os.path.join(BASE_PATH, "results", "results_ft")
os.makedirs(results_path, exist_ok=True)
save_name = get_save_name(args, TRAIN_SPLIT_LLMS, "ID_token")

try:
    torch.save(model.state_dict(), os.path.join(results_path, f"{save_name}.pt"))
except:  # noqa
    print("No state_dict for saving")

val_losses_df = pd.DataFrame({"val_losses": val_losses, "val_esterrors": val_esterrors})
val_losses_df.to_csv(os.path.join(results_path, f"{save_name}.csv"))

plt.figure()
plt.plot(np.array(val_losses))
plt.savefig(os.path.join(results_path, f"{save_name}.pdf"))

plt.figure()
plt.plot(np.array(val_esterrors), color="red")
plt.savefig(os.path.join(results_path, f"err_{save_name}.pdf"))
