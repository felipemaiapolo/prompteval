import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset

from utils import (
    MultiLabelRaschModel_ID_tokens,
    ModelConfig,
    n_examples_mmlu,
    build_feature_tensor,
    build_label_tensor,
    get_args,
    get_save_name,
    process_text,
    train_model,
    n_max_tasks
)

device = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(42)

# TODO:
# 3. generalise training to potential new benches -> standardise
# 4. refactor code

BASE_PATH = "./"

# Data hyperparameters
TRAIN_SPLIT_LLMS = "odds"
TRAIN_SPLIT_SIZE = 0.8
MAX_LENGTH = 300  # max length of tokenizer output

# Model hyperparameters
D = 25 # size of used latent representation space
CLS = False 

args = get_args()
print(args)

mmlu = args.bench == "MMLU"

# Selecting tasks to train
n_tasks = n_max_tasks[args.bench] if args.n_tasks == None else args.n_tasks
tasks = [s.replace(".csv", "") for s in [x[2] for x in os.walk(os.path.join(BASE_PATH, "data", "templates", args.bench))][0]]
tasks.sort()
tasks = tasks[:n_tasks]
print(tasks)

# Loading and encoding format templates; count them
formats = []
n_examples = []
n_formats = []
for i, task in enumerate(tasks):
    formats_df = pd.read_csv(os.path.join(BASE_PATH, "data", "templates", args.bench, f"{task}.csv"))
    formats += [process_text(text, id_token=True) for text in formats_df["prompt template"].tolist()]
    if task != "word_not_containing":
        n_examples.append(formats_df.samples[0] if not mmlu else n_examples_mmlu[i])
    else:
        # this one does not have the full Ys. Check at some point why.
        n_examples.append(26)
    n_formats.append(len(formats_df))

# Loading data and model splitting
with open(os.path.join(BASE_PATH, "data", "Ys.pickle"), "rb") as handle:
    Ys = pickle.load(handle)

used_llms = (
    list(range(0, len(Ys[args.bench][task]), 2))
    if TRAIN_SPLIT_LLMS == "even"
    else list(range(1, len(Ys[args.bench][task]), 2))
)
n_llms = len(used_llms)

# Initialize model, optimizer, scheduler
n_example_tokens = sum(n_examples)
config = config = ModelConfig(base_model=args.model_name, n_examples=n_example_tokens, n_llms=n_llms, d=D, cls=CLS, bias=False)
model = MultiLabelRaschModel_ID_tokens(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = LambdaLR(optimizer, 
                     lr_lambda=lambda step: (min(step / args.warmup_steps, 1) if step < args.warmup_steps else args.gamma ** (step - args.warmup_steps)),)

# Run tokenization of formats
formats_tokenized = model.tokenizer(
        formats, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH).to(device)

# Data splitting
train_indices = []
val_indices = []
used_Ys = []
n_previous_examples = 0
for n_examples_task, n_formats_task, task in zip(n_examples, n_formats, tasks):
    current_range = range(n_previous_examples, n_previous_examples + n_examples_task)
    # init empty tensor with separate labels for each task
    temp = np.zeros((n_llms, n_formats_task, n_example_tokens))
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

# Run training
model, val_losses, val_esterrors = train_model(
    model,
    optimizer,
    train_loader,
    val_loader,
    formats_tokenized=formats_tokenized,
    args=args,
    train_split_llms=TRAIN_SPLIT_LLMS,
    scheduler=scheduler,
)

# Save results
results_path = os.path.join(BASE_PATH, "results", "results_ft")
os.makedirs(results_path, exist_ok=True)
save_name = get_save_name(args, TRAIN_SPLIT_LLMS, "ID_token")

model.save_pretrained(os.path.join(results_path, f"{save_name}.pt"))

if args.push_to_hub:  
    name = f"id_token_{args.model_name.split('-')[0]}_{TRAIN_SPLIT_LLMS}_{args.bench}"  
    model.push_to_hub(name)

val_losses_df = pd.DataFrame({"val_losses": val_losses, "val_esterrors": val_esterrors})
val_losses_df.to_csv(os.path.join(results_path, f"{save_name}.csv"))

plt.figure()
plt.plot(np.array(val_losses))
plt.savefig(os.path.join(results_path, f"{save_name}.pdf"))

plt.figure()
plt.plot(np.array(val_esterrors), color="red")
plt.savefig(os.path.join(results_path, f"err_{save_name}.pdf"))
