# Efficient multi-prompt evaluation of LLMs

Welcome to the [*PromptEval* GitHub repository](https://github.com/felipemaiapolo/prompteval)! Here you will find more information about our implementation of *PromptEval* and datasets introduced in

[Maia Polo, Felipe, Ronald Xu, Lucas Weber, Mírian Silva, Onkar Bhardwaj, Leshem Choshen, Allysson Flavio Melo de Oliveira, Yuekai Sun, and Mikhail Yurochkin. "Efficient multi-prompt evaluation of LLMs." arXiv preprint arXiv:2405.17202 (2024).](https://arxiv.org/abs/2405.17202)

## Overview

Most popular benchmarks for comparing LLMs rely on a limited set of prompt templates, which may not fully capture the LLMs’ abilities and can affect the reproducibility of results on leaderboards. This repository introduces our implementation of *PromptEval*, a method for estimating performance across a large set of prompts by borrowing strength across prompts and examples to produce accurate estimates under practical evaluation budgets.

**Quick start:** please check our [demo](https://github.com/felipemaiapolo/prompteval/blob/package/notebooks/PromptEval_demo.ipynb) on how to use *PromptEval* in your own data.

## Repository Structure

- `data/`: Contains the evaluation data used in the experiments.
- `prompteval/`: Source code for the PromptEval method and utilities.
- `notebooks/`: Jupyter notebooks used to create plots for the PromptEval paper.
- `results/`: Results from the experiments conducted in the paper.

## Installation

To use the code in this repository, clone the repo and install the required dependencies:

```bash
git clone https://github.com/felipemaiapolo/prompteval.git
cd prompteval
pip install -e .
```

## Reproducing the results of the paper

To reproduce the results in our paper, please follow the steps after cloning the repo and installing dependencies:
1. Download the BBH and LMentry data, produced by the authors of "[State of What Art? A Call for Multi-Prompt LLM Evaluation](https://arxiv.org/abs/2401.00595)", from [here](https://www.dropbox.com/scl/fo/y9dd8zbteyf0xrjxdtm3e/h/raw%20open-source%20model%20responses%20with%20gold%20and%20auto%20validation%20values.zip?rlkey=okp52gleuibw72fhe62egr6lp&e=1&dl=0). Place the unzipped folder "raw open-source model responses with gold and auto validation values" inside the data directory;
2. Process data by running ``create_data.py``;
3. Run main experiments by running ``dist_evaluation.py``. Example: ``python dist_evaluation.py --bench 'BBH' --random_seeds 5``;
4. Run best prompt identification by running ``bai_evaluation.py``. Example: ``python bai_evaluation.py --bench 'BBH' --random_seeds 5``.
5. Create plots using the notebooks in the notebooks directory.

## Fine-tuning embeddings

To fine-tune BERT representations run the following:

```bash
python ft_representations.py --model_name "bert-base-uncased" \
                             --lr 2e-05 \
                             --weight_decay 1e-06 \
                             --gamma .99995 \
                             --bs 96 \
                             --n_epochs 5 \
                             --warmup_steps 200 \
                             --bench "BBH" 
```
Note, that this requires the file `./data/Ys.pickle` to contain correctness data for the respective benchmark as it is create by the `create_data.py` scrip. Add the `--push_to_hub` flag, to automatically push your resulting model to your namespace on the huggingface hub.

## MMLU Data

We make our MMLU collected data available on [Hugging Face](https://huggingface.co/PromptEval). The data includes evaluation for 15 different SOTA LLMs and 100 different prompt templates.

## Citing

    @article{polo2024efficient,
    title={Efficient multi-prompt evaluation of LLMs},
    author={Polo, Felipe Maia and Xu, Ronald and Weber, Lucas and Silva, M{\'\i}rian and Bhardwaj, Onkar and Choshen, Leshem and de Oliveira, Allysson Flavio Melo and Sun, Yuekai and Yurochkin, Mikhail},
    journal={arXiv preprint arXiv:2405.17202},
    year={2024}
    }
