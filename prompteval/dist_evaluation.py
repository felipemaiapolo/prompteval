import argparse
import pickle

import numpy as np
from joblib import Parallel, delayed

from prompteval.methods import Baseline, PromptEval

# python dist_evaluation.py --bench 'BBH' --random_seeds 5


# Functions
def flatten(xss):
    """
    Flattens a list of lists.

    Parameters:
    xss (list of list): The list of lists to be flattened.

    Returns:
    list: A single flattened list containing all the elements from the sublists.
    """
    return [x for xs in xss for x in xs]


def evaluate(Y, Xs, random_seed):
    """
    Evaluates predictions using the PromptEval method and baseline for a given dataset.

    Parameters:
    Y (numpy.ndarray): The true labels for the evaluation dataset.
    Xs (list of numpy.ndarray): Different covariates for the specific task and llm.
    random_seed (int): The seed used for random operations, ensures reproducibility.

    Returns:
    tuple: Contains quantile estimates, individual estimates, and the number of observations seen by each method.
    """
    methods = []
    n_seen = []
    quantile_estimates = []
    individual_estimates = []

    ### running
    methods.append(Baseline())
    methods[-1].fit(Y, quantiles=quantiles, rounds_eval=rounds_eval, random_seed=random_seed)

    methods.append(PromptEval())
    methods[-1].fit(Y, quantiles=quantiles, rounds_eval=rounds_eval, random_seed=random_seed)

    for X in Xs:
        methods.append(PromptEval())
        methods[-1].fit(Y, quantiles=quantiles, rounds_eval=rounds_eval, X=X, random_seed=random_seed)

        methods.append(PromptEval())
        methods[-1].fit(Y, quantiles=quantiles, rounds_eval=rounds_eval, X=X, logreg=True, random_seed=random_seed)

    ### storing results
    for method in methods:
        n_seen.append(method.estimates["n_seen"])
        individual_estimates.append(method.estimates["accs_hat"])
        try:
            quantile_estimates.append(method.estimates["pirt"])
        except:  # noqa
            quantile_estimates.append(method.estimates["estimates"])

    return quantile_estimates, individual_estimates, n_seen


if __name__ == "__main__":

    ### Definitions
    data_path = "data/"
    quantiles = [5, 25, 50, 75, 95]
    rounds_eval = [200, 400, 800, 1600]  # ordered budgets

    ### Inputs
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--bench", type=str)
    parser.add_argument("--random_seeds", type=int)
    args = parser.parse_args()
    bench = args.bench
    random_seeds = list(range(args.random_seeds))

    ### loading data
    with open(data_path + "Ys.pickle", "rb") as handle:
        Ys = pickle.load(handle)
    with open(data_path + "Xs.pickle", "rb") as handle:
        Xs = pickle.load(handle)

    ### getting tasks
    tasks = list(Ys[bench].keys())

    ### defining jobs and running in parallel
    jobs = flatten(
        flatten(
            [
                [[(llm, task, random_seed) for llm in range(len(Ys[bench][task]))] for task in tasks]
                for random_seed in random_seeds
            ]
        )
    )
    print("benchmark:", bench, " n_jobs:", len(jobs))

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(evaluate)(Ys[bench][job[1]][job[0]], Xs[bench][job[1]][job[0]], job[2]) for job in jobs
    )
    np.save(f"results/results_{bench}.npy", {"out": results})

    ### saving processed results
    for res in range(3):  # three types of results
        if res == 0:
            results_processed = [r[0] for r in results]
        elif res == 1:
            results_processed = [r[1] for r in results]
        else:
            results_processed = [r[1] for r in results]

        results_dic = {}
        for task in tasks:
            results_dic[task] = []
            for llm in range(len(Ys[bench][task])):
                results_dic[task].append([])
        for i, job in enumerate(jobs):
            task = job[1]
            llm = job[0]
            Y = Ys[bench][task][llm]
            if res == 0:
                error = np.abs(np.array(results_processed[i]) - np.percentile(Y.mean(-1), quantiles)[None, None, :])
            elif res == 1:
                error = np.abs(np.array(results_processed[i]) - Y.mean(-1)[None, None, :])
            else:
                error = np.abs(
                    np.sort(np.array(results_processed[i]), axis=-1) - np.sort(Y.mean(-1)[None, None, :], axis=-1)
                )
            results_dic[task][llm].append(error)

        if res == 0:
            final_results = np.stack([np.stack(results_dic[task]) for task in tasks])
            np.save(f"results/processed_results_{bench}_quantiles_{quantiles}.npy", final_results)
        elif res == 1:
            final_results = np.stack([np.stack(results_dic[task]).mean(-1) for task in tasks])
            np.save(f"results/processed_results_{bench}_individual.npy", final_results)
        else:
            final_results = np.stack([np.stack(results_dic[task]).mean(-1) for task in tasks])
            np.save(f"results/processed_results_{bench}_dist.npy", final_results)
