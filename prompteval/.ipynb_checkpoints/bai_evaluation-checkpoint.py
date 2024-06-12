import argparse
import copy
import pickle
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from methods import ExtendedRaschModel, GenXY, LogReg, StratSample
from utils import flatten

warnings.filterwarnings("ignore", category=ConvergenceWarning)
# python bai_evaluation.py --bench 'BBH' --random_seeds 5


### Functions
class MLPClassifierWithCV:
    """
    Multi-layer Perceptron classifier with built-in cross-validation for hyperparameter tuning.
    """

    def __init__(
        self,
        alphas=np.logspace(-3, 1, 5),
        learning_rate_inits=np.logspace(-3, -1, 4),
        hidden_layer_sizes=(30,),
        max_iter=200,
        cv=3,
        random_state=42,
    ):
        """
        Initializes the classifier with the given hyperparameters.

        Parameters:
        alphas (array-like): Regularization strengths.
        learning_rate_inits (array-like): Initial learning rates.
        hidden_layer_sizes (tuple): The ith element represents the number of neurons in the ith hidden layer.
        max_iter (int): Maximum number of epochs for training.
        cv (int): Number of folds in cross-validation.
        random_state (int): Seed for randomness to ensure reproducibility.
        """
        self.alphas = alphas
        self.learning_rate_inits = learning_rate_inits
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.cv = cv
        self.random_state = random_state
        self.early = True

    def fit(self, X, y):
        """
        Fits the MLPClassifier to the data, tuning hyperparameters with cross-validation.

        Parameters:
        X (numpy.ndarray): Training data.
        y (numpy.ndarray): Target values.

        Returns:
        self: Fitted estimator.
        """

        # Define a pipeline with scaling and MLP classifier
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        max_iter=self.max_iter,
                        hidden_layer_sizes=self.hidden_layer_sizes,
                        early_stopping=self.early,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

        # Define the parameter grid
        param_grid = {"mlp__alpha": self.alphas, "mlp__learning_rate_init": self.learning_rate_inits}

        # Initialize GridSearchCV
        grid_search = GridSearchCV(pipeline, param_grid, cv=self.cv, n_jobs=1)

        cut = self.cv
        if min((y == 0).sum(), (y == 1).sum()) < cut:  # This is just a trick to run the Scikit-Learn implementation
            y_copy = copy.deepcopy(y)
            local_state = np.random.RandomState(0)
            ind = local_state.choice(len(y_copy), cut)
            y_copy[ind] = 1 - np.median(y_copy)
            grid_search.fit(X, y_copy)
        else:
            # Fit the GridSearchCV
            grid_search.fit(X, y)

        # Extract the best parameters
        best_alpha = grid_search.best_params_["mlp__alpha"]
        best_learning_rate_init = grid_search.best_params_["mlp__learning_rate_init"]

        # Fit the final model on the full dataset
        self.best_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        max_iter=self.max_iter,
                        hidden_layer_sizes=self.hidden_layer_sizes,
                        early_stopping=self.early,
                        alpha=best_alpha,
                        learning_rate_init=best_learning_rate_init,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

        if min((y == 0).sum(), (y == 1).sum()) < cut:  # This is just a trick to run the Scikit-Learn implementation
            self.best_model.fit(X, y_copy)
        else:
            self.best_model.fit(X, y)

        return self.best_model

    def predict(self, X):
        """
        Predicts the labels for the input samples.

        Parameters:
        X (numpy.ndarray): Input data.

        Returns:
        numpy.ndarray: Predicted labels.
        """
        if self.best_model is None:
            raise ValueError("The model has not been fitted yet. Call the 'fit' method first.")
        return self.best_model.predict(X)

    def predict_proba(self, X):
        """
        Predicts the probability estimates for the input samples.

        Parameters:
        X (numpy.ndarray): Input data.

        Returns:
        numpy.ndarray: Probability estimates of the positive class.
        """
        if self.best_model is None:
            raise ValueError("The model has not been fitted yet. Call the 'fit' method first.")
        return self.best_model.predict_proba(X)

    def get_best_params(self):
        """
        Retrieves the best hyperparameters after fitting.

        Returns:
        dict: A dictionary of the best hyperparameters.
        """
        if self.best_model is None:
            raise ValueError("The model has not been fitted yet. Call the 'fit' method first.")
        return self.best_model.get_params()


class MLP:
    """A wrapper class for a Multi-layer Perceptron classifier with cross-validation."""

    def __init__(self):
        """Initializes an instance of the MLP class."""
        pass

    def fit(self, seen_items, Y, X=None):
        """
        Fits a multi-layer perceptron model to the data.

        Parameters:
        seen_items (numpy.ndarray): A boolean array indicating which items have been seen.
        Y (numpy.ndarray): The labels or responses for each format-example pair.
        X (numpy.ndarray, optional): The format covariates. If not provided, an identity matrix is used.

        The function process:
        - Determines format and example covariates based on inputs.
        - Reformats the data into a features matrix and a labels vector.
        - Fits the MLPClassifierWithCV model.
        - Predicts probabilities for each format's effectiveness.
        """
        # X (formats covariates)
        if type(X) != np.ndarray:
            self.X = np.eye(Y.shape[0])
        else:
            self.X = X
        self.x_dim = self.X.shape[1]

        # Z (examples covariates)
        self.Z = np.eye(Y.shape[1])
        self.z_dim = self.Z.shape[1]

        # Formatting the data
        self.n_formats, self.n_examples = seen_items.shape
        features, labels = GenXY(seen_items, Y, self.X, self.Z)
        features = features[:, : self.x_dim]

        # Fitting the model
        self.model = MLPClassifierWithCV()
        self.model.fit(features, labels)

        # Predicted probs
        self.thetas = self.model.predict_proba(self.X)[:, 1]


def compute_regrets(budget, Y, X, Z=None, random_seed=None):
    """
    Computes the regrets (in best prompt identification) for different models under a given budget.

    Parameters:
    budget (int): The total budget for evaluation.
    Y (numpy.ndarray): The labels or responses for each format-example pair.
    X (numpy.ndarray): The format covariates. If not provided, an identity matrix is used.
    Z (numpy.ndarray, optional): The example covariates.
    random_seed (int, optional): The seed for random operations to ensure reproducibility.

    Returns:
    list: A list of the computed regrets for the provided budget and models.
    """
    regrets = []

    n_formats, n_examples = Y.shape
    n_phases = int(np.ceil(np.log2(n_formats)))
    budget_phase = int(np.floor(budget / n_phases))
    extra_first_phase = budget - n_phases * budget_phase

    for mod in ["logreg", "mlp", "pe"]:

        active_arms = list(range(n_formats))

        if mod in ["logreg", "mlp"]:
            random_column = True
        else:
            random_column = False

        seen_examples = np.zeros(Y.shape).astype(bool)

        for phase in range(n_phases):
            if phase == 0:
                seen_examples = StratSample(
                    seen_examples, budget_phase + extra_first_phase, random_seed, active_arms, random_column
                )
            else:
                seen_examples = StratSample(
                    seen_examples, seen_examples.sum() + budget_phase, random_seed, active_arms, random_column
                )

            # If we have seen all examples from active arms, we stop
            if seen_examples[active_arms].mean() == 1:
                break

            # Fit IRT model
            if mod == "logreg":
                rasch_model = LogReg()
                rasch_model.fit(seen_examples, Y, X)
                mu = rasch_model.thetas[active_arms]
            elif mod == "mlp":
                rasch_model = MLP()
                rasch_model.fit(seen_examples, Y, X)
                mu = rasch_model.thetas[active_arms]
            else:
                rasch_model = ExtendedRaschModel()
                rasch_model.fit(seen_examples, Y, X, Z)
                mu = rasch_model.get_Y_hat().mean(-1)[active_arms]

            # Eliminating arms
            n_active = len(active_arms)
            n_eliminate = int(np.ceil(n_active / 2))
            if n_active - n_eliminate <= 0:
                break
            else:
                active_arms = [active_arms[i] for i in np.argsort(mu)[n_eliminate:].tolist()]

        ba = np.argmax([Y[active_arms[i]][seen_examples[active_arms][i]].mean() for i in range(len(active_arms))])
        ba = active_arms[ba]
        regrets.append(Y.mean(-1).max() - Y.mean(-1)[ba])

    return regrets


def evaluate_bai(Y, Xs, random_seed):
    """
    Evaluates the Best Arm Identification (BAI) performance across multiple budgets and contexts.

    BAI is a process in finding the best 'arm' or decision option. This function applies BAI
    to different evaluation contexts and budgets using a specified set of inputs and comparison models.

    Parameters:
    Y (numpy.ndarray): A matrix of true outcomes or labels for various test examples across different test formats.
    Xs (list of numpy.ndarray): A list of feature matrices representing the formats covariates used when fitting models.
    random_seed (int): An integer seed for ensuring deterministic behavior in randomized processes.

    Returns:
    list: A nested list of regret measurements for each budget and set of covariates.
           Each sublist corresponds to a budget and contains regrets for models using no covariates,
           covariates indexed by 0, and covariates indexed by 6 from the Xs list.
    """

    regrets = []

    ### running
    for budget in budgets:
        regrets.append([])

        # n_formats = Y.shape[0]
        # X = np.eye(n_formats)[:,1:]
        regrets[-1].append(compute_regrets(budget, Y, None, None, random_seed))

        for X in Xs:
            regrets[-1].append(compute_regrets(budget, Y, X, None, random_seed))

    return regrets


# Running script
if __name__ == "__main__":

    ### Definitions
    data_path = "../data/"
    results_path = "../results/"
    budgets = [200, 400, 800, 1600]  # ordered budgets

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
        delayed(evaluate_bai)(Ys[bench][job[1]][job[0]], Xs[bench][job[1]][job[0]], job[2]) for job in jobs
    )
    np.save(results_path+f"bai_results_{bench}.npy", {"out": results})

    results_dic = {}
    for task in tasks:
        results_dic[task] = []
        for llm in range(len(Ys[bench][task])):
            results_dic[task].append([])
    for i, job in enumerate(jobs):
        task = job[1]
        llm = job[0]
        results_dic[task][llm].append(results[i])

    final_results = np.stack([np.stack(results_dic[task]).mean(0).mean(0) for task in tasks])
    np.save(results_path+f"bai_processed_results_{bench}.npy", final_results)
