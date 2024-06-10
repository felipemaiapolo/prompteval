import numpy as np
from sklearn.decomposition import PCA  # type: ignore


def flatten(xss):
    """
    Flattens a list of lists into a single list.

    Parameters:
    xss (list of list): List of lists to be flattened.

    Returns:
    list: A single list with all the elements of the sub-lists.
    """
    return [x for xs in xss for x in xs]


def pca_filter(X, pca_tol=1e-5):
    """
    Applies PCA to the input data and filters out directions with no variation.

    Parameters:
    X (array-like): The input data.
    pca_tol (float): The tolerance level for explained variance ratio. Directions with a variance ratio less than this value are removed.

    Returns:
    array-like: The transformed data with reduced dimensions.
    """
    pca = PCA().fit(X)
    return pca.transform(X)[:, pca.explained_variance_ratio_ > pca_tol]


def check_multicolinearity(X, tol=1e-6):
    """
    Checks for multicollinearity in the input data by ensuring that the matrix of covariates is of full (column) rank when the intercept is included.

    Parameters:
    X (array-like): The input data.
    tol (float): The tolerance level for explained variance ratio.

    Raises:
    AssertionError: If the covariance matrix of X is not full rank.
    """
    pca = PCA().fit(X)
    assert (
        np.mean(pca.explained_variance_ratio_ > tol) == 1
    ), f"The covariance matrix of X should be full rank. We have pca.explained_variance_ratio_.min()={pca.explained_variance_ratio_.min()}"
