""" Principal Component Analysis (PCA) Transformer. """
from typing import List, Union

import numpy as np
from scipy.special import ndtri
from scipy.stats import f
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import (
    check_array,
    check_consistent_length,
    check_is_fitted,
)


class PCA(BaseEstimator, TransformerMixin):
    """
    Principal Component Analysis (PCA) Transformer.

    Parameters:
    - n_components: int, optional (default=2)
      Number of components to keep.
    - mean_center: bool, optional (default=False)
      Whether to center the data by subtracting the mean.
    - confidence_level: float, optional (default=0.95)
      Confidence level for calculating Q residuals and Hotelling's T-squared.

    Attributes:
    - explained_variance_: array, shape (n_components,)
      Percentage of variance explained by each of the selected components.
    - explained_variance_accumulative: array, shape (n_components,)
      Cumulative explained variance.
    - loadings_: array, shape (n_features, n_components)
      Principal axes in feature space, representing the directions of maximum variance.
    - mean_: array, shape (n_features,)
      Mean of the input data if mean_center is True.
    - q_limit_: float
      Q limit for Q residuals at the specified confidence level.
    - q_residuals_: array, shape (n_samples,)
      Q residuals for the fitted data.
    - q_residuals_predicted_: array, shape (n_samples,)
      Predicted Q residuals for new samples.
    - t_hotelling_: array, shape (n_samples,)
      Hotelling's T-squared values for the fitted data.
    - t_hotelling_predicted_: array, shape (n_samples,)
      Predicted Hotelling's T-squared values for new samples.
    - t_limit_: float
      T-squared limit at the specified confidence level.
    """

    def __init__(self, n_components: int = 2, mean_center: bool = False,
                 confidence_level: float = 0.95):
        super().__init__()
        self.n_components = n_components
        self.mean_center = mean_center
        self.confidence_level = confidence_level

        # Define the variables that will be calculated
        self.__covariance_matrix = None
        self.mean_ = None
        self.loadings_ = None
        self.explained_variance_ = None
        self.explained_variance_accumulative = None
        self.q_residuals_ = None
        self.q_limit_ = None
        self.t_hotelling_ = None
        self.t_limit_ = None
        self._sigma = None
        self.q_residuals_predicted_ = None
        self.t_hotelling_predicted_ = None

    def fit(self, X, y=None):
        """
        Fit the PCA model to the input data.


        Parameters
        ----------
        X : array-like or pd.DataFrame, shape (n_samples, n_features). Missing values are not allowed.
        y : None, it is only present for compatibility with sklearn

        Returns
        -------
        self: PCA
        returns an instance of self.
        """
        X = check_array(X, accept_sparse=False, ensure_min_samples=5,
                        ensure_min_features=self.n_components)

        if self.mean_center:
            self.mean_ = np.mean(X, axis=0)
            X = X - self.mean_

        # number of samples
        n_samples = X.shape[0]

        # calculate the SVD of the data
        singular_vectors, singular_values, v_transpose = np.linalg.svd(X)

        # keep only the top n_components
        singular_vectors = singular_vectors[:, :self.n_components]
        singular_values = singular_values[:self.n_components]
        v_transpose = v_transpose[:self.n_components]

        # project the data onto the principal components
        scores = np.dot(X, v_transpose.T)

        # calculate the explained variance
        explained_variance = self.__calculate_explained_variance(
            singular_values, n_samples)

        # calculate q residuals
        q_residuals = self.__calculate_q_residuals(X, scores, v_transpose.T)

        # calculate q limit
        q_limit = self.__calculate_q_limit(singular_values, n_samples)

        # calculate hotelling t^2
        t_hotelling = self.__calculate_hotelling_t2(
            scores, np.diag(singular_values**2))

        # calculate t^2 limit
        t_limit = self.__calculate_hotelling_t2_limit(n_samples)

        # Save Variables
        self.loadings_ = v_transpose.T
        self.explained_variance_ = explained_variance * 100
        self.explained_variance_accumulative = np.cumsum(
            self.explained_variance_)
        self.q_residuals_ = q_residuals
        self.q_limit_ = q_limit
        self.t_hotelling_ = t_hotelling
        self.t_limit_ = t_limit
        self.__covariance_matrix = np.diag(singular_values**2)
        return self

    def transform(self, X, y=None):
        """ Project the input data onto the principal components.

        Parameters
        ----------
        X : array-like or pd.DataFrame, shape (n_samples, n_features)
            Input data.
        y : None, it is only present for compatibility with sklearn

        Returns
        -------
        scores : array, shape (n_samples, n_components)
            Projected data onto the principal components.
        """
        X = check_array(X, accept_sparse=False, ensure_min_samples=5)
        check_consistent_length(X)
        check_is_fitted(self, attributes=["loadings_"])

        if self.loadings_ is None:
            raise ValueError(
                "The model has not been fitted yet. Please call fit() first.")

        if self.mean_center:
            X = X - self.mean_
        # Project the new sample onto the principal components
        if self.loadings_ is None:
            raise ValueError(
                "The model has not been fitted yet. Please call fit() first.")
        scores = np.dot(X, self.loadings_)
        return scores

    def predict(self, X, y=None):
        """
        Predict Q residuals and Hotelling's T-squared for new samples.

        Parameters:
        - X: array-like or pd.DataFrame, shape (n_samples, n_features)
          New samples.

        Returns:
        - scores: array, shape (n_samples, n_components)
          Projected data onto the principal components.
        """
        X = check_array(X, accept_sparse=False, ensure_min_samples=5)
        check_consistent_length(X)
        if self.loadings_ is None:
            raise ValueError(
                "The model has not been fitted yet. Please call fit() first.")

        check_is_fitted(self, attributes=["loadings_"])
        if self.mean_center:
            X = X - self.mean_

        # Project the new sample onto the principal components
        if self.loadings_ is None:
            raise ValueError(
                "The model has not been fitted yet. Please call fit() first.")

        # Project the new sample onto the principal components
        scores = np.dot(X, self.loadings_)

        # Calculate the Q residuals for the new sample
        q_residuals = self.__calculate_q_residuals(
            X, scores, self.loadings_)

        # Calculate Hotelling's T-squared for the new sample
        t_hotelling = self.__calculate_hotelling_t2(
            scores, self.__covariance_matrix)  # type: ignore

        # Save Variables
        self.q_residuals_predicted_ = q_residuals
        self.t_hotelling_predicted_ = t_hotelling
        return scores

    def __calculate_explained_variance(self, singular_values: Union[np.ndarray, List[float]], n_samples: int) -> Union[np.ndarray, List[float]]:
        """Calculate the explained variance ratio.

        Parameters
        ----------
        singular_values : array-like, shape (n_components,)
            Singular values corresponding to the principal components.
        n_samples : int
            Number of samples in the dataset.

        Returns
        -------
        array-like, shape (n_components,)
            Explained variance ratio for each principal component.
        """
        eig_val = np.square(singular_values) / (n_samples - 1)
        explained_variance = eig_val / eig_val.sum()
        return explained_variance

    def __calculate_q_residuals(self, X: np.ndarray, scores: np.ndarray,
                                loadings: np.ndarray) -> np.ndarray:
        """
        Calculate the Q residuals.

        Parameters
        ----------
        X : numpy.ndarray
            The input data matrix.
        scores : numpy.ndarray
            The scores obtained from PCA.
        loadings : numpy.ndarray
            The loadings obtained from PCA.

        Returns
        -------
        numpy.ndarray
            The Q residuals.
        """
        q = X - np.dot(scores, loadings.T)
        q_residuals = np.sum(q**2, axis=1)
        return q_residuals

    def __calculate_q_limit(self, singular_values: Union[np.ndarray, List[float]],
                            n_samples: int) -> float:
        """
        Calculate the Q limit for PCA decomposition.

        Parameters
        ----------
        singular_values : array-like
            Singular values obtained from PCA decomposition.
        n_samples : int
            Number of samples in the dataset.

        Returns
        -------
        float
            The Q limit value.

        """
        eigen_values = np.square(singular_values) * (n_samples - 1)
        sum_eigen_values = sum(eigen_values)
        sum_squared_eigen_values = sum(eigen_values**2)
        sum_cubed_eigen_values = sum(eigen_values**3)
        hoeffding = 1 - (2 * sum_eigen_values * sum_cubed_eigen_values) / \
            (3 * sum_squared_eigen_values**2)
        inverse_normal_distribution = ndtri(self.confidence_level)
        term1 = (hoeffding * inverse_normal_distribution *
                 (2 * sum_squared_eigen_values) ** (0.5)) / sum_eigen_values
        term2 = (sum_squared_eigen_values * hoeffding *
                 (hoeffding - 1)) / (sum_eigen_values**2)

        q_limit = sum_eigen_values * (term1 + 1 + term2) ** (1 / hoeffding)
        return q_limit

    def __calculate_hotelling_t2(self, scores_matrix: np.ndarray,
                                 covariance_matrix: Union[np.ndarray, List[float]]):
        """
        Calculate the Hotelling's T^2 statistic.

        Parameters
        ----------
        scores_matrix : numpy.ndarray
            The matrix of scores obtained from PCA.
        covariance_matrix : numpy.ndarray
            The covariance matrix.

        Returns
        -------
        numpy.ndarray
            The Hotelling's T^2 statistic for each sample.
        """
        intermediate_product = np.dot(scores_matrix, covariance_matrix)
        hotelling_t2 = np.dot(intermediate_product, scores_matrix.T)
        hotelling_t2 = np.diagonal(hotelling_t2)
        return hotelling_t2

    def __calculate_hotelling_t2_limit(self, n_samples: int):
        """Calculate the Hotelling's T^2 limit.

        Parameters
        ----------
        n_samples : int
            The number of samples.

        Returns
        -------
        float
            The Hotelling's T^2 limit.
        """
        f_value = f.ppf(self.confidence_level, self.n_components,
                        n_samples - self.n_components)
        t2_limit = (self.n_components * (n_samples - 1) /
                    (n_samples - self.n_components) * f_value)
        return t2_limit
