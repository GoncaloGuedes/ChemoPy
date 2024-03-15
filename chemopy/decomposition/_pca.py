""" Principal Component Analysis (PCA) Transformer. """

from typing import List, Union

import numpy as np
import numpy.typing as npt
from scipy.stats import norm
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

    def __init__(
        self,
        n_components: int = 2,
        mean_center: bool = False,
        confidence_level: float = 0.95,
    ):
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
        """Fit the PCA model to the input data.

        Parameters
        ----------
        X : array-like or pd.DataFrame, shape (n_samples, n_features)
            Input data.
        y : array-like, shape (n_samples,), optional
            To be compatible with sklearn's fit method.

        Returns
        -------
        self : PCA
            Fitted PCA model.
        """
        X = check_array(
            X,
            accept_sparse=False,
            ensure_min_samples=5,
            ensure_min_features=self.n_components,
        )

        if self.mean_center:
            self.mean_ = np.mean(X, axis=0)
            X = X - self.mean_

        # number of samples
        n_samples = X.shape[0]

        # calculate the SVD of the data
        singular_vectors, singular_values, v_transpose = np.linalg.svd(X)

        # keep only the top n_components
        singular_vectors = singular_vectors[:, : self.n_components]
        singular_values = singular_values[: self.n_components]
        v_transpose = v_transpose[: self.n_components]

        # project the data onto the principal components
        scores = np.dot(X, v_transpose.T)

        # calculate the explained variance
        explained_variance = self.__explained_variance(singular_values, n_samples)

        # calculate q residuals
        q_residuals = self.__q_residuals(X, scores, v_transpose.T)

        # calculate q limit
        q_limit = self.__confidence_interval(q_residuals)

        # calculate hotelling t^2
        t_hotelling = self.__hotelling_t2(scores, np.diag(singular_values**2))

        # calculate t^2 limit
        t_limit = self.__confidence_interval(t_hotelling)

        # Save Variables
        self.loadings_ = v_transpose.T
        self.explained_variance_ = explained_variance * 100
        self.explained_variance_accumulative = np.cumsum(self.explained_variance_)
        self.q_residuals_ = q_residuals
        self.q_limit_ = q_limit
        self.t_hotelling_ = t_hotelling
        self.t_limit_ = t_limit
        self.__covariance_matrix = np.diag(singular_values**2)
        return self

    def transform(self, X, y=None):
        """Project the input data onto the principal components.

        Parameters
        ----------
        X : array-like or pd.DataFrame, shape (n_samples, n_features)
            Input data.
        y : optional
            to be compatible with sklearn's transform method.

        Returns
        -------
        scores : array, shape (n_samples, n_components)
            Projected data onto the principal components.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        ValueError
            If the input data has less than 5 samples.
        """
        X = check_array(X, accept_sparse=False, ensure_min_samples=5)
        check_consistent_length(X)
        check_is_fitted(self, attributes=["loadings_"])

        if self.loadings_ is None:
            raise ValueError(
                "The model has not been fitted yet. Please call fit() first."
            )

        if self.mean_center:
            X = X - self.mean_
        # Project the new sample onto the principal components
        if self.loadings_ is None:
            raise ValueError(
                "The model has not been fitted yet. Please call fit() first."
            )
        scores = np.dot(X, self.loadings_)
        return scores

    def predict(self, X, y=None):
        """Predict the Q residuals and Hotelling's T-squared for new samples.

        Parameters
        ----------
        X : array-like or pd.DataFrame, shape (n_samples, n_features)
            Input data.
        y : None, optional
            to be compatible with sklearn's predict method.

        Returns
        -------
        scores : array, shape (n_samples, n_components)
            Projected data onto the principal components.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        X = check_array(X, accept_sparse=False)
        check_consistent_length(X)
        if self.loadings_ is None:
            raise ValueError(
                "The model has not been fitted yet. Please call fit() first."
            )

        check_is_fitted(self, attributes=["loadings_"])
        if self.mean_center:
            X = X - self.mean_

        # Project the new sample onto the principal components
        if self.loadings_ is None:
            raise ValueError(
                "The model has not been fitted yet. Please call fit() first."
            )

        # Project the new sample onto the principal components
        scores = np.dot(X, self.loadings_)

        # Calculate the Q residuals for the new sample
        q_residuals = self.__q_residuals(X, scores, self.loadings_)

        # Calculate Hotelling's T-squared for the new sample
        t_hotelling = self.__hotelling_t2(
            scores, self.__covariance_matrix
        )  # type: ignore

        # Save Variables
        self.q_residuals_predicted_ = q_residuals
        self.t_hotelling_predicted_ = t_hotelling
        return scores

    def __explained_variance(
        self,
        singular_values: Union[npt.NDArray[np.float64], List[float]],
        n_samples: int,
    ) -> npt.NDArray[np.float64]:
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

    def __q_residuals(
        self,
        X: npt.NDArray[np.float64],
        scores: npt.NDArray[np.float64],
        loadings: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
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

        Notes
        -----
        The Q residuals are a measure of the difference between the original data matrix X
        and the reconstructed data matrix obtained by multiplying the scores and loadings matrices.

        The Q residuals are calculated as the sum of squared differences between each row of X
        and its corresponding reconstructed row.

        The Q residuals can be used to assess the quality of the PCA model and identify outliers
        in the data.

        """
        q = X - np.dot(scores, loadings.T)
        q_residuals = np.sum(q**2, axis=1)
        return q_residuals

    def __hotelling_t2(
        self,
        scores_matrix: npt.NDArray[np.float64],
        covariance_matrix: Union[npt.NDArray[np.float64], List[float]],
    ) -> npt.NDArray[np.float64]:
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

    def __confidence_interval(
        self, population: Union[npt.NDArray[np.float64], List[float]]
    ) -> float:
        """Calculate the confidence interval for the population.

        Parameters
        ----------
        population : Union[np.ndarray, List[float]]
            The population. It can be either Q residuals or Hotelling's T^2.


        Returns
        -------
        float
            The confidence interval.
        """
        # Assuming population follow a normal distribution
        mu, std = norm.fit(population)

        # Calculate the confidence interval
        alpha = 1 - self.confidence_level
        interval = norm.ppf(1 - alpha / 2, loc=mu, scale=std)
        return interval  # type: ignore
