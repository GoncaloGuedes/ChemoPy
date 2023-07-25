from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import f
from scipy.special import ndtri
import numpy as np


class PCA(BaseEstimator, TransformerMixin):
    """
    Principal Component Analysis (PCA) Transformer.

    This class implements a transformer that performs Principal Component Analysis (PCA) on input data.

    Parameters:
    -----------
    n_components : int, optional (default=2)
        The number of principal components to keep.

    mean_center : bool, optional (default=False)
        Whether to mean center the input data before performing PCA.

    confidence_level : float, optional (default=0.95)
        The confidence level to calculate the control limits for Q residuals and Hotelling's T-squared.

    Methods:
    --------
    fit(X)
        Fits the transformer to the data, performing PCA and calculating control limits for Q residuals
        and Hotelling's T-squared.

    transform(X, y=None)
        Applies PCA to the input data X, projecting it onto the principal components.

    Attributes:
    -----------
    loadings_ : array-like, shape (n_components, n_features)
        The loadings of the principal components.

    explained_variance_ : array-like, shape (n_components,)
        The explained variance of each principal component, expressed as a percentage.

    explained_variance_accumulative : array-like, shape (n_components,)
        The accumulative explained variance of the principal components, expressed as a percentage.

    q_residuals_ : array-like, shape (n_samples,)
        The Q residuals of the fitted data.

    q_limit_ : float
        The control limit for Q residuals.

    t_hotelling_ : array-like, shape (n_samples,)
        The Hotelling's T-squared statistics of the fitted data.

    t_limit_ : float
        The control limit for Hotelling's T-squared.

    _sigma : array-like, shape (n_components, n_components)
        The inverse covariance matrix used to calculate Hotelling's T-squared.

    q_residuals_predicted_ : array-like, shape (n_samples,)
        The Q residuals of the transformed data (new samples).

    t_hotelling_predicted_ : array-like, shape (n_samples,)
        The Hotelling's T-squared statistics of the transformed data (new samples).
    """

    def __init__(self, n_components=2, mean_center=False, confidence_level=0.95) -> None:
        """
        Initialize the PCA transformer.

        Parameters:
        -----------
        n_components : int, optional (default=2)
            The number of principal components to keep.

        mean_center : bool, optional (default=False)
            Whether to mean center the input data before performing PCA.

        confidence_level : float, optional (default=0.95)
            The confidence level to calculate the control limits for Q residuals and Hotelling's T-squared.
        """
        super().__init__()
        self.n_components = n_components
        self.mean_center = mean_center
        self.confidence_level = confidence_level
    
    def fit(self, X):
        """
        Fit the transformer to the data, performing PCA and calculating control limits for Q residuals
        and Hotelling's T-squared.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data to be fitted.

        Returns:
        --------
        self : PCA
            The fitted transformer instance.
        """
        # Mean center
        if self.mean_center:
            self.mean_ = np.mean(X, axis=0)
            X -= self.mean_
        
        # number of samples
        n_samples = X.shape[0]
        
        # calculate the SVD of the centered data
        u, s, vt = np.linalg.svd(X)
        
        # keep only the top n_components
        u = u[:, :self.n_components]
        s_all_components = s
        s = s[:self.n_components]
        vt = vt[:self.n_components]

        # calculate the loadings
        loadings = vt

        # project the data onto the principal components
        scores = np.dot(X, loadings.T)

        # calculate the explained variance
        eig_val = s**2 / (n_samples-1)
        explained_variance = eig_val / eig_val.sum()

        # calculate the Q residuals
        q = X - np.dot(scores, loadings)
        q_residuals = np.sum(q**2, axis=1)
        
        # calculate the Q limit
        eig_val_all_components = s_all_components[self.n_components:]**2 / (n_samples-1)
        t1 = sum(eig_val_all_components)
        t2 = sum(eig_val_all_components ** 2)
        t3 = sum(eig_val_all_components ** 3)
        ho = 1 - (2 * t1 * t3) / (3 * t2 ** 2)
        ca = ndtri(self.confidence_level)
        term1 = (ho * ca * (2 * t2) ** (0.5)) / t1
        term2 = (t2 * ho * (ho - 1)) / (t1 ** 2)
        q_limit = t1 * (term1 + 1 + term2) ** (1 / ho)

        # calculate the Hotelling T^2 statistic
        # T^2 = scores.T * sigma *scores
        sigma = np.linalg.inv(np.diag(eig_val)) 
        aux = np.dot(scores, sigma)
        t_hotelling = np.dot(aux, scores.T)
        t_hotelling = np.diagonal(t_hotelling)

        # # calculate the T^2 limit
        f_value = f.ppf(self.confidence_level, self.n_components, n_samples - self.n_components)
        t2_limit = self.n_components * (n_samples - 1) / (n_samples - self.n_components) * f_value
        
        # Save Variables
        self.loadings_ = loadings
        self.explained_variance_ = explained_variance * 100
        self.explained_variance_accumulative = np.cumsum(self.explained_variance_)
        self.q_residuals_ = q_residuals
        self.q_limit_ = q_limit
        self.t_hotelling_ = t_hotelling
        self.t_limit_ = t2_limit
        self._sigma = sigma
        return self
    
    def transform(self, X, y=None):
        """
        Apply PCA to the input data, projecting it onto the principal components.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data to be transformed.

        y : array-like, shape (n_samples,), optional (default=None)
            Target data (not used in this transformer).

        Returns:
        --------
        scores : array-like, shape (n_samples, n_components)
            Transformed data projected onto the principal components.
        """
        # Mean center
        if self.mean_center:
            X -= self.mean_
        
        # Project the new sample onto the principal components
        scores = np.dot(X, self.loadings_.T)
        
        # Calculate the Q residuals for the new sample
        q = X - np.dot(scores, self.loadings_)
        q_residuals = np.sum(q ** 2, axis=1)
        
        # Calculate Hotelling's T-squared for the new sample
        aux = np.dot(scores, self._sigma)
        t_hotelling = np.dot(aux, scores.T)
        t_hotelling = np.diagonal(t_hotelling)
        
        # # Save Variables
        self.q_residuals_predicted_ = q_residuals
        self.t_hotelling_predicted_ = t_hotelling
        return scores
