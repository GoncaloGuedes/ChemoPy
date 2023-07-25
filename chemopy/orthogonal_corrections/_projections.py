from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class OrthogonalProjection(BaseEstimator, TransformerMixin):
    """
    Orthogonal Projection Transformer.

    This class implements a transformer that performs orthogonal projection on input data.

    Parameters:
    -----------
    P : array-like, shape (n_features, n_features)
        The orthogonal matrix used for projection. It should be symmetric and orthogonal,
        i.e., P @ P.T = P.T @ P = I, where I is the identity matrix.

    Methods:
    --------
    fit(X)
        Fits the transformer to the data. This method does nothing and only exists to comply
        with the scikit-learn TransformerMixin interface.

    transform(X)
        Applies the orthogonal projection to the input data X.

    Attributes:
    -----------
    P : array-like, shape (n_features, n_features)
        The orthogonal matrix used for projection.
    """

    def __init__(self, P) -> None:
        """
        Initialize the OrthogonalProjection transformer.

        Parameters:
        -----------
        P : array-like, shape (n_features, n_features)
            The orthogonal matrix used for projection.
        """
        super().__init__()
        self.P = P 

    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        This method does nothing and only exists to comply with the scikit-learn TransformerMixin interface.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data (not used in this transformer).

        Returns:
        --------
        self : OrthogonalProjection
            The fitted transformer instance.
        """
        return self
    
    def transform(self, X, y=None):
        """
        Apply orthogonal projection to the input data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data to be transformed.

        y : array-like, shape (n_samples,), optional (default=None)
            Target data (not used in this transformer).

        Returns:
        --------
        X_proj : array-like, shape (n_samples, n_features)
            Transformed data after orthogonal projection.
        """
        I = np.eye(self.P.shape[0])
        X_proj = X - np.dot(np.dot(X, self.P), self.P.T)
        return X_proj

