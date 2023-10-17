import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class SNV(BaseEstimator, TransformerMixin):
    """
    Standard Normal Variate (SNV) method performs a normalization of the spectra that consists
    in subtracting each spectrum by its own mean and dividing it by its own standard deviation.
    After SNV, each spectrum will have a mean of 0 and a standard deviation of 1.

    Parameters:
    None

    Attributes:
    None
    """
    def __init__(self, trainable=True) -> None:
        super().__init__()
        self.trainable = trainable

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        Fit the SNV transformer to the training data.

        Parameters:
        X (np.ndarray): Input data of shape (n_samples, n_features).
        y (np.ndarray): Target values (unused).

        Returns:
        self (SNV): The fitted SNV transformer object.
        """
        if self.trainable is False:
            return X
        X = check_array(
            X,
            ensure_2d=True,
        )
        self.X_ = X
        self.y_ = y
        self.n_features_in_ = np.shape(X)[1]
        return self

    def transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        Apply the Standard Normal Variate (SNV) transformation to the input data.

        Parameters:
        X (np.ndarray): Input data of shape (n_samples, n_features).
        y (np.ndarray): Target values (unused).

        Returns:
        X_snv (np.ndarray): Transformed data of shape (n_samples, n_features).
        """
        check_is_fitted(self)
        if self.trainable is False:
            return X

        X = check_array(
            X,
            ensure_2d=True,
        )

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in transform is different from the number of features in fit."
            )

        x_mean = np.mean(X, keepdims=True, axis=1)
        x_std = np.std(X, keepdims=True, axis=1)
        x_snv = (X - x_mean) / x_std
        return x_snv
