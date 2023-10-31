from typing import Any, List, Union

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


class MSC(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        reference: Union[str, List[Any], np.ndarray] = "mean",
        trainable: bool = True,
    ) -> None:
        """
        Initialize the MSC transformer.

        Parameters:
        reference (Union[str, List[Any], np.ndarray]): Reference spectrum, "Mean," or "Median."
            If reference is a list or array, it should have the same number of features as the input data.
        trainable (bool): If True, the MSC transformer will be fitted to the data. If False, the MSC transformer is not fitted and the transform method will return the input data.
        """
        super().__init__()
        if not (
            reference == "mean"
            or reference == "median"
            or isinstance(reference, (list, np.ndarray))
        ):
            raise ValueError("Reference must be 'Mean,' 'Median,' or a list/array.")
        self.reference = reference
        self.trainable = trainable

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "MSC":
        """
        Fit the MSC transformer to the training data.

        Parameters:
        X (np.ndarray): Input data of shape (n_samples, n_features).
        y (np.ndarray): Target values (unused).

        Returns:
        self (MSC): The fitted MSC transformer object.
        """
        X = check_array(X, ensure_2d=True)
        if self.reference == "mean":
            self.reference_ = np.mean(X, axis=0)
        elif self.reference == "median":
            self.reference_ = np.median(X, axis=0)
        else:
            if X.shape[1] != self.reference.shape[1]:
                raise ValueError(
                    "The number of features in the reference is different from the number of features in X."
                )
            self.reference_ = self.reference

        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        Transform the data using MSC correction.

        Parameters:
        X (np.ndarray): Input data to be transformed of shape (n_samples, n_features).
        y (np.ndarray): Target values (unused).

        Returns:
        X_msc (np.ndarray): The transformed data after MSC correction.
        """
        check_is_fitted(self)
        if self.trainable is False:
            return X
        X = check_array(X, ensure_2d=True)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in transform is different from the number of features in fit."
            )
        poly_coefficients = np.polyfit(self.reference_, X.T, 1)
        a = poly_coefficients[0].reshape(-1, 1)
        b = poly_coefficients[1].reshape(-1, 1)
        X_msc = (X - b) / a
        return X_msc
