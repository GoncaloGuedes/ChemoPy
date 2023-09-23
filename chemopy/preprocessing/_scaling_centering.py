import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class ScaleMaxMin(BaseEstimator, TransformerMixin):
    """
    Transform spectra by scaling each features to a given range.
    """

    def __init__(self, min_value: int = 0, max_value: int = 1) -> np.array:
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

        if min_value > max_value:
            raise ValueError("Min Value cannot be greater than Max Value")

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> np.array:
        """
        Fit the ScaleMaxMin transformer to the training data.

        Parameters:
        X (np.ndarray): Input data of shape (n_samples, n_features).
        y (np.ndarray): Target values (unused).

        Returns:
        self (SNV): The fitted SNV transformer object.
        """
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
        Apply the ScaleMaxMin transformation to the input data.

        Parameters:
        X (np.ndarray): Input data of shape (n_samples, n_features).
        y (np.ndarray): Target values (unused).

        Returns:
        X_snv (np.ndarray): Transformed data of shape (n_samples, n_features).
        """
        check_is_fitted(self)

        X = check_array(
            X,
            ensure_2d=True,
        )

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in transform is different from the number of features in fit."
            )

        X_std = (X - X.min(axis=1, keepdims=True)) / (
            X.max(axis=1, keepdims=True) - X.min(axis=1, keepdims=True)
        )
        X_scaled = X_std * (self.max_value - self.min_value) + self.min_value
        return X_scaled


class Centering(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer to center the data by subtracting the mean or median.

    Parameters
    ----------
    center_method : str, optional (default='mean')
        The method to use for centering. Can be 'mean' or 'median'.

    Attributes
    ----------
    center_method_ : str
        The method used for centering. Either 'mean' or 'median'.

    center_value_ : array-like, shape (n_features,)
        The center (mean or median) of each feature in the input data.

    Methods
    -------
    fit(X, y=None)
        Compute the center (mean or median) of each feature in the input data X.

    transform(X)
        Center the input data X by subtracting the center (mean or median) of each feature.

    Notes
    -----
    The input data X can be a numpy array or a pandas DataFrame.

    Parameters
    ----------
    center_method : str, optional (default='mean')
        The method to use for centering. Can be 'mean' or 'median'.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.metrics import accuracy_score
    >>> from my_transformers import CenterChoice

    >>> iris = load_iris()
    >>> X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
    ...                                                     random_state=0)
    >>> # Using mean centering
    >>> pipe = Pipeline([('center', CenterChoice(center_method='mean')),
    ...                  ('clf', LogisticRegression(random_state=0))])
    >>> pipe.fit(X_train, y_train)
    >>> y_pred = pipe.predict(X_test)
    >>> accuracy_score(y_test, y_pred)
    0.9736842105263158

    >>> # Using median centering
    >>> pipe = Pipeline([('center', CenterChoice(center_method='median')),
    ...                  ('clf', LogisticRegression(random_state=0))])
    >>> pipe.fit(X_train, y_train)
    >>> y_pred = pipe.predict(X_test)
    >>> accuracy_score(y_test, y_pred)
    Some_accuracy_value
    """

    def __init__(self, center_method="mean"):
        """
        Initialize the CenterChoice transformer.

        Parameters
        ----------
        center_method : str, optional (default='mean')
            The method to use for centering. Can be 'mean' or 'median'.
        """
        if center_method not in ["mean", "median"]:
            raise ValueError("Invalid centering method. Use 'mean' or 'median'.")
        self.center_method = center_method

    def fit(self, X, y=None):
        """
        Compute the center (mean or median) of each feature in the input data X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        y : array-like, shape (n_samples,), optional (default=None)
            The target values.

        Returns
        -------
        self : CenterChoice
            The fitted transformer.
        """
        X = check_array(X)
        self.n_features_in_ = X.shape[1]

        if self.center_method == "mean":
            self.center_method_ = "mean"
            self.center_value_ = np.mean(X, axis=0)
        elif self.center_method == "median":
            self.center_method_ = "median"
            self.center_value_ = np.median(X, axis=0)

        return self

    def transform(self, X, y=None):
        """
        Center the input data X by subtracting the center (mean or median) of each feature.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        X_centered : array-like, shape (n_samples, n_features)
            The centered input data.
        """
        check_is_fitted(self)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in transform is different from the number of features in fit."
            )

        return X - self.center_value_
