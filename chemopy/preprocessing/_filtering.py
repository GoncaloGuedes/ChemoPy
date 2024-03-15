from typing import List, Tuple

import numpy as np
from scipy.signal import detrend, savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_scalar


class ConvolutionSmoothing(BaseEstimator, TransformerMixin):
    """A custom Scikit-learn estimator to perform convolution smoothing on 1D data.

    Parameters:
    -----------
    kernel_size : int
        The size of the convolution kernel used for smoothing.

    keep_dims: bool
        If True the output signal will have the same dimensions as X.
        If False (default) the output signal will have max(M, N) - min(M, N) + 1

    Attributes:
    -----------
    kernel_size : int
        The size of the convolution kernel used for smoothing.

    Methods:
    --------
    fit(X, y=None):
        Fit the estimator to the data. Does nothing in this case.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        y : array-like or None (default: None)
            The target values (ignored).

        Returns:
        --------
        self : object
            Returns the instance itself.

    transform(X):
        Apply convolution smoothing on each sample in the input data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data to be smoothed.

        Returns:
        --------
        smoothed_data : array-like, shape (n_samples, n_features - kernel_size + 1)
            The smoothed data after applying the convolution operation.
    """

    def __init__(self, kernel_size: int, keep_dims: bool = False):
        self.kernel_size = check_scalar(
            kernel_size,
            name="kernel_size",
            min_val=2,
            target_type=int,
        )
        self.keep_dims = keep_dims

    def _smooth_data(self, x, kernel_size=3, keep_dims: bool = False):
        kernel = np.ones(kernel_size) / kernel_size
        if not keep_dims:
            return np.convolve(x, kernel, mode="valid")
        else:
            x_convoluted = np.convolve(x, kernel, mode="same")
        # Concatenate the first and last windows with the original data
        x_convoluted[:kernel_size] = x[:kernel_size]
        x_convoluted[-kernel_size:] = x[-kernel_size:]
        return x_convoluted

    def fit(self, X, y=None):
        """Fit the estimator to the data. Does nothing in this case.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        y : None
            The target values (ignored).

        Returns
        -------
        _type_
            _description_
        """
        X = check_array(X, accept_sparse=False, ensure_2d=True)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """Apply convolution smoothing on each sample in the input data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data to be smoothed.

        Returns:
        --------
        smoothed_data : array-like, shape (n_samples, n_features - kernel_size + 1)
            The smoothed data after applying the convolution operation.
        """
        X = check_array(X)
        check_is_fitted(self)
        return np.apply_along_axis(
            self._smooth_data,
            axis=1,
            arr=X,
            **{"kernel_size": self.kernel_size, "keep_dims": self.keep_dims}
        )


class SavitzkyGolay(BaseEstimator, TransformerMixin):
    """
    Transformer to apply Savitzky-Golay filtering to the input data.

    Parameters
    ----------
    window_length : int, optional (default=5)
        The length of the window used for filtering.

    polyorder : int, optional (default=2)
        The order of the polynomial used for fitting the samples in the window.

    deriv : int, optional (default=1)
        The order of the derivative to compute. Zero corresponds to smoothing.

    Methods
    -------
    fit(X, y=None)
        Do nothing and return self.

    transform(X, y=None)
        Apply Savitzky-Golay filtering to the input data X.
    """

    def __init__(self, window_length=5, polyorder=2, deriv=1) -> None:
        super().__init__()
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv

    def fit(self, X, y=None):
        """
        Do nothing and return self.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        y : array-like, shape (n_samples,), optional (default=None)
            The target values.

        Returns
        -------
        self : SavGol
            The fitted transformer.
        """
        return self

    def transform(self, X, y=None):
        """
        Apply Savitzky-Golay filtering to the input data X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        y : array-like, shape (n_samples,), optional (default=None)
            The target values.

        Returns
        -------
        X_filtered : array-like, shape (n_samples, n_features)
            The filtered input data.
        """
        return savgol_filter(
            X,
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.deriv,
        )


class SelectIntervals(BaseEstimator, TransformerMixin):
    """
    Select specific intervals from a 2D numpy array and concatenate them along the columns to create a final array.
    Parameters:
    intervals (List[Tuple[int, int]]): List of tuples containing the start and end index of the intervals to select.

    Returns:
    numpy.ndarray: The final array with the selected intervals.
    """

    def __init__(self, intervals: List[Tuple[int, int]]) -> None:
        super().__init__()
        self.intervals = intervals

    def fit(self, X, y=None):
        """
        This method does not perform any operation and just returns the fitted estimator.
        """
        return self

    def transform(self, X, y=None):
        """
        Selects specific intervals from a 2D numpy array and concatenates them along the columns to create a final array.

        Parameters:
            X (numpy.ndarray): The input 2D numpy array to select intervals from.
            y: Ignored.

        Returns:
        numpy.ndarray: The final array with the selected intervals.
        """
        X = np.array(X)
        if self.intervals is None:
            return X
        selected_arrays = []
        for interval in self.intervals:
            # select the specific interval and append it to the list
            selected_arrays.append(X[:, interval[0] : interval[1]])
        # concatenate the selected arrays along the columns to create final array
        final_array = np.hstack(selected_arrays)
        return final_array


class Detrend(BaseEstimator, TransformerMixin):
    """
    Transformer to remove the linear trend from the input data.
    
    Methods
    -------
    fit(X, y=None)
        Do nothing and return self.
    
    transform(X, y=None)
        Remove the linear trend from the input data X.
        
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return detrend(X)
