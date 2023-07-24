import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from typing import Union


class TransmittanceToAbsorbance(BaseEstimator, TransformerMixin):
    """
    Transformer to convert transmittance values to absorbance values using the formula: 
    absorbance = log10(1 / transmittance)

    Parameters:
    percentage: bool = True is you have the %T

    Attributes:
    None
    """
    
    def __init__(self, percentage:bool=False) -> None:
        super().__init__()
        self.percentage = percentage

    def fit(self, X: Union[np.ndarray, list], y=None):
        """
        Fit the transformer.

        Parameters:
        X (array-like of shape (n_samples, n_features)): Input transmittance values.

        Returns:
        self (object): The fitted transformer object.
        """
        X = check_array(X,
                        ensure_2d=True,
                        force_all_finite=True, 
                        accept_sparse=True)
        
        self.n_features_in_ = np.shape(X)[1]
        self.X_ = X
        self.y_ = y
        return self
    
    def transform(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Transform the transmittance values to absorbance values.

        Parameters:
        X (array-like of shape (n_samples, n_features)): Input transmittance values.

        Returns:
        transformed_X (ndarray of shape (n_samples, n_features)): Transformed absorbance values.
        """
        
        # Check if fit has been called
        check_is_fitted(self)
        
        X = check_array(X,
                        ensure_2d=True,
                        force_all_finite=True, 
                        accept_sparse=True,
                        )
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError("The number of features in transform is different from the number of features in fit.")
        
        if self.percentage:
            return 2- np.log10(X)
        return np.log10(1 / X)



class AbsoluteValues(BaseEstimator, TransformerMixin):
    """
    Transformer to convert all negative values in positive

    Parameters:
    None

    Attributes:
    None
    """
    def fit(self, X: Union[np.ndarray, list], y=None):
        """
        Fit the transformer.

        Parameters:
        X (array-like of shape (n_samples, n_features)): Input transmittance values.

        Returns:
        self (object): The fitted transformer object.
        """
        X = check_array(X,
                        ensure_2d=True,
                        force_all_finite=True, 
                        accept_sparse=True)
        
        self.n_features_in_ = np.shape(X)[1]
        self.X_ = X
        self.y_ = y
        return self
    
    def transform(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Transform the negative values into positive.

        Parameters:
        X (array-like of shape (n_samples, n_features)): Input values.

        Returns:
        transformed_X (ndarray of shape (n_samples, n_features)): Transformed values.
        """
        
        # Check if fit has been called
        check_is_fitted(self)
        
        X = check_array(X,
                        ensure_2d=True,
                        force_all_finite=True, 
                        accept_sparse=True,
                        )
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError("The number of features in transform is different from the number of features in fit.")
        
        return np.abs(X)