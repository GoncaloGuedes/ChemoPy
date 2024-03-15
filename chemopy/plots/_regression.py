import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict


def plot_cv_regression(
    estimator: BaseEstimator, X_train: np.ndarray, y_train: np.ndarray, title: str = ""
) -> None:
    """Plot the cross-validated predictions against the actual values and the calibration predictions.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator that was fitted using GridSearchCV or RandomizedSearchCV
    X_train : np.ndarray
        The training set
    y_train : np.ndarray
        The training set labels
    title : str, optional
        title of the plot, by default None
    """
    # Predict on the test set using the best estimator found by GridSearchCV
    y_pred = estimator.predict(X_train)  # type:ignore

    # Compute R2 and RMSECV on the test set
    r2 = r2_score(y_train, y_pred)
    rmse_cal = np.sqrt(mean_squared_error(y_train, y_pred))

    # Compute CV predictions on the training set
    cv_pred = cross_val_predict(
        estimator.best_estimator_, # type:ignore
        X_train,
        y_train,
        cv=estimator.n_splits_,  # type:ignore
    )

    # Compute R2 and RMSECV on the CV predictions
    r2cv = r2_score(y_train, cv_pred)
    rmse_cv = -estimator.best_score_  # type:ignore

    # Plot CV vs. actual values
    plt.scatter(y_train, cv_pred, alpha=0.5)
    plt.scatter(y_train, y_pred, alpha=0.5)
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], "k--", lw=2)
    plt.xlabel("Actual Values")
    plt.ylabel("CV Predictions")
    plt.title(f"{title} CV vs. Actual Values")
    plt.legend(["CV predictions", "Calibration predictions", "Perfect prediction"])

    # Format R2 and RMSECV with three decimal points
    r2_str = f"{r2:.3f}"
    rmse_cal_str = f"{rmse_cal:.3f}"
    r2cv_str = f"{r2cv:.3f}"
    rmse_cv_str = f"{rmse_cv:.3f}"

    # Add text annotations for R2 and RMSECV in the right lower corner
    text = f"R2 Cal: {r2_str}\nRMSE Cal: {rmse_cal_str}\nR2  CV: {r2cv_str}\nRMSE-CV : {rmse_cv_str}"
    plt.text(max(y_train), min(y_train), text, va="bottom", ha="right")

    plt.show()


def plot_predictions_regression(
    estimator: BaseEstimator, X: np.ndarray, y: np.ndarray, title: str = ""
):
    """Plot the predictions against the actual values.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator that was fitted using GridSearchCV or RandomizedSearchCV
    X : np.ndarray
        The test set
    y : np.ndarray
        The test set labels
    title : str, optional
        title of the plot, by default None
    """
    # Predict on the test set using the best estimator found by GridSearchCV
    y_pred = estimator.predict(X)  # type:ignore

    # Compute R2 and RMSECV on the test set
    r2_pred = r2_score(y, y_pred)
    rmse_pred = np.sqrt(mean_squared_error(y, y_pred))

    # Compute CV predictions on the training set
    rmse_cv = -estimator.best_score_  # type:ignore

    # Plot CV vs. actual values
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([min(y), max(y)], [min(y), max(y)], "k--", lw=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predictions")
    plt.title(f"{title} Predicted vs. Actual Values")
    plt.legend(["Predictions", "Perfect prediction"])

    # Format R2 and RMSECV with three decimal points
    rmse_pred_str = f"{rmse_pred:.3f}"
    r2_pred_str = f"{r2_pred:.3f}"
    rmse_cv_str = f"{rmse_cv:.3f}"

    # Add text annotations for R2 and RMSECV in the right lower corner
    text = (
        f"R2 Pred: {r2_pred_str}\nRMSE Pred: {rmse_pred_str}\nRMSE-CV : {rmse_cv_str}"
    )
    plt.text(max(y), min(y), text, va="bottom", ha="right")
    plt.show()
