import time

from sklearn.metrics import (
    fbeta_score,
    precision_score,
    recall_score,
    f1_score,
    make_scorer,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold


SEED = 42

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train, hp=False):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier(n_jobs=-1, random_state=SEED)

    if hp:
        cv = StratifiedKFold(n_splits=5, shuffle=False)
        parameters = {
            "max_depth": [3, 5, 10, 30, 60, 100, None],
            "n_estimators": [100, 200, 300, 400, 500],
            "max_features": ["auto", "sqrt"],
            "criterion": ["gini", "entropy"],
            "bootstrap": [True, False],
            "min_samples_leaf": [1, 2, 4],
            "min_samples_split": [2, 5, 10],
        }
        optimal_model = RandomizedSearchCV(
            model,
            param_distributions=parameters,
            n_iter=50,
            cv=cv,
            n_jobs=-1,
            verbose=2,
            scoring=make_scorer(f1_score),
            random_state=SEED,
        )

        start_time = time.time()

        optimal_model.fit(X_train, y_train)

        stop_time = time.time()

        print(
            "Elapsed Time:",
            time.strftime("%H:%M:%S", time.gmtime(stop_time - start_time)),
        )
        print("Best Score: {:.3f}".format(optimal_model.best_score_))
        print("Best Parameters: {}".format(optimal_model.best_params_))

        return optimal_model
    else:

        parameters = {
            "n_estimators": 300,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "max_depth": 100,
            "criterion": "gini",
            "bootstrap": False,
        }
        model.set_params(**parameters)
        model.fit(X_train, y_train)

        return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds
