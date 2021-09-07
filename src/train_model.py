# Script to train machine learning model.

import json
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump


from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference


def compute_model_metrics_slice(
    model, data, encoder, lb, cat_features, sliced_feature, label
):

    dict_result = {}

    for i in data[sliced_feature].unique():
        data_slice = data[data[sliced_feature] == i]

        X, y, _, _ = process_data(
            data_slice,
            categorical_features=cat_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb,
        )

        preds = inference(model, X)
        precision, recall, fbeta = compute_model_metrics(y, preds)

        dict_result[i] = {
            "precision": precision,
            "recall": recall,
            "fbeta": fbeta,
            "sample": len(y),
        }

    return {sliced_feature: dict_result}


if __name__ == "main":
    # Load in the data.
    data = pd.read_csv("../data/census_clean.csv")

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20, random_state=42)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    # Train and save a model.
    model = train_model(X_train, y_train, hp=True)

    dump(model, "../model/model.joblib")
    dump(encoder, "../model/encoder.joblib")
    dump(lb, "../model/lb.joblib")

    # Performance overall
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    # Performance slice education
    metrics_education = compute_model_metrics_slice(
        model, test, encoder, lb, cat_features, "education", "salary"
    )
    with open("../screenshots/slice_output.json", "w") as fp:
        json.dump(metrics_education, fp)
