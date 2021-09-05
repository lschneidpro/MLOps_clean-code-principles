#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 20:59:26 2021

@author: luca
"""
from joblib import load
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.ml.data import process_data
from src.ml.model import compute_model_metrics, inference


@pytest.fixture
def data():
    df = pd.read_csv('../starter/data/census_clean.csv')
    return df

@pytest.fixture
def model():
    model = load('../starter/model/model.joblib')

    return model

@pytest.fixture
def encoder():
    encoder = load('../starter/model/encoder.joblib')
    return encoder

@pytest.fixture
def lb():
    lb = load('../starter/model/lb.joblib')
    return lb

def test_data_shape(data):
    assert data.shape == data.dropna().shape, "Dropping null changes shape."
    
def test_process_data(data, encoder, lb):
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
    
    X, y, _, _ = process_data(
        data, categorical_features=cat_features, label="salary", training=False,
        encoder = encoder, lb=lb
    )
    
    assert data.shape[0]==np.shape(X)[0], "The number of samples changed"
    assert data.shape[0]==np.shape(y)[0], "The number of samples changed"
    assert data.shape[1]<=np.shape(X)[1], "Issue with Categorical Encoding"
    assert np.ndim(y)==1, "Issue with Labels"
    
def test_minimal_performance(data, model, encoder, lb):
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
    
    train, test = train_test_split(data, test_size=0.20, random_state=42)
    
    X, y, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False,
        encoder = encoder, lb=lb
    )
    
    preds = inference(model, X)
    precision, recall, fbeta = compute_model_metrics(y, preds)
    
    assert fbeta>=0.5, "Model is not performant enough"
