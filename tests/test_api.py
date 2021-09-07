#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 22:25:28 2021

@author: luca
"""

from fastapi.testclient import TestClient

from api.main import app


def test_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json == {'greeting': 'Hello World!'}


def test_prediction_good_payload():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "workclass": "State-gov",
                "education": "Bachelors",
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "native-country": "United-States",
            },
        )

        assert response.status_code == 200
        assert response.json() == {"prediction": {"salary": "<=50K"}}


def test_prediction_bad_payload():
    with TestClient(app) as client:

        response = client.post(
            "/predict",
            json={
                "workclass": "State-gov",
                "education": "Bachelors",
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "native-country": "lol",
            },
        )

        assert response.status_code == 422
        assert response.json() == {'detail': [{'loc': ['body', 'workclass'],
           'msg': "value is not a valid enumeration member; permitted: 'State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'",
           'type': 'type_error.enum',
           'ctx': {'enum_values': ['State-gov',
             'Self-emp-not-inc',
             'Private',
             'Federal-gov',
             'Local-gov',
             '?',
             'Self-emp-inc',
             'Without-pay',
             'Never-worked']}}]}
