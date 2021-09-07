#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 00:24:09 2021

@author: luca
"""

import requests

url = "https://app-fastapi-cicd.herokuapp.com/predict"

headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
}

data = '{ "workclass": "State-gov", "education": "Bachelors", "marital-status": "Never-married", "occupation": "Adm-clerical", "relationship": "Not-in-family", "race": "White", "sex": "Male", "native-country": "United-States" }'

response = requests.post(url, headers=headers, data=data)

status_code = response.status_code
data = response.json()

print(status_code)
print(data)
