import pandas as pd
import numpy as np
from ml.data import process_data
from sklearn.model_selection import train_test_split
import os
import pytest


#  testing if any ML functions return the expected type.

@pytest.fixture(scope="module")
def data():
    PTH = "./data/census.csv"
    return pd.read_csv(PTH, skipinitialspace = True)

@pytest.fixture(scope="module")
def features():
    cat_features = [    "workclass",
                        "education",
                        "marital-status",
                        "occupation",
                        "relationship",
                        "race",
                        "sex",
                        "native-country"]
    return cat_features

@pytest.fixture(scope="module")
def train_dataset(data, features):
    train, test = train_test_split( data, test_size=0.20)
    X_train, y_train, encoder, lb = process_data(train,categorical_features=features,label="salary",training=True)
    return X_train, y_train

def test_import_data(data):
    try:
        assert isinstance(data, pd.DataFrame)
        print(" - Data type is DataFrame")
    except AssertionError as err:
        print(" - Error: data type is not DataFrame")
        raise err
    

def test_process_data(train_dataset):
    X_train = train_dataset
    try:
        assert type(X_train) is tuple
        print(" - X_train type is tuple")
    except AssertionError as err:
        print(" - Error: X_train type is not tuple")
        raise err


def test_model():
    savepath = "./model/model.pkl"
    try:
        assert os.path.isfile(savepath) == True
        #_ = joblib.load(open(savepath, 'rb'))
        print(" - Model is present")
    except AssertionError as err:
        print(" - Error: Model does not exist")
        raise err
    

def test_features(data, features):
    try:
        assert sorted(set(data.columns).intersection(features)) == sorted(features)
        print(" - Categorical features are in dataset")
    except AssertionError as err:
        print(" - Error: Features are missing from dataset")
        raise err


if __name__ == "__main__":
    test_import_data(data)
    test_process_data(data)
    test_model()
    test_features(data, features)