import json
from fastapi.testclient import TestClient
from fastapi import HTTPException
from main import app


data =  {  'age':53,
                'workclass':"Private", 
                'fnlgt':234721,
                'education':"11th",
                'education_num':7,
                'marital_status':"Married-civ-spouse",
                'occupation':"Handlers-cleaners",
                'relationship':"Husband",
                'race':"Black",
                'sex':"Male",
                'capital_gain':0,
                'capital_loss':0,
                'hours_per_week':40,
                'native_country':"United-States"
        }

data2 =  {  'age':50,
                'workclass':"Private", 
                'fnlgt':234721,
                'education':"Doctorate",
                'education_num':16,
                'marital_status':"Separated",
                'occupation':"Exec-managerial",
                'relationship':"Not-in-family",
                'race':"Black",
                'sex':"Female",
                'capital_gain':0,
                'capital_loss':0,
                'hours_per_week':50,
                'native_country':"United-States"
            }

# Instantiate the testing client with our app.
client = TestClient(app)

# Test GET method with "statenumbers/60" endpoint
def test_get_item():

    r = client.get("/")

    assert r.status_code == 200
    assert r.json() == "Hello! Please post your data to get inference result!"

def test_post_data_success():

    r = client.post("/inference/", data=json.dumps(data))
    assert r.status_code == 200

def test_post_data_success_result_and_inference():
    r = client.post("/inference/", data=json.dumps(data))
    assert r.status_code == 200
    assert r.json()["age"] == 53
    assert r.json()["fnlgt"] == 234721
    assert r.json()["prediction"][0] == '<=50K'

def test_post_data_success_result_and_inference_v2():
    r = client.post("/inference/", data=json.dumps(data2))
    assert r.status_code == 200
    assert r.json()["age"] == 50
    assert r.json()["fnlgt"] == 234721
    assert r.json()["prediction"][0] == '>=50K'