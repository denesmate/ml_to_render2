from fastapi import FastAPI, HTTPException
from typing import Union, Optional
# BaseModel from Pydantic is used to define data objects
from pydantic import BaseModel
import pandas as pd
import os
from ml.data import process_data
from ml.model import inference
import joblib

 # path to saved artifacts
PTH = './model/'
filename = ['model.pkl', 'encoder.pkl', 'labelizer.pkl']

# Declare the data object with its components and their type.
class InputData(BaseModel):
    age: int
    workclass: str 
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
                        "example": {
                                    'age':38,
                                    'workclass':"Private", 
                                    'fnlgt':215646,
                                    'education':"HS-grad",
                                    'education_num':9,
                                    'marital_status':"Divorced",
                                    'occupation':"Handlers-cleaners",
                                    'relationship':"Not-in-family",
                                    'race':"White",
                                    'sex':"Male",
                                    'capital_gain':0,
                                    'capital_loss':0,
                                    'hours_per_week':40,
                                    'native_country':"United-States"
                                    }
                        }


# instantiate FastAPI instance
app = FastAPI()


# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return "Hello! Please post your data to get inference result!"



@app.post("/inference/")
async def ingest_data(inference: InputData):
    data = {  'age': inference.age,
                'workclass': inference.workclass, 
                'fnlgt': inference.fnlgt,
                'education': inference.education,
                'education-num': inference.education_num,
                'marital-status': inference.marital_status,
                'occupation': inference.occupation,
                'relationship': inference.relationship,
                'race': inference.race,
                'sex': inference.sex,
                'capital-gain': inference.capital_gain,
                'capital-loss': inference.capital_loss,
                'hours-per-week': inference.hours_per_week,
                'native-country': inference.native_country,
                }


    sample = pd.DataFrame(data, index=[0])


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


    if os.path.isfile(os.path.join(PTH + filename[0])):
        model = joblib.load(PTH + filename[0])
        encoder = joblib.load(PTH + filename[1])
        lb = joblib.load(PTH + filename[2])
        
    sample,_,_,_ = process_data(
                                sample, 
                                categorical_features=cat_features, 
                                training=False, 
                                encoder=encoder, 
                                lb=lb
                                )

                        
    prediction = model.predict(sample)


    if prediction[0]>0.5:
        prediction = '>50K'
    else:
        prediction = '<=50K', 
    data['prediction'] = prediction


    return data


if __name__ == '__main__':
    pass