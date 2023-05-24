import requests
import json

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

r = requests.post("https://ml-model-with-fastapi-02.onrender.com/inference", data=json.dumps(data))

print(r.status_code)
print(r.json())