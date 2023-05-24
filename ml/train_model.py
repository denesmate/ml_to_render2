# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from model import train_model, compute_model_metrics, inference, compute_slice
from data import process_data
import pandas as pd
import logging
import joblib

# Add code to load in the data.

PTH = "../data/census.csv"
data = pd.read_csv(PTH, skipinitialspace = True)

logging.basicConfig(filename='metrics.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')

train, test = train_test_split(data, test_size=0.20)

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

# Proces the test data with the process_data function.

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder,
    lb=lb
)

# Train and save a model.
savepath = '../model/'
filename = ['model.pkl', 'encoder.pkl', 'labelizer.pkl']

model = train_model(X_train, y_train)

joblib.dump(model, savepath + filename[0])
joblib.dump(encoder, savepath + filename[1])
joblib.dump(lb, savepath + filename[2])

preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

logging.info(f"Classification target labels: {list(lb.classes_)}")
logging.info(
    f"precision:{precision:.3f}, recall:{recall:.3f}, fbeta:{fbeta:.3f}")


slice_path = "./slice_output.txt"

for feature in cat_features:
    performance_df = compute_slice(test, feature, y_test, preds)
    performance_df.to_csv(slice_path,  mode='a', index=False)
    logging.info(f"Performance on slice {feature}")
    logging.info(performance_df)