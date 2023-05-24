# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Denes Kovacs created the model. It is Random Forest Classifier using the default hyperparameters in scikit-learn.

## Intended Use

This model should be used to predict if someone's income is below or above 50k based on his/her workclass, education, marital-status, occupation relationship, race, sex , or native-country

## Training Data

The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income).

The original data set has 48842 instances, and a split was used to break this into a train and test set. Stratification was done. To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.

## Metrics
The model was evaluation results are: precision:0.748, recall:0.263, fbeta:0.389.

## Caveats and Recommendations
Further information on this can be found on this link: https://archive.ics.uci.edu/ml/support/census+income#686ccd571de3e92d89394f207348bd5a669558d2

## Evaluation Data
20% of the dataset has been used for model evaluation.

## Ethical Considerations
The dataset is not a fair representation of salary distribution, further assumptions should be avoided.
