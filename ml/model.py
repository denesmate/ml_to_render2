from sklearn.metrics import precision_score, recall_score, fbeta_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    param_grid = { 
        'n_estimators': [10],
        'max_features': ['auto'],
        'max_depth' : [5],
        'criterion' :['entropy']
    }

    rfc = RandomForestClassifier(random_state=42)

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

    cv_rfc.fit(X_train, y_train)

    return cv_rfc


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)

    return preds


def compute_slice(df, feature, y, preds):
    slice_options = df[feature].unique().tolist()
    performance_df = pd.DataFrame(index=slice_options, 
                            columns=['feature','n_samples', 'fbeta', 'precision', 'recall'])
    for option in slice_options:
        slice_mask = df[feature]==option

        slice_y = y[slice_mask]
        slice_preds = preds[slice_mask]
        precision, recall, fbeta = compute_model_metrics(slice_y, slice_preds)
        
        performance_df.at[option, 'feature'] = feature
        performance_df.at[option, 'n_samples'] = len(slice_y)
        performance_df.at[option, 'fbeta'] = fbeta
        performance_df.at[option, 'precision'] = precision
        performance_df.at[option, 'recall'] = recall

    performance_df.reset_index(names='feature value', inplace=True)
    colList = list(performance_df.columns)
    colList[0], colList[1] =  colList[1], colList[0]
    performance_df = performance_df[colList]

    return performance_df
