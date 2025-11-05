import pandas as pd
import joblib

# Load model
clf = joblib.load("models/seizure_clf.joblib")

def predict_eeg(df):
    """
    df: a dataframe with the same columns as training features
    """
    prediction = clf.predict(df)
    return prediction
