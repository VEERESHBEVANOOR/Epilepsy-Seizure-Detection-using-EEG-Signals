# src/utils.py
import pandas as pd

def load_data():
    """
    Loads the Epileptic Seizure Recognition dataset and prepares it for
    binary classification, correctly handling the identifier column.
    """
    data_path = "data/raw/Epileptic Seizure Recognition.csv"
    df = pd.read_csv(data_path)

    # The dataset has an identifier column, 178 feature columns, 
    # and a final label column. The identifier contains strings and must be removed.

    # We select columns by position to avoid issues with column names.
    # Features (X) are all columns from the second (index 1) to the second-to-last.
    X = df.iloc[:, 1:-1]
    # The label (y) is the very last column.
    y = df.iloc[:, -1]

    # The label 'y' has values 1,2,3,4,5. Only '1' indicates a seizure.
    # We will convert this to a binary problem: 1 for seizure, 0 for non-seizure.
    y = y.apply(lambda val: 1 if val == 1 else 0)

    # Ensure all feature data is numeric before returning
    X = X.astype(float)

    print(f"✅ Loaded {len(X)} samples with {X.shape[1]} features.")
    return X, y