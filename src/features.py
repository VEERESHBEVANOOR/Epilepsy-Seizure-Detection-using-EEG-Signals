# src/features.py
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def hjorth_parameters(x):
    """Calculate Hjorth parameters (Activity, Mobility, Complexity)."""
    if np.std(x) == 0:
        return 0, 0, 0
    activity = np.var(x)
    mobility = np.sqrt(np.var(np.diff(x)) / activity)
    diff2 = np.diff(np.diff(x))
    if np.var(np.diff(x)) == 0:
        complexity = 0
    else:
        complexity = np.sqrt(np.var(diff2) / np.var(np.diff(x))) / mobility
    return activity, mobility, complexity

def extract_statistical_features(signal_window: np.ndarray) -> dict:
    """
    Extracts a dictionary of statistical features from a 1D signal window.
    """
    # --- THIS IS THE NEW, ROBUST NORMALIZATION STEP ---
    # It makes the feature extraction immune to scaling issues.
    # We check for a non-zero standard deviation to avoid division by zero.
    if np.std(signal_window) > 1e-6:
        signal_window = (signal_window - np.mean(signal_window)) / np.std(signal_window)
    # --- END OF NEW STEP ---

    features = {}
    features['mean'] = np.mean(signal_window)
    features['std'] = np.std(signal_window)
    features['skew'] = skew(signal_window)
    features['kurtosis'] = kurtosis(signal_window)
    
    activity, mobility, complexity = hjorth_parameters(signal_window)
    features['hjorth_activity'] = activity
    features['hjorth_mobility'] = mobility
    features['hjorth_complexity'] = complexity
    
    return features

def dataframe_to_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function is now a wrapper that processes a single-column DataFrame
    by applying the feature extraction to its signal.
    """
    signal = df.iloc[:, 0].dropna().values
    feature_dict = extract_statistical_features(signal)
    return pd.DataFrame([feature_dict])