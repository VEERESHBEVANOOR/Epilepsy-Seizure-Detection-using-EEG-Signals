# src/train.py
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

from utils import load_data
from features import extract_statistical_features

def train_model():
    # 1. Load Raw Data
    print("🟢 1/6: Loading raw data...")
    X_raw, y = load_data()

    # 2. Extract Statistical Features from every row
    print("🟢 2/6: Extracting statistical features... (This may take a moment)")
    # Apply the feature extraction function to each row (which is a 178-point signal)
    features_list = X_raw.apply(lambda row: extract_statistical_features(row.values), axis=1)
    X_features = pd.DataFrame(list(features_list))

    print(f"✅ Feature extraction complete. New shape: {X_features.shape}")

    # 3. Split Data
    print("🟢 3/6: Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Create and Train Pipeline
    print("🟢 4/6: Building and training the new model pipeline...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])
    pipeline.fit(X_train, y_train)

    # 5. Evaluate Model
    print("🟢 5/6: Evaluating model performance...")
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print(f"✅ New Model Accuracy: {accuracy:.4f}")
    print("="*50 + "\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Non-Seizure", "Seizure"]))

    # 6. Save Pipeline and Confusion Matrix
    print("🟢 6/6: Saving the new model and confusion matrix...")
    os.makedirs("src/models", exist_ok=True)
    model_path = "src/models/seizure_model_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    print(f"✅ New pipeline saved successfully to {model_path}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Seizure", "Seizure"], yticklabels=["Non-Seizure", "Seizure"])
    plt.title('Confusion Matrix'); plt.ylabel('Actual Label'); plt.xlabel('Predicted Label')
    plt.savefig("src/models/confusion_matrix.png")
    print("✅ New confusion matrix saved.")

if __name__ == "__main__":
    train_model()