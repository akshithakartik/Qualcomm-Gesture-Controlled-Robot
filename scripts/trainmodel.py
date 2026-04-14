import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

MODEL_FILE = 'gesture_classifier111.pkl'
CSV_FILE = 'gesture_data.csv'
N_SPLITS = 5 

def train_and_validate_model():
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: CSV file '{CSV_FILE}' not found. Run 1_collect_data.py first.")
        return

    print(f"Data loaded: {len(df)} samples with {len(df.columns) - 1} features each.")

    X = df.drop('label', axis=1)
    y = df['label']

    print(X.shape, y.shape, df.shape)

    # scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv_strategy = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    model = SVC(kernel='linear', C=1.0, random_state=42)
    
    print(f"Performing {N_SPLITS}-Fold Cross-Validation...")

    # cross val
    cv_scores = cross_val_score(
        estimator=model, 
        X=X_scaled, 
        y=y, 
        cv=cv_strategy, 
        scoring='accuracy'
    )
    
    print("\n--- Cross-Validation Results ---")
    print(f"Individual Fold Accuracies: {cv_scores}")
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f} 95% CI)")
    
    # train model
    print("\nTraining final model on all data for deployment...")
    model.fit(X_scaled, y)

    # save model and scaler
    full_model_package = {
        'model': model, 
        'scaler': scaler, 
        'cv_mean_accuracy': np.mean(cv_scores)
    }
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(full_model_package, f)

    print(f"\nFinal model and scaler saved as '{MODEL_FILE}'.")

if __name__ == "__main__":
    train_and_validate_model()