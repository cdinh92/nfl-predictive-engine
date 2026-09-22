import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
import joblib
import os

def train_baseline_model(df):
    print("Prepping data for XGBoost...")
    
    # 1. Create the Target Variable: 1 if Home Team wins, 0 otherwise
    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
    
    # 2. Define Features (X)
    # Note: Ensure these match the exact features you intend to predict with
    features = [
        'home_roll_pts_scored', 'home_roll_pts_allowed', 
        'away_roll_pts_scored', 'away_roll_pts_allowed'
    ]
    
    # 3. Chronological Train/Test Split
    train_df = df[df['season'] < 2025]
    test_df = df[df['season'] >= 2025]
    
    X_train, y_train = train_df[features], train_df['home_win']
    X_test, y_test = test_df[features], test_df['home_win']
    
    print(f"Training on {len(X_train)} games...")
    print(f"Testing against {len(X_test)} games in the 2025 season...")
    
    # 4. Initialize the Base Model
    base_model = xgb.XGBClassifier(
        n_estimators=100,      
        learning_rate=0.1,     
        random_state=42        
    )
    
    # 5. Wrap with Isotonic Calibration
    # 'cv=5' uses cross-validation to fit both the base model and the calibrator
    print("Calibrating probabilities with Isotonic Regression...")
    calibrated_model = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv=5)
    calibrated_model.fit(X_train, y_train)
    
    # 6. Make Predictions and Evaluate
    predictions = calibrated_model.predict(X_test)
    prob_predictions = calibrated_model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, predictions)
    brier_score = brier_score_loss(y_test, prob_predictions)
    
    print("\n==================================")
    print("       CALIBRATED MODEL RESULTS     ")
    print("==================================")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Brier Score (Closer to 0 is better): {brier_score:.4f}\n")
    
    return calibrated_model

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "src" else SCRIPT_DIR
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
    
    # Ensure the models directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    input_path = os.path.join(DATA_DIR, "model_features.csv")
    
    if not os.path.exists(input_path):
        print("Error: model_features.csv not found! Run 02_feature_engineering.py first.")
    else:
        df = pd.read_csv(input_path)
        final_model = train_baseline_model(df)
        
        # Save the fully calibrated pipeline
        model_path = os.path.join(MODEL_DIR, "calibrated_xgb_model.joblib")
        joblib.dump(final_model, model_path)
        print(f"✅ Calibrated model serialized and saved to {model_path}")