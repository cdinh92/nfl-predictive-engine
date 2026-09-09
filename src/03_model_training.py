import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import os

def train_baseline_model(df):
    print("Prepping data for XGBoost...")
    
    # 1. Create the Target Variable: 1 if Home Team wins, 0 otherwise
    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
    
    # 2. Define our Features (X)
    features = [
        'home_roll_pts_scored', 'home_roll_pts_allowed', 
        'away_roll_pts_scored', 'away_roll_pts_allowed'
    ]
    
    # 3. Chronological Train/Test Split
    # Train on 2023 and 2024, test on 2025
    train_df = df[df['season'] < 2025]
    test_df = df[df['season'] >= 2025]
    
    X_train, y_train = train_df[features], train_df['home_win']
    X_test, y_test = test_df[features], test_df['home_win']
    
    print(f"Training on {len(X_train)} games...")
    print(f"Testing against {len(X_test)} games in the 2025 season...")
    
    # 4. Initialize and Train the Model
    model = xgb.XGBClassifier(
        n_estimators=100,      # Number of decision trees
        learning_rate=0.1,     # How aggressively it learns
        random_state=42        # Ensures we get the exact same results every time
    )
    model.fit(X_train, y_train)
    
    # 5. Make Predictions and Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print("\n==================================")
    print("       BASELINE MODEL RESULTS       ")
    print("==================================")
    print(f"Accuracy: {accuracy * 100:.2f}%\n")
    
    # 6. See which stats the model thinks are most important
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance Weight': model.feature_importances_
    }).sort_values(by='Importance Weight', ascending=False)
    
    print("What drove the predictions?")
    print(importance_df.to_string(index=False))
    
    return model

if __name__ == "__main__":
    # Smart Pathing 
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "src" else SCRIPT_DIR
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")

    # Load the engineered features
    input_path = os.path.join(DATA_DIR, "model_features.csv")
    
    if not os.path.exists(input_path):
        print("Error: model_features.csv not found! Run 02_feature_engineering.py first.")
    else:
        df = pd.read_csv(input_path)
        model = train_baseline_model(df)