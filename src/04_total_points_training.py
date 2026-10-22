import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import joblib
import os

def train_total_points_model(df):
    print("Prepping data for Advanced Total Points XGBoost Regressor...")
    
    # 1. Target Variable: Total Game Points
    df['total_points'] = df['home_score'] + df['away_score']
    
    # Handle optional situational features if not present in ingested data
    if 'rest_diff' not in df.columns:
        df['rest_diff'] = 0
    if 'wind_speed' not in df.columns:
        df['wind_speed'] = 0.0
    if 'temperature' not in df.columns:
        df['temperature'] = 70.0

    # 2. Comprehensive Feature Matrix (Volume Sums + Differentials + Situational)
    features = [
        'sum_pts_scored', 'sum_pts_allowed',
        'sum_pass_yds', 'sum_rush_yds',
        'sum_net_yds_play', 'sum_off_epa', 'sum_def_epa',
        'diff_pts_scored', 'diff_pts_allowed',
        'diff_pass_yds', 'diff_rush_yds',
        'diff_comp_pct', 'diff_net_yds_play',
        'diff_off_epa', 'diff_def_epa', 'diff_off_cpoe',
        'diff_adj_off_epa',
        'rest_diff', 'wind_speed', 'temperature'
    ]
    
    active_features = [f for f in features if f in df.columns]
    
    clean_df = df.dropna(subset=active_features + ['total_points']).copy()
    
    X_train = clean_df[active_features]
    y_train = clean_df['total_points']
    
    # Isolate 2026 season as the test evaluation set
    test_df = clean_df[clean_df['season'] == 2026].copy()
    X_test = test_df[active_features]
    y_test_total = test_df['total_points']
    
    print(f"Training Advanced Total Points model on {len(X_train)} games with {len(active_features)} features...")

    model = xgb.XGBRegressor(
        colsample_bytree=0.8,
        learning_rate=0.01,
        max_depth=4,
        n_estimators=350,
        subsample=0.8,
        reg_lambda=1.5,
        random_state=42,
        objective='reg:absoluteerror'
    )
    
    model.fit(X_train, y_train)
    
    total_predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test_total, total_predictions)
    
    print("\n==================================")
    print("     TOTAL POINTS MODEL RESULTS   ")
    print("==================================")
    print(f"Mean Absolute Error (Total Points): {mae:.2f} points\n")
    
    return model

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "src" else SCRIPT_DIR
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    input_path = os.path.join(DATA_DIR, "model_features.csv")
    if not os.path.exists(input_path):
        print("Error: model_features.csv not found! Run 02_feature_engineering.py first.")
    else:
        df = pd.read_csv(input_path)
        total_model = train_total_points_model(df)
        
        model_path = os.path.join(MODEL_DIR, "xgb_total_points_model.joblib")
        joblib.dump(total_model, model_path)
        print(f"Advanced total points model serialized and saved to {model_path}")