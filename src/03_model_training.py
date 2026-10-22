import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, accuracy_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import norm
import joblib
import os
from sklearn.linear_model import Ridge

def american_odds_to_raw_prob(ml):
    if pd.isna(ml):
        return 0.50
    ml = float(ml)
    return (-ml) / (-ml + 100) if ml < 0 else 100 / (ml + 100)

def calculate_implied_prob(row):
    home_raw = american_odds_to_raw_prob(row['home_moneyline'])
    away_raw = american_odds_to_raw_prob(row['away_moneyline'])
    if home_raw + away_raw > 0:
        return home_raw / (home_raw + away_raw)
    return 0.50

def train_regression_model(df):
    print("Prepping data for XGBoost Regressor...")
    
    df['home_margin'] = df['home_score'] - df['away_score']
    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
    
    if 'home_moneyline' in df.columns:
        df['vegas_implied_prob'] = df.apply(calculate_implied_prob, axis=1)
    else:
        df['vegas_implied_prob'] = 0.50
        
    features = [
        'diff_pts_scored', 'diff_pts_allowed',
        'diff_pass_yds', 'diff_rush_yds',
        'diff_comp_pct', 'diff_net_yds_play',
        'diff_off_epa', 'diff_def_epa', 'diff_off_cpoe',
        'diff_adj_off_epa'
    ]
    
    clean_df = df.dropna(subset=features).copy()
    
    X_train = clean_df[features]
    y_train = clean_df['home_margin']
    
    # Keep 2026 as the test set for robust out-of-sample evaluation
    test_df = clean_df[clean_df['season'] == 2026].copy()
    X_test = test_df[features]
    y_test_margin = test_df['home_margin']
    y_test_win = test_df['home_win']
    
    print(f"Training on {len(X_train)} games with {len(features)} features...")

    model = xgb.XGBRegressor(
        colsample_bytree=0.8,
        learning_rate=0.01,
        max_depth=3,
        n_estimators=250,
        subsample=0.8,
        random_state=42,
        objective='reg:absoluteerror'
    )
    
    model.fit(X_train, y_train)
    
    margin_predictions = model.predict(X_test)
    STD_DEV = y_test_margin.std()
    prob_predictions = norm.cdf(margin_predictions / STD_DEV)
    binary_predictions = (prob_predictions > 0.5).astype(int)
    
    mae = mean_absolute_error(y_test_margin, margin_predictions)
    accuracy = accuracy_score(y_test_win, binary_predictions)
    brier_score = brier_score_loss(y_test_win, prob_predictions)
    
    print("\n==================================")
    print("       REGRESSION MODEL RESULTS     ")
    print("==================================")
    print(f"Mean Absolute Error (Margin): {mae:.2f} points")
    print(f"Win/Loss Accuracy: {accuracy * 100:.2f}%")
    print(f"Brier Score: {brier_score:.4f}\n")
    
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
        model_bundle = train_regression_model(df)
        
        # Save the ensemble model bundle
        model_path = os.path.join(MODEL_DIR, "xgb_margin_model.joblib")
        joblib.dump(model_bundle, model_path)
        print(f"Ensemble model bundle serialized and saved to {model_path}")