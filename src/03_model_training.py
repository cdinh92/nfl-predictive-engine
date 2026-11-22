import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error


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

    # 1. Strict temporal split: train on pre-2026 seasons, validate out-of-sample on 2026
    train_df = clean_df[clean_df['season'] < 2026].copy()
    test_df = clean_df[clean_df['season'] == 2026].copy()

    X_train, y_train = train_df[features], train_df['home_margin']
    X_test, y_test_margin = test_df[features], test_df['home_margin']
    y_test_win = test_df['home_win']

    xgb_params = dict(
        colsample_bytree=0.8,
        learning_rate=0.01,
        max_depth=3,
        n_estimators=250,
        subsample=0.8,
        random_state=42,
        objective='reg:absoluteerror'
    )

    print(f"Validating on {len(X_train)} pre-2026 games -> {len(X_test)} 2026 games...")
    val_model = xgb.XGBRegressor(**xgb_params)
    val_model.fit(X_train, y_train)

    # 2. Out-of-sample predictions & variance scaling factor
    raw_test_preds = val_model.predict(X_test)
    oof_std = float(np.std(raw_test_preds))

    # Match historical closing spread std dev (~5.5 pts), or use actual spread_line std if present
    if 'spread_line' in test_df.columns and test_df['spread_line'].notna().sum() > 20:
        target_spread_std = float(test_df['spread_line'].dropna().std())
    else:
        target_spread_std = 5.5

    scale_factor = round(target_spread_std / oof_std, 4) if oof_std > 0 else 1.0
    calibrated_test_preds = raw_test_preds * scale_factor

    # Win probability calibration uses scaled margin against actual game margin std dev
    game_margin_std = float(y_test_margin.std())
    prob_predictions = norm.cdf(calibrated_test_preds / game_margin_std)
    binary_predictions = (prob_predictions > 0.5).astype(int)

    raw_mae = mean_absolute_error(y_test_margin, raw_test_preds)
    cal_mae = mean_absolute_error(y_test_margin, calibrated_test_preds)
    accuracy = accuracy_score(y_test_win, binary_predictions)
    brier_score = brier_score_loss(y_test_win, prob_predictions)

    print("\n==================================")
    print("       REGRESSION MODEL RESULTS     ")
    print("==================================")
    print(f"OOF Prediction Std Dev: {oof_std:.3f} -> Target Spread Std: {target_spread_std:.3f}")
    print(f"Variance Scale Factor:  {scale_factor:.4f}x")
    print(f"MAE (Raw / Calibrated): {raw_mae:.2f} / {cal_mae:.2f} points")
    print(f"Win/Loss Accuracy:      {accuracy * 100:.2f}%")
    print(f"Brier Score:            {brier_score:.4f}\n")

    # 3. Refit final production model on all completed games (including 2026)
    X_full = clean_df[features]
    y_full = clean_df['home_margin']
    print(f"Refitting production model on all {len(X_full)} games...")

    final_model = xgb.XGBRegressor(**xgb_params)
    final_model.fit(X_full, y_full)

    return {
        "model": final_model,
        "features": features,
        "scale_factor": scale_factor,
        "oof_std": round(oof_std, 4),
        "target_spread_std": round(target_spread_std, 4),
        "game_margin_std": round(game_margin_std, 4),
    }


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

        model_path = os.path.join(MODEL_DIR, "xgb_margin_model.joblib")
        joblib.dump(model_bundle, model_path)
        print(f"Ensemble model bundle serialized and saved to {model_path}")