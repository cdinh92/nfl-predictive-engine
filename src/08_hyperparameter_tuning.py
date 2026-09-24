import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import os

# --- 1. Load the Data ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "src" else SCRIPT_DIR
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

print("Loading historical data for tuning...")
features_df = pd.read_csv(os.path.join(DATA_DIR, "model_features.csv"))

# Ensure data is strictly chronological
features_df = features_df.sort_values(by=['season', 'week']).reset_index(drop=True)

# 1. Target Variable: Point Margin (Home Score - Away Score)
features_df['home_margin'] = features_df['home_score'] - features_df['away_score']

# 2. Use the 9 core differential features (matching your current model)
features = [
    'diff_pts_scored', 'diff_pts_allowed',
    'diff_pass_yds', 'diff_rush_yds',
    'diff_comp_pct', 'diff_net_yds_play',
    'diff_off_epa', 'diff_def_epa', 'diff_off_cpoe'
]

# Filter down to rows with complete feature data up to the test boundary
tuning_df = features_df.dropna(subset=features + ['home_margin']).copy()
X = tuning_df[features]
y = tuning_df['home_margin']

# --- 2. Configure the Tuning Grid ---
param_grid = {
    'max_depth': [3, 4, 5],                  # Tree depth to prevent overfitting
    'learning_rate': [0.01, 0.05, 0.1],      # Step size shrinkage
    'n_estimators': [100, 150, 250],         # Number of trees
    'subsample': [0.8, 1.0],                 # Row sampling per tree
    'colsample_bytree': [0.8, 1.0]           # Feature sampling per tree
}

# --- 3. Set Up Time-Series Cross-Validation ---
# Prevents data leakage by ensuring training sets always precede validation sets chronologically
tscv = TimeSeriesSplit(n_splits=5)

# Use XGBRegressor to optimize for point margins using Negative Mean Absolute Error
model = xgb.XGBRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=tscv,               
    scoring='neg_mean_absolute_error',    # Optimize for margin accuracy (MAE)
    verbose=1,             
    n_jobs=-1              
)

# --- 4. Run the Grid Search ---
print("\nStarting Hyperparameter Tuning for XGBRegressor... This may take a few moments...")
grid_search.fit(X, y)

# --- 5. Output the Results ---
print("\n✅ Tuning Complete!")
print("-" * 30)
print(f"Best Hyperparameters Found:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"Best Cross-Validation MAE: {-grid_search.best_score_:.2f} points")
print("-" * 30)
print("Update your model initialization in 03_model_training.py to use these best parameters!")