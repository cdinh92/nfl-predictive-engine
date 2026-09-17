import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import os

# --- 1. Load the Data ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "src" else SCRIPT_DIR
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

print("Loading historical data for tuning...")
features_df = pd.read_csv(os.path.join(DATA_DIR, "model_features.csv"))

# Ensure data is strictly chronological for TimeSeriesSplit
features_df = features_df.sort_values(by=['season', 'week']).reset_index(drop=True)

# Define features and target
features = [
    'home_roll_pts_scored', 'home_roll_pts_allowed', 'away_roll_pts_scored', 'away_roll_pts_allowed',
    'home_roll_passing_yards', 'home_roll_completion_pct', 'home_roll_net_yds_per_play',
    'away_roll_passing_yards', 'away_roll_completion_pct', 'away_roll_net_yds_per_play'
]
X = features_df[features]
y = (features_df['home_score'] > features_df['away_score']).astype(int)

# --- 2. Configure the Tuning Grid ---
# These are the hyperparameters the system will test
param_grid = {
    'max_depth': [3, 4, 5],                  # How deep the trees go
    'learning_rate': [0.01, 0.05, 0.1],      # How fast the model learns
    'n_estimators': [100, 200, 300],         # Number of trees
    'subsample': [0.8, 1.0]                  # Percentage of data used per tree (prevents overfitting)
}

# --- 3. Set Up Time-Series Cross-Validation ---
# n_splits=5 means it will test the model across 5 different points in time
tscv = TimeSeriesSplit(n_splits=5)

model = xgb.XGBClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=tscv,               # Use TimeSeriesSplit instead of random K-Fold
    scoring='accuracy',    # We want to optimize for pure prediction accuracy
    verbose=1,             # Print progress to the terminal
    n_jobs=-1              # Use all available CPU cores to speed up training
)

# --- 4. Run the Grid Search ---
print("\nStarting Hyperparameter Tuning. This may take a few minutes...")
grid_search.fit(X, y)

# --- 5. Output the Results ---
print("\n✅ Tuning Complete!")
print("-" * 30)
print(f"Best Hyperparameters Found:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_ * 100:.2f}%")
print("-" * 30)
print("Update your prediction scripts to use these new parameters inside xgb.XGBClassifier()!")