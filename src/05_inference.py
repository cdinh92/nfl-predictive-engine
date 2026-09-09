import pandas as pd
import xgboost as xgb
import os

def get_latest_team_stats(features_df):
    print("Extracting current team strengths...")
    latest_stats = {}
    teams = features_df['home_team'].unique()
    
    for team in teams:
        # Find all games this team played and grab the absolute most recent one
        team_games = features_df[(features_df['home_team'] == team) | (features_df['away_team'] == team)]
        last_game = team_games.iloc[-1]
        
        # Extract their rolling stats depending on if they were Home or Away in their last game
        if last_game['home_team'] == team:
            latest_stats[team] = {
                'roll_pts_scored': last_game['home_roll_pts_scored'],
                'roll_pts_allowed': last_game['home_roll_pts_allowed']
            }
        else:
            latest_stats[team] = {
                'roll_pts_scored': last_game['away_roll_pts_scored'],
                'roll_pts_allowed': last_game['away_roll_pts_allowed']
            }
    return latest_stats

def american_odds_to_prob(ml):
    """Converts Vegas Moneyline (e.g., -150 or +130) to an implied win percentage."""
    if pd.isna(ml): 
        return 0.0
    if ml < 0:
        return (-ml) / (-ml + 100)
    else:
        return 100 / (ml + 100)

if __name__ == "__main__":
    # Smart Pathing
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "src" else SCRIPT_DIR
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")

    # 1. Load Data
    features_df = pd.read_csv(os.path.join(DATA_DIR, "model_features.csv"))
    odds_df = pd.read_csv(os.path.join(DATA_DIR, "upcoming_odds.csv"))

    # 2. Train Model on all historical data
    print("Training XGBoost on historical data...")
    features = ['home_roll_pts_scored', 'home_roll_pts_allowed', 'away_roll_pts_scored', 'away_roll_pts_allowed']
    features_df['home_win'] = (features_df['home_score'] > features_df['away_score']).astype(int)
    
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(features_df[features], features_df['home_win'])

    # 3. Build features for upcoming games
    latest_stats = get_latest_team_stats(features_df)
    matchups = []

    for _, row in odds_df.iterrows():
        home, away = row['home_team'], row['away_team']
        if home in latest_stats and away in latest_stats:
            matchups.append({
                'Away': away,
                'Home': home,
                'Vegas_Home_ML': row['home_moneyline'],
                'home_roll_pts_scored': latest_stats[home]['roll_pts_scored'],
                'home_roll_pts_allowed': latest_stats[home]['roll_pts_allowed'],
                'away_roll_pts_scored': latest_stats[away]['roll_pts_scored'],
                'away_roll_pts_allowed': latest_stats[away]['roll_pts_allowed']
            })
    
    predict_df = pd.DataFrame(matchups)

    # 4. Make Predictions
    print("Simulating upcoming games...\n")
    # predict_proba returns [Probability of Loss, Probability of Win]
    predict_df['Model_Win_Prob'] = model.predict_proba(predict_df[features])[:, 1]
    
    # 5. Calculate Edge vs Vegas
    predict_df['Vegas_Implied_Prob'] = predict_df['Vegas_Home_ML'].apply(american_odds_to_prob)
    predict_df['Edge'] = predict_df['Model_Win_Prob'] - predict_df['Vegas_Implied_Prob']

    # Format for display
    predict_df['Model_Win_Prob'] = (predict_df['Model_Win_Prob'] * 100).round(1).astype(str) + '%'
    predict_df['Vegas_Implied_Prob'] = (predict_df['Vegas_Implied_Prob'] * 100).round(1).astype(str) + '%'
    
    # Show games where the model finds the most value (highest positive edge)
    results = predict_df[['Away', 'Home', 'Vegas_Home_ML', 'Vegas_Implied_Prob', 'Model_Win_Prob', 'Edge']]
    results = results.sort_values(by='Edge', ascending=False)
    
    print("==================================================================")
    print("                   UPCOMING NFL PREDICTIONS                       ")
    print("==================================================================")
    print(results.to_string(index=False))
    print("==================================================================")
    print("* Edge > 0.05 (5%) indicates the model likes the Home Team more than Vegas does.")
    print("* Edge < -0.05 indicates the model likes the Away Team more than Vegas does.")