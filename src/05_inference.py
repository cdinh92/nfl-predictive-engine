import pandas as pd
import xgboost as xgb
import os
import json
from datetime import datetime, timezone

def get_latest_team_stats(features_df):
    print("Extracting current team strengths...")
    latest_stats = {}
    teams = features_df['home_team'].unique()
    
    for team in teams:
        team_games = features_df[(features_df['home_team'] == team) | (features_df['away_team'] == team)]
        if team_games.empty:
            continue
            
        last_game = team_games.iloc[-1]
        
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
    """Converts Vegas Moneyline to an implied win percentage."""
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

    features_df = pd.read_csv(os.path.join(DATA_DIR, "model_features.csv"))
    odds_df = pd.read_csv(os.path.join(DATA_DIR, "upcoming_odds.csv"))

    print("Training XGBoost on historical data...")
    features = ['home_roll_pts_scored', 'home_roll_pts_allowed', 'away_roll_pts_scored', 'away_roll_pts_allowed']
    features_df['home_win'] = (features_df['home_score'] > features_df['away_score']).astype(int)
    
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(features_df[features], features_df['home_win'])

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

    print("Simulating upcoming games...\n")
    predict_df['Model_Win_Prob'] = model.predict_proba(predict_df[features])[:, 1]
    
    predict_df['Vegas_Implied_Prob'] = predict_df['Vegas_Home_ML'].apply(american_odds_to_prob)
    predict_df['Edge'] = predict_df['Model_Win_Prob'] - predict_df['Vegas_Implied_Prob']

    # Keep a copy of the raw numbers for the JSON before we format them as strings
    raw_results = predict_df.copy()

    predict_df['Model_Win_Prob'] = (predict_df['Model_Win_Prob'] * 100).round(1).astype(str) + '%'
    predict_df['Vegas_Implied_Prob'] = (predict_df['Vegas_Implied_Prob'] * 100).round(1).astype(str) + '%'
    
    results = predict_df[['Away', 'Home', 'Vegas_Home_ML', 'Vegas_Implied_Prob', 'Model_Win_Prob', 'Edge']]
    results = results.sort_values(by='Edge', ascending=False)
    
    print("==================================================================")
    print("                   UPCOMING NFL PREDICTIONS                       ")
    print("==================================================================")
    print(results.to_string(index=False))
    print("==================================================================")

    # --- NEW: JSON EXPORT FOR WEB ---
    print("\nPackaging predictions for web display...")
    
    # Sort the raw results to match the terminal output
    raw_results = raw_results.sort_values(by='Edge', ascending=False)
    
    json_payload = {
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "season": 2025, 
        "week": "Upcoming",
        "games": []
    }
    
    for _, row in raw_results.iterrows():
        # Tag high confidence if the edge is > 5% or < -5%
        is_high_confidence = abs(row['Edge']) >= 0.05
        
        game_data = {
            "game_id": f"2025_{row['Away']}_{row['Home']}",
            "home_team": {
                "abbr": row['Home'],
                "moneyline": row['Vegas_Home_ML']
            },
            "away_team": {
                "abbr": row['Away']
            },
            "prediction": {
                "model_home_win_prob": round(row['Model_Win_Prob'], 3),
                "vegas_home_implied_prob": round(row['Vegas_Implied_Prob'], 3),
                "edge": round(row['Edge'], 3),
                "pick": row['Home'] if row['Edge'] > 0 else row['Away'],
                "confidence_tier": "High" if is_high_confidence else "Standard"
            }
        }
        json_payload["games"].append(game_data)
        
    json_path = os.path.join(DATA_DIR, "predictions_latest.json")
    with open(json_path, 'w') as f:
        json.dump(json_payload, f, indent=2)
        
    print(f"✅ Successfully exported structured JSON to {json_path}")