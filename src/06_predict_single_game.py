import pandas as pd
import xgboost as xgb
import os
from datetime import datetime, timezone
import json

def get_team_latest_stats(features_df, team):
    """Finds the most recent rolling stats for a specific team."""
    team_games = features_df[(features_df['home_team'] == team) | (features_df['away_team'] == team)]
    if team_games.empty:
        return None
        
    last_game = team_games.iloc[-1]
    
    if last_game['home_team'] == team:
        return {
            'roll_pts_scored': last_game['home_roll_pts_scored'],
            'roll_pts_allowed': last_game['home_roll_pts_allowed']
        }
    else:
        return {
            'roll_pts_scored': last_game['away_roll_pts_scored'],
            'roll_pts_allowed': last_game['away_roll_pts_allowed']
        }

def american_odds_to_prob(ml):
    """Converts Vegas Moneyline to an implied win percentage."""
    if ml < 0:
        return (-ml) / (-ml + 100)
    else:
        return 100 / (ml + 100)

def log_prediction_to_ledger(data_dir, game_id, away_team, home_team, vegas_ml, vegas_prob, model_prob, edge, pick):
    ledger_path = os.path.join(data_dir, "prediction_ledger.csv")
    
    new_entry = pd.DataFrame([{
        "prediction_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "game_id": game_id,
        "away_team": away_team,
        "home_team": home_team,
        "vegas_home_ml": vegas_ml,
        "vegas_implied_prob": round(vegas_prob, 4),
        "model_home_win_prob": round(model_prob, 4),
        "edge": round(edge, 4),
        "model_pick": pick,
        "actual_home_score": None,
        "actual_away_score": None,
        "actual_winner": None,
        "pick_correct": None,
        "units_won": None
    }])
    
    if not os.path.exists(ledger_path):
        new_entry.to_csv(ledger_path, index=False)
    else:
        new_entry.to_csv(ledger_path, mode='a', header=False, index=False)
        
    print(f"\n✅ Logged prediction to {os.path.abspath(ledger_path)}")

if __name__ == "__main__":
    # Smart Pathing
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "src" else SCRIPT_DIR
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")

    # Load Data
    features_df = pd.read_csv(os.path.join(DATA_DIR, "model_features.csv"))

    # Train Model on all historical data
    print("Training XGBoost on historical data...")
    features = ['home_roll_pts_scored', 'home_roll_pts_allowed', 'away_roll_pts_scored', 'away_roll_pts_allowed']
    features_df['home_win'] = (features_df['home_score'] > features_df['away_score']).astype(int)
    
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(features_df[features], features_df['home_win'])

    # Hardcode the specific matchup details
    home_team = 'SEA'
    away_team = 'NE'
    vegas_home_ml = -170
    
    print(f"\nAnalyzing: {away_team} @ {home_team}...")
    
    home_stats = get_team_latest_stats(features_df, home_team)
    away_stats = get_team_latest_stats(features_df, away_team)
    
    if home_stats and away_stats:
        # Construct the single row for prediction
        matchup_data = pd.DataFrame([{
            'home_roll_pts_scored': home_stats['roll_pts_scored'],
            'home_roll_pts_allowed': home_stats['roll_pts_allowed'],
            'away_roll_pts_scored': away_stats['roll_pts_scored'],
            'away_roll_pts_allowed': away_stats['roll_pts_allowed']
        }])

        print("\n--- MODEL INPUT FEATURES ---")
        print(matchup_data.to_string(index=False))
        
        # Calculate Probabilities
        model_home_win_prob = model.predict_proba(matchup_data[features])[0][1]
        vegas_implied_prob = american_odds_to_prob(vegas_home_ml)
        edge = model_home_win_prob - vegas_implied_prob
        pick = home_team if edge > 0 else away_team
        
        print("\n==================================================================")
        print("                   SINGLE GAME PREDICTION                         ")
        print("==================================================================")
        print(f"Matchup:             {away_team} @ {home_team}")
        print(f"Vegas Moneyline:     {vegas_home_ml} (Implied Prob: {vegas_implied_prob * 100:.1f}%)")
        print(f"Model Win Prob:      {model_home_win_prob * 100:.1f}%")
        print(f"Mathematical Edge:   {edge * 100:.1f}%")
        print("==================================================================")

        log_prediction_to_ledger(
                data_dir=DATA_DIR,
                game_id=f'2026_{away_team}_{home_team}',
                away_team=away_team,
                home_team=home_team,
                vegas_ml=vegas_home_ml,
                vegas_prob=vegas_implied_prob,
                model_prob=model_home_win_prob,
                edge=edge,
                pick=pick,
            )

        # Export for Khoa's frontend dashboard
        json_path = os.path.join(DATA_DIR, "predictions_latest.json")
        is_high_confidence = abs(edge) >= 0.05

        json_payload = {
            "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "season": 2026,
            "week": "Kickoff",
            "games": [
                {
                    "game_id": f"2026_{away_team}_{home_team}",
                    "home_team": {
                        "abbr": home_team,
                        "moneyline": int(vegas_home_ml)
                    },
                    "away_team": {
                        "abbr": away_team
                    },
                    "prediction": {
                        # Explicitly cast to native Python float to avoid float32 JSON errors
                        "model_home_win_prob": round(float(model_home_win_prob), 3),
                        "vegas_home_implied_prob": round(float(vegas_implied_prob), 3),
                        "edge": round(float(edge), 3),
                        "pick": str(pick),
                        "confidence_tier": "High" if is_high_confidence else "Standard"
                    }
                }
            ]
        }

        with open(json_path, 'w') as f:
            json.dump(json_payload, f, indent=2)

        print(f"✅ Exported single-game JSON to {os.path.abspath(json_path)}")
        
        if edge > 0:
            print(f"✅ The model likes the {home_team} (Home) more than Vegas does.")
        else:
            print(f"✅ The model likes the {away_team} (Away) more than Vegas does.")
    else:
        print("Error: Could not find rolling stats for one of the teams.")

        

