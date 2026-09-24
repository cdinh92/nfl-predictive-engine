from datetime import datetime, timezone
import json
import os
import nflreadpy as nfl
import pandas as pd
import joblib
from scipy.stats import norm

TEAM_NAMES = {
    'ARI': 'Cardinals', 'ATL': 'Falcons', 'BAL': 'Ravens', 'BUF': 'Bills',
    'CAR': 'Panthers', 'CHI': 'Bears', 'CIN': 'Bengals', 'CLE': 'Browns',
    'DAL': 'Cowboys', 'DEN': 'Broncos', 'DET': 'Lions', 'GB': 'Packers',
    'HOU': 'Texans', 'IND': 'Colts', 'JAX': 'Jaguars', 'KC': 'Chiefs',
    'LAC': 'Chargers', 'LAR': 'Rams', 'LA': 'Rams', 'LV': 'Raiders',
    'MIA': 'Dolphins', 'MIN': 'Vikings', 'NE': 'Patriots', 'NO': 'Saints',
    'NYG': 'Giants', 'NYJ': 'Jets', 'PHI': 'Eagles', 'PIT': 'Steelers',
    'SEA': 'Seahawks', 'SF': '49ers', 'TB': 'Buccaneers', 'TEN': 'Titans',
    'WAS': 'Commanders',
}

def get_full_name(abbr):
    return TEAM_NAMES.get(abbr, abbr)

def format_gametime(gametime_str):
    if not gametime_str or gametime_str == 'TBD' or pd.isna(gametime_str):
        return 'TBD'
    try:
        parts = str(gametime_str).strip()[:5].split(':')
        hour = int(parts[0])
        minute = int(parts[1])
        # Convert Eastern to Mountain (-2 hours offset)
        hour_mt = (hour - 2) % 24
        dt_temp = datetime.strptime(f"{hour_mt:02d}:{minute:02d}", '%H:%M')
        return dt_temp.strftime('%I:%M %p').lstrip('0')
    except ValueError:
        return str(gametime_str)

def get_team_latest_stats(df, team):
    team_games = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
    if team_games.empty:
        return None
        
    last_game = team_games.iloc[-1]
    is_home = last_game['home_team'] == team
    prefix = 'home_' if is_home else 'away_'
    
    last_4 = team_games.tail(4)
    pts_scored, pts_allowed = [], []
    for _, game in last_4.iterrows():
        if game['home_team'] == team:
            pts_scored.append(game['home_score'])
            pts_allowed.append(game['away_score'])
        else:
            pts_scored.append(game['away_score'])
            pts_allowed.append(game['home_score'])
            
    return {
        'roll_pts_scored': sum(pts_scored) / len(pts_scored) if pts_scored else 0,
        'roll_pts_allowed': sum(pts_allowed) / len(pts_allowed) if pts_allowed else 0,
        'roll_passing_yards': last_game.get(f'{prefix}roll_passing_yards', 0.0),
        'roll_rushing_yards': last_game.get(f'{prefix}roll_rushing_yards', 0.0),
        'roll_completion_pct': last_game.get(f'{prefix}roll_completion_pct', 0.0),
        'roll_net_yds_per_play': last_game.get(f'{prefix}roll_net_yds_per_play', 0.0),
        'roll_off_epa': last_game.get(f'{prefix}roll_off_epa', 0.0),
        'roll_def_epa': last_game.get(f'{prefix}roll_def_epa', 0.0),
        'roll_off_cpoe': last_game.get(f'{prefix}roll_off_cpoe', 0.0),
    }

def american_odds_to_raw_prob(ml):
    if pd.isna(ml):
        return 0.50
    ml = float(ml)
    return (-ml) / (-ml + 100) if ml < 0 else 100 / (ml + 100)

if __name__ == '__main__':
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == 'src' else SCRIPT_DIR
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

    # Target Configuration
    TARGET_SEASON = 2026
    TARGET_WEEK = 3

    # Folders
    PREDICTIONS_BASE = os.path.join(PROJECT_ROOT, 'predictions')
    WEEK_FOLDER_NAME = f'week_{TARGET_WEEK}'
    PREDICTIONS_DIR = os.path.join(PREDICTIONS_BASE, WEEK_FOLDER_NAME)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    current_time_utc = datetime.now(timezone.utc)
    formatted_timestamp = current_time_utc.strftime('%I:%M %p (Mountain Time), %Y-%m-%d')

    print(f'=========================================================================================================')
    print(f' NFL PREDICTIVE ENGINE | RUN TIMESTAMP: {formatted_timestamp}')
    print(f'=========================================================================================================')
    
    # 1. Load the Schedule
    print(f'Pulling {TARGET_SEASON} NFL Schedule...')
    schedule = nfl.load_schedules([TARGET_SEASON]).to_pandas()
    target_games = schedule[schedule['week'] == TARGET_WEEK].copy()
    
    if target_games.empty:
        print('No games found for the selected week.')
        exit()

    if 'gameday' in target_games.columns:
        target_games['gameday'] = pd.to_datetime(target_games['gameday'])
        target_games = target_games.sort_values(by=['gameday', 'gametime'])

    # 2. Load Historical Data for Feature Calculation
    features_df = pd.read_csv(os.path.join(DATA_DIR, 'model_features.csv'))
    
    # 3. Load the Serialized XGBRegressor Model
    model_path = os.path.join(MODEL_DIR, 'xgb_margin_model.joblib')
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}. Run 03_model_training.py first.")
        exit()
    model = joblib.load(model_path)
    
    # 4. Load Live Odds
    odds_path = os.path.join(DATA_DIR, 'upcoming_odds.csv')
    if os.path.exists(odds_path):
        live_odds = pd.read_csv(odds_path)
    else:
        print("⚠️ Warning: upcoming_odds.csv not found. Reverting to schedule defaults.")
        live_odds = pd.DataFrame()

    batch_payload = {
        'last_updated': current_time_utc.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'season': TARGET_SEASON,
        'week': TARGET_WEEK,
        'games_by_date': {},
    }

    print('---------------------------------------------------------------------------------------------------------')

    processed_games_by_date = {}

    for _, game in target_games.iterrows():
        home_team_abbr = game['home_team']
        away_team_abbr = game['away_team']
        game_id = game['game_id']

        gameday = str(game.get('gameday', 'TBD')).split('T')[0] if pd.notna(game.get('gameday')) else 'TBD'
        formatted_time = format_gametime(game.get('gametime', 'TBD'))

        # Fetch Live Odds if available
        home_ml, away_ml = -110, -110 
        if not live_odds.empty:
            match_odds = live_odds[(live_odds['home_team'] == home_team_abbr) & (live_odds['away_team'] == away_team_abbr)]
            if not match_odds.empty:
                home_ml = match_odds.iloc[0]['home_moneyline']
                away_ml = match_odds.iloc[0]['away_moneyline']

        # Remove Bookmaker Vig to get true implied probability
        home_raw = american_odds_to_raw_prob(home_ml)
        away_raw = american_odds_to_raw_prob(away_ml)
        vegas_implied_prob = home_raw / (home_raw + away_raw) if (home_raw + away_raw) > 0 else 0.50

        # Calculate True Rolling Stats (No Lag)
        home_stats = get_team_latest_stats(features_df, home_team_abbr)
        away_stats = get_team_latest_stats(features_df, away_team_abbr)

        if not home_stats or not away_stats:
            continue

        # 1. Provide only the 9 stats to the model
        matchup_data = pd.DataFrame([{
            'diff_pts_scored': home_stats['roll_pts_scored'] - away_stats['roll_pts_scored'],
            'diff_pts_allowed': away_stats['roll_pts_allowed'] - home_stats['roll_pts_allowed'],
            'diff_pass_yds': home_stats['roll_passing_yards'] - away_stats['roll_passing_yards'],
            'diff_rush_yds': home_stats['roll_rushing_yards'] - away_stats['roll_rushing_yards'],
            'diff_comp_pct': home_stats['roll_completion_pct'] - away_stats['roll_completion_pct'],
            'diff_net_yds_play': home_stats['roll_net_yds_per_play'] - away_stats['roll_net_yds_per_play'],
            'diff_off_epa': home_stats['roll_off_epa'] - away_stats['roll_off_epa'],
            'diff_def_epa': away_stats['roll_def_epa'] - home_stats['roll_def_epa'],
            'diff_off_cpoe': home_stats['roll_off_cpoe'] - away_stats['roll_off_cpoe']
        }])

        # 2. Get the pure statistical margin and probability
        predicted_margin = float(model.predict(matchup_data)[0])
        STD_DEV = 13.5
        stats_home_win_prob = float(norm.cdf(predicted_margin / STD_DEV))

        # 3. Enforce the 80/20 Ensemble Split
        model_home_win_prob = (0.80 * stats_home_win_prob) + (0.20 * vegas_implied_prob)

        # Compute Uncompressed Edge
        edge = model_home_win_prob - vegas_implied_prob
        pick_abbr = home_team_abbr if model_home_win_prob > 0.5 else away_team_abbr

        model_picks_home = model_home_win_prob > 0.5
        vegas_picks_home = vegas_implied_prob > 0.5
        same_winner = model_picks_home == vegas_picks_home

        model_winner_prob = model_home_win_prob if model_picks_home else (1.0 - model_home_win_prob)
        vegas_winner_prob = vegas_implied_prob if vegas_picks_home else (1.0 - vegas_implied_prob)

        # The projected margin is no longer a rough estimate, it's the actual model output
        projected_margin = round(abs(predicted_margin), 1)

        if same_winner and (model_winner_prob > 0.70) and (vegas_winner_prob > 0.65):
            confidence_tier = 'High'
        elif same_winner and (0.60 <= model_winner_prob) and (0.55 <= vegas_winner_prob):
            confidence_tier = 'Moderate'
        else:
            confidence_tier = 'Low'

        home_full = get_full_name(home_team_abbr)
        away_full = get_full_name(away_team_abbr)
        pick_full = get_full_name(pick_abbr)

        game_info = {
            'game_id': game_id,
            'gametime': formatted_time,
            'home_team': {'abbr': home_team_abbr, 'name': home_full, 'moneyline': home_ml},
            'away_team': {'abbr': away_team_abbr, 'name': away_full, 'moneyline': away_ml},
            'prediction': {
                'model_home_win_prob': round(model_home_win_prob, 3),
                'vegas_home_implied_prob': round(vegas_implied_prob, 3),
                'edge': round(edge, 3),
                'pick_abbr': str(pick_abbr),
                'pick_name': str(pick_full),
                'projected_margin': projected_margin,
                'confidence_tier': confidence_tier,
            },
        }

        if gameday not in processed_games_by_date:
            processed_games_by_date[gameday] = []
        processed_games_by_date[gameday].append(game_info)

    for gameday, games in sorted(processed_games_by_date.items()):
        print(f'\nDATE: {gameday}')
        batch_payload['games_by_date'][gameday] = games
        for g in games:
            away = g['away_team']['name']
            home = g['home_team']['name']
            p = g['prediction']
            pick_display = f"{p['pick_name']} (by {p['projected_margin']} pts)"
            print(f'  [{g["gametime"]:<10}] {away:>11} @ {home:<11} | Model: {p["model_home_win_prob"]*100:>4.1f}% | Vegas: {p["vegas_home_implied_prob"]*100:>4.1f}% | Edge: {p["edge"]*100:>+5.1f}% | Pick: {pick_display:<25} | Tier: {p["confidence_tier"]:<8}')

    print('\n========================================================================================================_')

    output_filename = f'predictions_{TARGET_SEASON}_week_{TARGET_WEEK}.json'
    json_path = os.path.join(PREDICTIONS_DIR, output_filename)
    with open(json_path, 'w') as f:
        json.dump(batch_payload, f, indent=2, default=int)

    print(f'Exported weekly JSON slate to {os.path.abspath(json_path)}')