from datetime import datetime, timezone
import json
import os
import nflreadpy as nfl
import pandas as pd
import xgboost as xgb

# --- NFL TEAM CODE TO NICKNAME MAPPING ---
TEAM_NAMES = {
    'ARI': 'Cardinals',
    'ATL': 'Falcons',
    'BAL': 'Ravens',
    'BUF': 'Bills',
    'CAR': 'Panthers',
    'CHI': 'Bears',
    'CIN': 'Bengals',
    'CLE': 'Browns',
    'DAL': 'Cowboys',
    'DEN': 'Broncos',
    'DET': 'Lions',
    'GB': 'Packers',
    'HOU': 'Texans',
    'IND': 'Colts',
    'JAX': 'Jaguars',
    'KC': 'Chiefs',
    'LAC': 'Chargers',
    'LAR': 'Rams',
    'LA': 'Rams',
    'LV': 'Raiders',
    'MIA': 'Dolphins',
    'MIN': 'Vikings',
    'NE': 'Patriots',
    'NO': 'Saints',
    'NYG': 'Giants',
    'NYJ': 'Jets',
    'PHI': 'Eagles',
    'PIT': 'Steelers',
    'SEA': 'Seahawks',
    'SF': '49ers',
    'TB': 'Buccaneers',
    'TEN': 'Titans',
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


def get_team_latest_stats(features_df, team):
  team_games = features_df[
      (features_df['home_team'] == team) | (features_df['away_team'] == team)
  ]
  if team_games.empty:
    return None
  last_game = team_games.iloc[-1]

  if last_game['home_team'] == team:
    return {
        'roll_pts_scored': last_game['home_roll_pts_scored'],
        'roll_pts_allowed': last_game['home_roll_pts_allowed'],
        'roll_passing_yards': last_game['home_roll_passing_yards'],
        'roll_completion_pct': last_game['home_roll_completion_pct'],
        'roll_net_yds_per_play': last_game['home_roll_net_yds_per_play'],
    }
  else:
    return {
        'roll_pts_scored': last_game['away_roll_pts_scored'],
        'roll_pts_allowed': last_game['away_roll_pts_allowed'],
        'roll_passing_yards': last_game['away_roll_passing_yards'],
        'roll_completion_pct': last_game['away_roll_completion_pct'],
        'roll_net_yds_per_play': last_game['away_roll_net_yds_per_play'],
    }


def american_odds_to_prob(ml):
  if pd.isna(ml):
    return 0.50
  return (-ml) / (-ml + 100) if ml < 0 else 100 / (ml + 100)


def estimate_win_margin(winner_prob):
  """Roughly maps win probability to an estimated point margin."""
  # Simple heuristic scale mapping probabilities from 0.5 to 1.0 to point spreads
  prob_diff = abs(winner_prob - 0.5) * 2  # ranges from 0.0 to 1.0
  # Say a 100% win probability model maps roughly to a 14-point blowout, 50% maps to 1 point
  margin = 1.0 + (prob_diff * 13.0)
  return round(margin, 1)


if __name__ == '__main__':
  SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
  PROJECT_ROOT = (
      os.path.dirname(SCRIPT_DIR)
      if os.path.basename(SCRIPT_DIR) == 'src'
      else SCRIPT_DIR
  )
  DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

  # Target Configuration (WEEKLY)
  TARGET_SEASON = 2026
  TARGET_WEEK = 2

  # --- DYNAMIC FOLDER STRUCTURE SETUP ---
  PREDICTIONS_BASE = os.path.join(PROJECT_ROOT, 'predictions')
  WEEK_FOLDER_NAME = f'week {TARGET_WEEK}'
  PREDICTIONS_DIR = os.path.join(PREDICTIONS_BASE, WEEK_FOLDER_NAME)
  os.makedirs(PREDICTIONS_DIR, exist_ok=True)
  # --------------------------------------

  current_time_utc = datetime.now(timezone.utc)
  formatted_timestamp = current_time_utc.strftime(
      '%I:%M %p (Mountain Time), %Y-%m-%d'
  )

  print(
      f'========================================================================================================='
  )
  print(
      f' NFL PREDICTIVE ENGINE | RUN TIMESTAMP: {formatted_timestamp}'
  )
  print(
      f'========================================================================================================='
  )
  print(f'Pulling {TARGET_SEASON} NFL Schedule...')
  schedule = nfl.load_schedules([TARGET_SEASON]).to_pandas()

  target_games = schedule[schedule['week'] == TARGET_WEEK].copy()
  print(
      f'Running WEEKLY mode for Week {TARGET_WEEK}. Found'
      f' {len(target_games)} games.'
  )

  if target_games.empty:
    print('No games found for the selected week.')
    exit()

  if 'gameday' in target_games.columns:
    target_games['gameday'] = pd.to_datetime(target_games['gameday'])
    target_games = target_games.sort_values(by=['gameday', 'gametime'])

  features_df = pd.read_csv(os.path.join(DATA_DIR, 'model_features.csv'))
  features = [
      'home_roll_pts_scored',
      'home_roll_pts_allowed',
      'away_roll_pts_scored',
      'away_roll_pts_allowed',
      'home_roll_passing_yards',
      'home_roll_completion_pct',
      'home_roll_net_yds_per_play',
      'away_roll_passing_yards',
      'away_roll_completion_pct',
      'away_roll_net_yds_per_play',
  ]
  features_df['home_win'] = (
      features_df['home_score'] > features_df['away_score']
  ).astype(int)

  model = xgb.XGBClassifier(
      learning_rate=0.01,
      max_depth=4,
      n_estimators=200,
      subsample=0.8,
      random_state=42,
  )
  model.fit(features_df[features], features_df['home_win'])

  batch_payload = {
      'last_updated': current_time_utc.strftime('%Y-%m-%dT%H:%M:%SZ'),
      'season': TARGET_SEASON,
      'week': TARGET_WEEK,
      'games_by_date': {},
  }

  print(
      '---------------------------------------------------------------------------------------------------------'
  )

  processed_games_by_date = {}

  for _, game in target_games.iterrows():
    home_team_abbr = game['home_team']
    away_team_abbr = game['away_team']
    game_id = game['game_id']
    vegas_home_ml = game.get('home_moneyline', -110)

    gameday = (
        str(game.get('gameday', 'TBD')).split('T')[0]
        if pd.notna(game.get('gameday'))
        else 'TBD'
    )
    raw_gametime = game.get('gametime', 'TBD')
    formatted_time = format_gametime(raw_gametime)

    home_stats = get_team_latest_stats(features_df, home_team_abbr)
    away_stats = get_team_latest_stats(features_df, away_team_abbr)

    if not home_stats or not away_stats:
      continue

    matchup_data = pd.DataFrame([{
        'home_roll_pts_scored': home_stats['roll_pts_scored'],
        'home_roll_pts_allowed': home_stats['roll_pts_allowed'],
        'away_roll_pts_scored': away_stats['roll_pts_scored'],
        'away_roll_pts_allowed': away_stats['roll_pts_allowed'],
        'home_roll_passing_yards': home_stats['roll_passing_yards'],
        'home_roll_completion_pct': home_stats['roll_completion_pct'],
        'home_roll_net_yds_per_play': home_stats['roll_net_yds_per_play'],
        'away_roll_passing_yards': away_stats['roll_passing_yards'],
        'away_roll_completion_pct': away_stats['roll_completion_pct'],
        'away_roll_net_yds_per_play': away_stats['roll_net_yds_per_play'],
    }])

    model_home_win_prob = float(
        model.predict_proba(matchup_data[features])[0][1]
    )
    vegas_implied_prob = float(american_odds_to_prob(vegas_home_ml))

    model_weight = 0.8
    vegas_weight = 1.0 - model_weight
    blended_home_win_prob = (model_weight * model_home_win_prob) + (
        vegas_weight * vegas_implied_prob
    )

    edge = blended_home_win_prob - vegas_implied_prob
    pick_abbr = (
        home_team_abbr if blended_home_win_prob > 0.5 else away_team_abbr
    )

    model_picks_home = blended_home_win_prob > 0.5
    vegas_picks_home = vegas_implied_prob > 0.5
    same_winner = model_picks_home == vegas_picks_home

    model_winner_prob = (
        blended_home_win_prob
        if model_picks_home
        else (1.0 - blended_home_win_prob)
    )
    vegas_winner_prob = (
        vegas_implied_prob if vegas_picks_home else (1.0 - vegas_implied_prob)
    )

    # Calculate win margin based on the blended winner probability
    projected_margin = estimate_win_margin(model_winner_prob)

    if (
        same_winner
        and (model_winner_prob > 0.70)
        and (vegas_winner_prob > 0.65)
    ):
      confidence_tier = 'High'
    elif (
        same_winner
        and (0.60 <= model_winner_prob)
        and (0.55 <= vegas_winner_prob)
    ):
      confidence_tier = 'Moderate'
    else:
      confidence_tier = 'Low'

    home_full = get_full_name(home_team_abbr)
    away_full = get_full_name(away_team_abbr)
    pick_full = get_full_name(pick_abbr)

    game_info = {
        'game_id': game_id,
        'gametime': formatted_time,
        'home_team': {
            'abbr': home_team_abbr,
            'name': home_full,
            'moneyline': vegas_home_ml,
        },
        'away_team': {'abbr': away_team_abbr, 'name': away_full},
        'prediction': {
            'model_home_win_prob': round(blended_home_win_prob, 3),
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
    print(
        f'  [{g["gametime"]:<10}] {away:>11} @ {home:<11} | Model Blend:'
        f' {p["model_home_win_prob"]*100:>4.1f}% | Vegas:'
        f' {p["vegas_home_implied_prob"]*100:>4.1f}% | Edge:'
        f' {p["edge"]*100:>+5.1f}% | Pick: {pick_display:<25} | Tier:'
        f' {p["confidence_tier"]:<8}'
    )

print(
    '\n========================================================================================================_'
)

output_filename = f'predictions_{TARGET_SEASON}_week_{TARGET_WEEK}.json'
json_path = os.path.join(PREDICTIONS_DIR, output_filename)
with open(json_path, 'w') as f:
  json.dump(batch_payload, f, indent=2)

print(f'Exported weekly JSON slate to {os.path.abspath(json_path)}')