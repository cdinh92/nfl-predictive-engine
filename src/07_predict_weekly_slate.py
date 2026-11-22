from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import json
import os
import requests
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

ODDS_API_TO_ABBR = {
    'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
    'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
    'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
    'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
    'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
    'Los Angeles Rams': 'LA', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
    'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
    'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
    'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
    'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS',
}


def get_full_name(abbr):
    return TEAM_NAMES.get(abbr, abbr)


def format_gametime_utc(gameday_val, gametime_str):
    if not gametime_str or gametime_str == 'TBD' or pd.isna(gametime_str) or pd.isna(gameday_val):
        return 'TBD'
    try:
        date_str = pd.to_datetime(gameday_val).strftime('%Y-%m-%d')
        time_str = str(gametime_str).strip()[:5]

        dt_str = f"{date_str} {time_str}"
        et_zone = ZoneInfo("America/New_York")
        local_dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M").replace(tzinfo=et_zone)

        return local_dt.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    except Exception:
        return str(gametime_str)


def fetch_betmgm_odds():
    """Pulls live BetMGM spreads, totals, and moneylines from The Odds API."""
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        print("⚠️  ODDS_API_KEY not found in environment; falling back to nflreadpy schedule lines.")
        return {}

    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "spreads,totals,h2h",
        "bookmakers": "betmgm",
        "oddsFormat": "american",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        fetched_at = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    except Exception as e:
        print(f"⚠️  Failed to fetch BetMGM lines ({e}); falling back to schedule lines.")
        return {}

    odds_map = {}
    for g in resp.json():
        home_abbr = ODDS_API_TO_ABBR.get(g["home_team"], g["home_team"])
        away_abbr = ODDS_API_TO_ABBR.get(g["away_team"], g["away_team"])
        # Normalize LAR -> LA to match nflreadpy
        if home_abbr == 'LAR':
            home_abbr = 'LA'
        if away_abbr == 'LAR':
            away_abbr = 'LA'

        entry = {"odds_fetched_at": fetched_at, "sportsbook": "betmgm"}
        for book in g.get("bookmakers", []):
            if book["key"] != "betmgm":
                continue
            for m in book.get("markets", []):
                outcomes = {o["name"]: o for o in m["outcomes"]}
                if m["key"] == "spreads":
                    h_o = outcomes.get(g["home_team"], {})
                    a_o = outcomes.get(g["away_team"], {})
                    entry["home_spread"] = h_o.get("point")
                    entry["home_spread_odds"] = h_o.get("price", -110)
                    entry["away_spread"] = a_o.get("point")
                    entry["away_spread_odds"] = a_o.get("price", -110)
                elif m["key"] == "totals":
                    over_o = outcomes.get("Over", {})
                    under_o = outcomes.get("Under", {})
                    entry["total_line"] = over_o.get("point")
                    entry["over_odds"] = over_o.get("price", -110)
                    entry["under_odds"] = under_o.get("price", -110)
                elif m["key"] == "h2h":
                    entry["home_ml"] = outcomes.get(g["home_team"], {}).get("price")
                    entry["away_ml"] = outcomes.get(g["away_team"], {}).get("price")

        odds_map[(away_abbr, home_abbr)] = entry

    print(f"Pulled live BetMGM odds for {len(odds_map)} matchups.")
    return odds_map


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
        'roll_adj_off_epa': last_game.get(f'{prefix}adj_off_epa', 0.0),
    }


def american_odds_to_raw_prob(ml):
    if pd.isna(ml) or ml is None:
        return 0.50
    ml = float(ml)
    return (-ml) / (-ml + 100) if ml < 0 else 100 / (ml + 100)


if __name__ == '__main__':
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == 'src' else SCRIPT_DIR
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

    TARGET_SEASON = 2026
    TARGET_WEEK = 11

    PREDICTIONS_BASE = os.path.join(PROJECT_ROOT, 'predictions')
    WEEK_FOLDER_NAME = f'week_{TARGET_WEEK}'
    PREDICTIONS_DIR = os.path.join(PREDICTIONS_BASE, WEEK_FOLDER_NAME)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    current_time_utc = datetime.now(timezone.utc)
    mt_time = current_time_utc.astimezone(ZoneInfo("America/Denver"))
    formatted_timestamp = mt_time.strftime('%I:%M %p (Mountain Time), %Y-%m-%d')

    print('=========================================================================================================')
    print(f' NFL PREDICTIVE ENGINE | RUN TIMESTAMP: {formatted_timestamp}')
    print('=========================================================================================================')

    print(f'Pulling {TARGET_SEASON} NFL Schedule...')
    schedule = nfl.load_schedules([TARGET_SEASON]).to_pandas()
    target_games = schedule[schedule['week'] == TARGET_WEEK].copy()

    if target_games.empty:
        print('No games found for the selected week.')
        exit()

    if 'gameday' in target_games.columns:
        target_games['gameday'] = pd.to_datetime(target_games['gameday'])
        target_games = target_games.sort_values(by=['gameday', 'gametime'])

    features_df = pd.read_csv(os.path.join(DATA_DIR, 'model_features.csv'))

    # Load Models & Unpack Variance Scale Factor
    margin_model_path = os.path.join(MODEL_DIR, 'xgb_margin_model.joblib')
    total_model_path = os.path.join(MODEL_DIR, 'xgb_total_points_model.joblib')

    if not os.path.exists(margin_model_path) or not os.path.exists(total_model_path):
        print(f"❌ Error: Required model artifacts missing in {MODEL_DIR}.")
        exit()

    loaded_margin_artifact = joblib.load(margin_model_path)
    if isinstance(loaded_margin_artifact, dict):
        margin_model = loaded_margin_artifact['model']
        scale_factor = float(loaded_margin_artifact.get('scale_factor', 1.0))
        STD_DEV = float(loaded_margin_artifact.get('game_margin_std', 13.5))
    else:
        margin_model = loaded_margin_artifact
        scale_factor = 1.0
        STD_DEV = 13.5

    total_points_model = joblib.load(total_model_path)
    print(f"Loaded Margin Model | Variance Scale Factor: {scale_factor:.4f}x | Margin StdDev: {STD_DEV:.2f}")

    # Pull live BetMGM lines
    betmgm_odds_map = fetch_betmgm_odds()

    batch_payload = {
        'last_updated': current_time_utc.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'season': TARGET_SEASON,
        'week': TARGET_WEEK,
        'scale_factor': scale_factor,
        'games_by_date': {},
    }

    print('---------------------------------------------------------------------------------------------------------')
    processed_games_by_date = {}

    for _, game in target_games.iterrows():
        home_team_abbr = game['home_team']
        away_team_abbr = game['away_team']
        game_id = game['game_id']

        gameday = pd.to_datetime(game['gameday']).strftime('%Y-%m-%d') if pd.notna(game.get('gameday')) else 'TBD'
        formatted_time = format_gametime_utc(gameday, game.get('gametime', 'TBD'))

        # Lookup BetMGM lines first, fallback to nflreadpy schedule lines
        lookup_home = 'LA' if home_team_abbr == 'LAR' else home_team_abbr
        lookup_away = 'LA' if away_team_abbr == 'LAR' else away_team_abbr
        mgm = betmgm_odds_map.get((lookup_away, lookup_home), {})

        home_ml = mgm.get('home_ml') if mgm.get('home_ml') is not None else game.get('home_moneyline')
        away_ml = mgm.get('away_ml') if mgm.get('away_ml') is not None else game.get('away_moneyline')
        if pd.isna(home_ml) or home_ml is None or pd.isna(away_ml) or away_ml is None:
            home_ml, away_ml = -110, -110

        # nflreadpy spread_line convention: positive = home favored; BetMGM home_spread: negative = home favored
        sched_spread = game.get('spread_line')
        fallback_home_spread = -float(sched_spread) if pd.notna(sched_spread) else None
        home_spread = mgm.get('home_spread') if mgm.get('home_spread') is not None else fallback_home_spread
        away_spread = mgm.get('away_spread') if mgm.get('away_spread') is not None else (
            -home_spread if home_spread is not None else None
        )
        home_spread_odds = mgm.get('home_spread_odds', -110)
        away_spread_odds = mgm.get('away_spread_odds', -110)

        fallback_total = float(game['total_line']) if pd.notna(game.get('total_line')) else None
        total_line = mgm.get('total_line') if mgm.get('total_line') is not None else fallback_total
        over_odds = mgm.get('over_odds', -110)
        under_odds = mgm.get('under_odds', -110)

        home_raw = american_odds_to_raw_prob(home_ml)
        away_raw = american_odds_to_raw_prob(away_ml)
        vegas_implied_prob = home_raw / (home_raw + away_raw) if (home_raw + away_raw) > 0 else 0.50

        home_stats = get_team_latest_stats(features_df, home_team_abbr)
        away_stats = get_team_latest_stats(features_df, away_team_abbr)

        if not home_stats or not away_stats:
            continue

        # 1. 10-Feature Matrix for Margin Model
        margin_matchup_data = pd.DataFrame([{
            'diff_pts_scored': home_stats['roll_pts_scored'] - away_stats['roll_pts_scored'],
            'diff_pts_allowed': away_stats['roll_pts_allowed'] - home_stats['roll_pts_allowed'],
            'diff_pass_yds': home_stats['roll_passing_yards'] - away_stats['roll_passing_yards'],
            'diff_rush_yds': home_stats['roll_rushing_yards'] - away_stats['roll_rushing_yards'],
            'diff_comp_pct': home_stats['roll_completion_pct'] - away_stats['roll_completion_pct'],
            'diff_net_yds_play': home_stats['roll_net_yds_per_play'] - away_stats['roll_net_yds_per_play'],
            'diff_off_epa': home_stats['roll_off_epa'] - away_stats['roll_off_epa'],
            'diff_def_epa': home_stats['roll_def_epa'] - away_stats['roll_def_epa'],
            'diff_off_cpoe': home_stats['roll_off_cpoe'] - away_stats['roll_off_cpoe'],
            'diff_adj_off_epa': home_stats['roll_adj_off_epa'] - away_stats['roll_adj_off_epa']
        }])

        # 2. 20-Feature Matrix for Total Points Model
        total_matchup_data = pd.DataFrame([{
            'sum_pts_scored': home_stats['roll_pts_scored'] + away_stats['roll_pts_scored'],
            'sum_pts_allowed': home_stats['roll_pts_allowed'] + away_stats['roll_pts_allowed'],
            'sum_pass_yds': home_stats['roll_passing_yards'] + away_stats['roll_passing_yards'],
            'sum_rush_yds': home_stats['roll_rushing_yards'] + away_stats['roll_rushing_yards'],
            'sum_net_yds_play': home_stats['roll_net_yds_per_play'] + away_stats['roll_net_yds_per_play'],
            'sum_off_epa': home_stats['roll_off_epa'] + away_stats['roll_off_epa'],
            'sum_def_epa': home_stats['roll_def_epa'] + away_stats['roll_def_epa'],
            'diff_pts_scored': home_stats['roll_pts_scored'] - away_stats['roll_pts_scored'],
            'diff_pts_allowed': away_stats['roll_pts_allowed'] - home_stats['roll_pts_allowed'],
            'diff_pass_yds': home_stats['roll_passing_yards'] - away_stats['roll_passing_yards'],
            'diff_rush_yds': home_stats['roll_rushing_yards'] - away_stats['roll_rushing_yards'],
            'diff_comp_pct': home_stats['roll_completion_pct'] - away_stats['roll_completion_pct'],
            'diff_net_yds_play': home_stats['roll_net_yds_per_play'] - away_stats['roll_net_yds_per_play'],
            'diff_off_epa': home_stats['roll_off_epa'] - away_stats['roll_off_epa'],
            'diff_def_epa': home_stats['roll_def_epa'] - away_stats['roll_def_epa'],
            'diff_off_cpoe': home_stats['roll_off_cpoe'] - away_stats['roll_off_cpoe'],
            'diff_adj_off_epa': home_stats['roll_adj_off_epa'] - away_stats['roll_adj_off_epa'],
            'rest_diff': game.get('rest_diff', 0),
            'wind_speed': game.get('wind_speed', 0.0),
            'temperature': game.get('temperature', 70.0)
        }])

        # Predict Raw & Variance-Scaled Margins (positive = home favored)
        raw_home_margin = float(margin_model.predict(margin_matchup_data)[0])
        calibrated_home_margin = raw_home_margin * scale_factor

        stats_home_win_prob = float(norm.cdf(calibrated_home_margin / STD_DEV))
        model_home_win_prob = (0.80 * stats_home_win_prob) + (0.20 * vegas_implied_prob)

        edge = model_home_win_prob - vegas_implied_prob
        pick_abbr = home_team_abbr if model_home_win_prob > 0.5 else away_team_abbr

        model_picks_home = model_home_win_prob > 0.5
        vegas_picks_home = vegas_implied_prob > 0.5
        same_winner = model_picks_home == vegas_picks_home

        model_winner_prob = model_home_win_prob if model_picks_home else (1.0 - model_home_win_prob)
        vegas_winner_prob = vegas_implied_prob if vegas_picks_home else (1.0 - vegas_implied_prob)
        projected_margin = round(abs(calibrated_home_margin), 1)

        if same_winner and (model_winner_prob > 0.70) and (vegas_winner_prob > 0.65):
            confidence_tier = 'High'
        elif same_winner and (0.60 <= model_winner_prob) and (0.55 <= vegas_winner_prob):
            confidence_tier = 'Moderate'
        else:
            confidence_tier = 'Low'

        # Total Points & Reconstructed Team Scores
        projected_total = float(total_points_model.predict(total_matchup_data)[0])
        home_projected_score = round((projected_total + calibrated_home_margin) / 2.0, 2)
        away_projected_score = round((projected_total - calibrated_home_margin) / 2.0, 2)

        # ATS & Over/Under Edges vs BetMGM
        if home_spread is not None:
            vegas_home_margin = -float(home_spread)
            ats_edge = round(calibrated_home_margin - vegas_home_margin, 2)
            spread_pick_side = 'home' if ats_edge > 0 else 'away'
            spread_pick_abbr = home_team_abbr if ats_edge > 0 else away_team_abbr
        else:
            ats_edge = None
            spread_pick_side = None
            spread_pick_abbr = None

        if total_line is not None:
            total_edge = round(projected_total - float(total_line), 2)
            total_pick_side = 'over' if total_edge > 0 else 'under'
        else:
            total_edge = None
            total_pick_side = None

        ml_pick_side = 'home' if calibrated_home_margin > 0 else 'away'

        home_full = get_full_name(home_team_abbr)
        away_full = get_full_name(away_team_abbr)
        pick_full = get_full_name(pick_abbr)

        game_info = {
            'game_id': game_id,
            'gametime': formatted_time,
            'home_team': {'abbr': home_team_abbr, 'name': home_full, 'moneyline': int(home_ml)},
            'away_team': {'abbr': away_team_abbr, 'name': away_full, 'moneyline': int(away_ml)},
            'betmgm_lines': {
                'home_spread': home_spread,
                'home_spread_odds': int(home_spread_odds),
                'away_spread': away_spread,
                'away_spread_odds': int(away_spread_odds),
                'total_line': total_line,
                'over_odds': int(over_odds),
                'under_odds': int(under_odds),
                'home_ml': int(home_ml),
                'away_ml': int(away_ml),
                'odds_fetched_at': mgm.get('odds_fetched_at'),
            },
            'prediction': {
                'model_home_win_prob': round(model_home_win_prob, 3),
                'vegas_home_implied_prob': round(vegas_implied_prob, 3),
                'edge': round(edge, 3),
                'pick_abbr': str(pick_abbr),
                'pick_name': str(pick_full),
                'raw_home_margin': round(raw_home_margin, 2),
                'calibrated_home_margin': round(calibrated_home_margin, 2),
                'projected_margin': projected_margin,
                'projected_total_points': round(projected_total, 2),
                'home_projected_score': home_projected_score,
                'away_projected_score': away_projected_score,
                'ats_edge': ats_edge,
                'spread_pick_side': spread_pick_side,
                'spread_pick_abbr': spread_pick_abbr,
                'total_edge': total_edge,
                'total_pick_side': total_pick_side,
                'ml_pick_side': ml_pick_side,
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
            away = g['away_team']['abbr']
            home = g['home_team']['abbr']
            p = g['prediction']
            mgm_l = g['betmgm_lines']
            score_str = f"{away} {p['away_projected_score']:.1f} @ {p['home_projected_score']:.1f} {home}"
            if p['ats_edge'] is not None:
                picked_spread = (
                    mgm_l['home_spread'] if p['spread_pick_side'] == 'home' else mgm_l['away_spread']
                )
                ats_str = f"{p['spread_pick_abbr']} ({picked_spread:+.1f} / Edge {abs(p['ats_edge']):.1f})"
            else:
                ats_str = "N/A"
            print(
                f"  [{g['gametime'][-9:-1]:<10}] {score_str:<24} | "
                f"Tot: {p['projected_total_points']:<5.1f} (Line {mgm_l['total_line']}) | "
                f"ATS: {ats_str:<26} | ML: {p['pick_abbr']} ({p['confidence_tier']})"
            )

    print('\n=========================================================================================================')

    output_filename = f'predictions_{TARGET_SEASON}_week_{TARGET_WEEK}.json'
    json_path = os.path.join(PREDICTIONS_DIR, output_filename)
    with open(json_path, 'w') as f:
        json.dump(batch_payload, f, indent=2, default=int)

    print(f'Exported weekly JSON slate to {os.path.abspath(json_path)}')