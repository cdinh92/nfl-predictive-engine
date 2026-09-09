import nflreadpy as nfl
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

def fetch_historical_games(seasons):
    print(f"Downloading NFL schedules for {seasons}...")
    df = nfl.load_schedules(seasons).to_pandas()
    
    # --- UPDATED FILTER ---
    # We now include Regular Season (REG) and all Postseason rounds (WC, DIV, CON, SB)
    valid_game_types = ['REG', 'WC', 'DIV', 'CON', 'SB']
    df = df[(df['game_type'].isin(valid_game_types)) & (~df['home_score'].isna())].copy()
    
    print("Downloading Team Passing Stats...")
    # Pull team-level stats to get raw completions, attempts, and passing yards
    team_stats = nfl.load_team_stats(seasons).to_pandas()
    pass_cols = ['game_id', 'team', 'passing_yards', 'completions', 'attempts']
    team_stats = team_stats[pass_cols]
    
    # Merge Home Team Stats
    df = df.merge(
        team_stats.rename(columns={'team': 'home_team', 'passing_yards': 'home_pass_yds', 'completions': 'home_pass_cmp', 'attempts': 'home_pass_att'}),
        on=['game_id', 'home_team'], how='left'
    )
    
    # Merge Away Team Stats
    df = df.merge(
        team_stats.rename(columns={'team': 'away_team', 'passing_yards': 'away_pass_yds', 'completions': 'away_pass_cmp', 'attempts': 'away_pass_att'}),
        on=['game_id', 'away_team'], how='left'
    )
    
    columns_to_keep = [
        'game_id', 'season', 'week', 'gameday', 
        'home_team', 'away_team', 'home_score', 'away_score',
        'home_pass_yds', 'home_pass_cmp', 'home_pass_att',
        'away_pass_yds', 'away_pass_cmp', 'away_pass_att'
    ]
    
    df = df[columns_to_keep].dropna()
    df = df.sort_values(by=['season', 'week', 'gameday']).reset_index(drop=True)
    return df

if __name__ == "__main__":
    target_seasons = [2023, 2024, 2025]
    games_df = fetch_historical_games(target_seasons)
    
    print("\n--- Data Preview ---")
    print(games_df.head())
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "src" else SCRIPT_DIR
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    output_path = os.path.join(DATA_DIR, "raw_games.csv")
    games_df.to_csv(output_path, index=False)
    print(f"\n✅ Successfully saved {len(games_df)} games to {output_path}")