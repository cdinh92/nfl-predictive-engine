import nflreadpy as nfl
import pandas as pd
import os
from dotenv import load_dotenv

# Load hidden API keys
load_dotenv()

def fetch_historical_games(seasons):
    print(f"Downloading NFL schedules for {seasons}...")
    
    # Pull the official schedule and results data (Returns a Polars DataFrame)
    df = nfl.load_schedules(seasons)
    
    # Instantly convert the Polars DataFrame back to a standard Pandas DataFrame
    df = df.to_pandas()
    
    # Now our standard Pandas logic will work perfectly!
    df = df[(df['game_type'] == 'REG') & (~df['home_score'].isna())].copy()
    
    # Keep only the columns we need for our model
    columns_to_keep = [
        'game_id', 'season', 'week', 'gameday', 
        'home_team', 'away_team', 'home_score', 'away_score'
    ]
    df = df[columns_to_keep]
    
    # Sort chronologically (Crucial for preventing data leakage!)
    df = df.sort_values(by=['season', 'week', 'gameday']).reset_index(drop=True)
    
    return df

'''
def fetch_historical_games(seasons):
    print(f"Downloading NFL schedules for {seasons}...")
    
    # Pull the official schedule and results data
    df = nfl.load_schedules(seasons)
    
    # Filter for Regular Season games that have actually been played
    df = df[(df['game_type'] == 'REG') & (~df['home_score'].isna())].copy()
    
    # Keep only the columns we need for our model
    columns_to_keep = [
        'game_id', 'season', 'week', 'gameday', 
        'home_team', 'away_team', 'home_score', 'away_score'
    ]
    df = df[columns_to_keep]
    
    # Sort chronologically (Crucial for preventing data leakage!)
    df = df.sort_values(by=['season', 'week', 'gameday']).reset_index(drop=True)
    
    return df
'''

if __name__ == "__main__":
    # Target the 2023, 2024, and 2025 seasons for our training baseline
    target_seasons = [2023, 2024, 2025]
    
    games_df = fetch_historical_games(target_seasons)

    print("\n--- Data Preview ---")
    print(games_df.head())
    
    # 1. Find exactly where this script lives
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Smart Project Root Detection
    # If the script is inside 'src', go up one level. Otherwise, stay here.
    if os.path.basename(SCRIPT_DIR) == "src":
        PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    else:
        PROJECT_ROOT = SCRIPT_DIR
    
    # 3. Create the data folder securely inside the project
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Save the raw data
    output_path = os.path.join(DATA_DIR, "raw_games.csv")
    games_df.to_csv(output_path, index=False)
    print(f"\n✅ Successfully saved {len(games_df)} games to {output_path}")

    '''
    print("\n--- Data Preview ---")
    print(games_df.head())
    
    # 1. Find exactly where this script lives on your Mac (.../src)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Go up one level to your main project folder (.../nfl-predictive-engine)
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    
    # 3. Create the exact, unbreakable path to the data folder
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    
    # Force Python to create it safely
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Save the raw data using the absolute path
    output_path = os.path.join(DATA_DIR, "raw_games.csv")
    games_df.to_csv(output_path, index=False)
    print(f"\n✅ Successfully saved {len(games_df)} games to {output_path}")
    '''