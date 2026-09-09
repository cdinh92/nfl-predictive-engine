import pandas as pd
import os

def calculate_rolling_features(df):
    print("Calculating time-shifted rolling averages...")
    
    # 1. Break the data into a "per-team" perspective
    home_df = df[['game_id', 'season', 'week', 'gameday', 'home_team', 'home_score', 'away_score']].copy()
    home_df = home_df.rename(columns={'home_team': 'team', 'home_score': 'pts_scored', 'away_score': 'pts_allowed'})
    home_df['is_home'] = 1

    away_df = df[['game_id', 'season', 'week', 'gameday', 'away_team', 'away_score', 'home_score']].copy()
    away_df = away_df.rename(columns={'away_team': 'team', 'away_score': 'pts_scored', 'home_score': 'pts_allowed'})
    away_df['is_home'] = 0

    # Stack them vertically and sort by time
    team_df = pd.concat([home_df, away_df]).sort_values(by=['team', 'season', 'week']).reset_index(drop=True)

    # 2. Calculate the shifted rolling averages (Last 4 games)
    def get_rolling(group):
        # .shift(1) is the magic key to prevent data leakage!
        group['roll_pts_scored'] = group['pts_scored'].shift(1).rolling(4, min_periods=1).mean()
        group['roll_pts_allowed'] = group['pts_allowed'].shift(1).rolling(4, min_periods=1).mean()
        return group

    team_df = team_df.groupby('team', group_keys=False).apply(get_rolling)

    # 3. Split back into Home and Away to attach back to the main schedule
    home_stats = team_df[team_df['is_home'] == 1][['game_id', 'roll_pts_scored', 'roll_pts_allowed']]
    home_stats = home_stats.rename(columns={
        'roll_pts_scored': 'home_roll_pts_scored', 
        'roll_pts_allowed': 'home_roll_pts_allowed'
    })

    away_stats = team_df[team_df['is_home'] == 0][['game_id', 'roll_pts_scored', 'roll_pts_allowed']]
    away_stats = away_stats.rename(columns={
        'roll_pts_scored': 'away_roll_pts_scored', 
        'roll_pts_allowed': 'away_roll_pts_allowed'
    })

    # 4. Merge back to the original schedule
    model_df = df.merge(home_stats, on='game_id').merge(away_stats, on='game_id')
    
    # Drop Week 1 games because they have no prior rolling history
    model_df = model_df.dropna().reset_index(drop=True)
    return model_df

if __name__ == "__main__":
    # Smart Pathing (bulletproof)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "src" else SCRIPT_DIR
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")

    # Load the raw data
    input_path = os.path.join(DATA_DIR, "raw_games.csv")
    df = pd.read_csv(input_path)
    
    # Process features
    features_df = calculate_rolling_features(df)
    
    print("\n--- Engineered Features Preview ---")
    print(features_df[['season', 'week', 'home_team', 'away_team', 'home_roll_pts_scored', 'away_roll_pts_scored']].head())
    
    # Save the machine learning ready dataset
    output_path = os.path.join(DATA_DIR, "model_features.csv")
    features_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved {len(features_df)} rows of training data to {output_path}")