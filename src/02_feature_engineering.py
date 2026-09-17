import pandas as pd
import os
import nflreadpy as nfl

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

def calculate_advanced_rolling_stats(seasons):
    """
    Pulls team-level stats to calculate 4-game rolling passing efficiency 
    and net yards per play (NY/P).
    """
    print(f"Fetching team stats from nflreadpy for seasons: {seasons}...")
    team_stats = nfl.load_team_stats(seasons).to_pandas()
    
    # 1. Feature Creation: Calculate raw game-level metrics
    team_stats['completion_pct'] = team_stats['completions'] / team_stats['attempts']
    team_stats['total_yards'] = team_stats['passing_yards'] + team_stats.get('rushing_yards', 0)
    team_stats['total_plays'] = team_stats['attempts'] + team_stats['sacks_suffered'] + team_stats.get('carries', 0)
    team_stats['net_yds_per_play'] = team_stats['total_yards'] / team_stats['total_plays']

    team_stats = team_stats.sort_values(by=['team', 'season', 'week'])

    # 2. Rolling Math: 4-Game Averages (using .shift(1) to avoid leakage)
    features_to_roll = ['passing_yards', 'completion_pct', 'net_yds_per_play']
    
    for feature in features_to_roll:
        roll_col_name = f"roll_{feature}"
        team_stats[roll_col_name] = team_stats.groupby('team')[feature].transform(
            lambda x: x.rolling(window=4, min_periods=1).mean().shift(1)
        )
        
    print("✅ Advanced rolling stats engineered.")
    return team_stats[['game_id', 'team', 'roll_passing_yards', 'roll_completion_pct', 'roll_net_yds_per_play']]

def merge_advanced_features(main_df, advanced_stats_df):
    """
    Merges home and away advanced rolling stats into the core feature dataframe.
    """
    # Merge for Home Team
    home_merged = pd.merge(
        main_df,
        advanced_stats_df.rename(columns={
            'team': 'home_team',
            'roll_passing_yards': 'home_roll_passing_yards',
            'roll_completion_pct': 'home_roll_completion_pct',
            'roll_net_yds_per_play': 'home_roll_net_yds_per_play'
        }),
        on=['game_id', 'home_team'],
        how='left'
    )
    
    # Merge for Away Team
    fully_merged = pd.merge(
        home_merged,
        advanced_stats_df.rename(columns={
            'team': 'away_team',
            'roll_passing_yards': 'away_roll_passing_yards',
            'roll_completion_pct': 'away_roll_completion_pct',
            'roll_net_yds_per_play': 'away_roll_net_yds_per_play'
        }),
        on=['game_id', 'away_team'],
        how='left'
    )
    
    # Fill missing early-season data with 0 to prevent NaN errors in XGBoost
    advanced_cols = [
        'home_roll_passing_yards', 'home_roll_completion_pct', 'home_roll_net_yds_per_play',
        'away_roll_passing_yards', 'away_roll_completion_pct', 'away_roll_net_yds_per_play'
    ]
    fully_merged[advanced_cols] = fully_merged[advanced_cols].fillna(0)
    
    print("✅ Advanced features successfully merged into model dataset.")
    return fully_merged

if __name__ == "__main__":
    # Smart Pathing (bulletproof)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "src" else SCRIPT_DIR
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")

    # Load the raw data
    input_path = os.path.join(DATA_DIR, "raw_games.csv")
    df = pd.read_csv(input_path)
    
    # Process basic point features
    features_df = calculate_rolling_features(df)

    # Process and merge advanced passing/efficiency features
    unique_seasons = df['season'].unique().tolist()
    advanced_stats_df = calculate_advanced_rolling_stats(seasons=unique_seasons)
    final_features_df = merge_advanced_features(features_df, advanced_stats_df)
    
    print("\n--- Engineered Features Preview ---")
    print(final_features_df[['season', 'week', 'home_team', 'away_team', 'home_roll_pts_scored', 'away_roll_pts_scored', 'home_roll_passing_yards', 'away_roll_passing_yards']].head())
    
    # Save the machine learning ready dataset
    output_path = os.path.join(DATA_DIR, "model_features.csv")
    final_features_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved {len(final_features_df)} rows of training data to {output_path}")