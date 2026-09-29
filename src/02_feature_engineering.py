import os
import nflreadpy as nfl
import pandas as pd

def calculate_rolling_features(df):
    print("Calculating time-shifted exponential rolling averages for scoring, EPA, and CPOE...")

    # 1. Break data into per-team perspectives
    home_cols = [
        'game_id', 'season', 'week', 'gameday', 'home_team',
        'home_score', 'away_score',
        'home_off_epa_per_play', 'home_def_epa_per_play', 'home_off_cpoe'
    ]
    home_df = df[[c for c in home_cols if c in df.columns]].copy()
    home_df = home_df.rename(columns={
        'home_team': 'team',
        'home_score': 'pts_scored',
        'away_score': 'pts_allowed',
        'home_off_epa_per_play': 'off_epa',
        'home_def_epa_per_play': 'def_epa',
        'home_off_cpoe': 'off_cpoe'
    })
    home_df['is_home'] = 1

    away_cols = [
        'game_id', 'season', 'week', 'gameday', 'away_team',
        'away_score', 'home_score',
        'away_off_epa_per_play', 'away_def_epa_per_play', 'away_off_cpoe'
    ]
    away_df = df[[c for c in away_cols if c in df.columns]].copy()
    away_df = away_df.rename(columns={
        'away_team': 'team',
        'away_score': 'pts_scored',
        'home_score': 'pts_allowed',
        'away_off_epa_per_play': 'off_epa',
        'away_def_epa_per_play': 'def_epa',
        'away_off_cpoe': 'off_cpoe'
    })
    away_df['is_home'] = 0

    # Stack vertically and sort chronologically
    team_df = pd.concat([home_df, away_df]).sort_values(by=['team', 'season', 'week']).reset_index(drop=True)

    # 2. Compute Exponential Moving Average (EMA) shifted to prevent data leakage
    metrics_to_roll = ['pts_scored', 'pts_allowed', 'off_epa', 'def_epa', 'off_cpoe']
    SPAN_VAL = 4
    
    def get_rolling(group):
        for metric in metrics_to_roll:
            if metric in group.columns:
                # Shift by 1 so current game is never seen, then apply EMA
                group[f'roll_{metric}'] = group[metric].shift(1).ewm(span=SPAN_VAL, adjust=False).mean()
        return group

    team_df = team_df.groupby('team', group_keys=False).apply(get_rolling)

    # 3. Split back into home and away subsets
    stat_cols = [f'roll_{m}' for m in metrics_to_roll if f'roll_{m}' in team_df.columns]
    
    home_stats = team_df[team_df['is_home'] == 1][['game_id'] + stat_cols]
    home_stats = home_stats.rename(columns={c: f'home_{c}' for c in stat_cols})

    away_stats = team_df[team_df['is_home'] == 0][['game_id'] + stat_cols]
    away_stats = away_stats.rename(columns={c: f'away_{c}' for c in stat_cols})

    # 4. Merge back to the main schedule
    model_df = df.merge(home_stats, on='game_id').merge(away_stats, on='game_id')
    return model_df

def calculate_advanced_rolling_stats(seasons):
    print(f'Fetching team stats from nflreadpy for seasons: {seasons}...')
    team_stats = nfl.load_team_stats(seasons).to_pandas()

    team_stats['completion_pct'] = team_stats['completions'] / team_stats['attempts']
    team_stats['total_yards'] = team_stats['passing_yards'] + team_stats.get('rushing_yards', 0)
    team_stats['total_plays'] = team_stats['attempts'] + team_stats['sacks_suffered'] + team_stats.get('carries', 0)
    team_stats['net_yds_per_play'] = team_stats['total_yards'] / team_stats['total_plays']

    team_stats = team_stats.sort_values(by=['team', 'season', 'week'])

    # Apply EMA to advanced stats as well
    features_to_roll = ['passing_yards', 'rushing_yards', 'completion_pct', 'net_yds_per_play']
    SPAN_VAL = 4
    
    for feature in features_to_roll:
        roll_col_name = f'roll_{feature}'
        team_stats[roll_col_name] = (
            team_stats.groupby('team')[feature]
            .transform(lambda x: x.shift(1).ewm(span=SPAN_VAL, adjust=False).mean())
        )
        team_stats[roll_col_name] = team_stats[roll_col_name].fillna(0)

    print('✅ Advanced rushing and passing stats engineered with EMA.')
    return team_stats[['game_id', 'team', 'roll_passing_yards', 'roll_rushing_yards', 'roll_completion_pct', 'roll_net_yds_per_play']]

def merge_advanced_features(main_df, advanced_stats_df):
    home_merged = pd.merge(
        main_df,
        advanced_stats_df.rename(columns={
            'team': 'home_team',
            'roll_passing_yards': 'home_roll_passing_yards',
            'roll_rushing_yards': 'home_roll_rushing_yards',
            'roll_completion_pct': 'home_roll_completion_pct',
            'roll_net_yds_per_play': 'home_roll_net_yds_per_play',
        }),
        on=['game_id', 'home_team'],
        how='left'
    )

    fully_merged = pd.merge(
        home_merged,
        advanced_stats_df.rename(columns={
            'team': 'away_team',
            'roll_passing_yards': 'away_roll_passing_yards',
            'roll_rushing_yards': 'away_roll_rushing_yards',
            'roll_completion_pct': 'away_roll_completion_pct',
            'roll_net_yds_per_play': 'away_roll_net_yds_per_play',
        }),
        on=['game_id', 'away_team'],
        how='left'
    )

    fill_cols = [
        'home_roll_passing_yards', 'home_roll_rushing_yards', 'home_roll_completion_pct', 'home_roll_net_yds_per_play',
        'away_roll_passing_yards', 'away_roll_rushing_yards', 'away_roll_completion_pct', 'away_roll_net_yds_per_play',
        'home_roll_off_epa', 'home_roll_def_epa', 'home_roll_off_cpoe',
        'away_roll_off_epa', 'away_roll_def_epa', 'away_roll_off_cpoe'
    ]
    for col in fill_cols:
        if col in fully_merged.columns:
            fully_merged[col] = fully_merged[col].fillna(0)

    # --- CONDENSE TO 9 MATHEMATICAL DIFFERENTIALS ---
    fully_merged['diff_pts_scored'] = fully_merged['home_roll_pts_scored'] - fully_merged['away_roll_pts_scored']
    fully_merged['diff_pts_allowed'] = fully_merged['away_roll_pts_allowed'] - fully_merged['home_roll_pts_allowed']
    
    fully_merged['diff_pass_yds'] = fully_merged['home_roll_passing_yards'] - fully_merged['away_roll_passing_yards']
    fully_merged['diff_rush_yds'] = fully_merged['home_roll_rushing_yards'] - fully_merged['away_roll_rushing_yards']
    fully_merged['diff_comp_pct'] = fully_merged['home_roll_completion_pct'] - fully_merged['away_roll_completion_pct']
    fully_merged['diff_net_yds_play'] = fully_merged['home_roll_net_yds_per_play'] - fully_merged['away_roll_net_yds_per_play']
    
    fully_merged['diff_off_epa'] = fully_merged['home_roll_off_epa'] - fully_merged['away_roll_off_epa']
    fully_merged['diff_def_epa'] = fully_merged['away_roll_def_epa'] - fully_merged['home_roll_def_epa']
    fully_merged['diff_off_cpoe'] = fully_merged['home_roll_off_cpoe'] - fully_merged['away_roll_off_cpoe']

    print('✅ All rolling features successfully assembled into EMA differentials.')
    return fully_merged

if __name__ == '__main__':
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == 'src' else SCRIPT_DIR
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

    input_path = os.path.join(DATA_DIR, 'raw_games.csv')
    df = pd.read_csv(input_path)

    features_df = calculate_rolling_features(df)
    unique_seasons = df['season'].unique().tolist()
    advanced_stats_df = calculate_advanced_rolling_stats(seasons=unique_seasons)
    final_features_df = merge_advanced_features(features_df, advanced_stats_df)

    output_path = os.path.join(DATA_DIR, 'model_features.csv')
    final_features_df.to_csv(output_path, index=False)
    print(f'✅ Saved {len(final_features_df)} rows with EMA differential features to {output_path}')