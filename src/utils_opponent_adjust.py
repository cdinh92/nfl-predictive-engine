import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV

def compute_adjusted_epa(pbp_df):
    """
    Computes opponent-adjusted offensive and defensive EPA per play 
    using Ridge Regression across historical game data.
    """
    # Filter valid offensive plays
    plays = pbp_df[pbp_df['posteam'].notna() & pbp_df['defteam'].notna() & ((pbp_df['pass'] == 1) | (pbp['rush'] == 1))].copy()
    
    # Aggregate actual EPA per game for each matchup
    game_plays = plays.groupby(['game_id', 'posteam', 'defteam', 'home_team'], as_index=False).agg(
        sepa=('epa', 'mean'),
        play_count=('epa', 'count')
    )
    
    game_plays['hfa'] = (game_plays['posteam'] == game_plays['home_team']).astype(int)
    
    # Prepare Design Matrix (X) using One-Hot Encoding for Offense and Defense teams
    offense_dummies = pd.get_dummies(game_plays['posteam'], prefix='offense')
    defense_dummies = pd.get_dummies(game_plays['defteam'], prefix='defense')
    
    X = pd.concat([offense_dummies, defense_dummies, game_plays[['hfa']]], axis=1)
    y = game_plays['sepa']
    
    # Fit Ridge Regression to isolate team coefficients from schedule strength
    alphas = np.logspace(-1, 3, 50)
    reg = RidgeCV(alphas=alphas, fit_intercept=False)
    reg.fit(X, y)
    
    # Extract coefficients into clean dataframes
    coefs = pd.DataFrame({'coef_name': X.columns.values, 'val': reg.coef_})
    
    off_coefs = coefs[coefs['coef_name'].str.startswith('offense_')].copy()
    off_coefs['team'] = off_coefs['coef_name'].str.replace('offense_', '')
    off_coefs['adj_off_epa'] = off_coefs['val']
    
    def_coefs = coefs[coefs['coef_name'].str.startswith('defense_')].copy()
    def_coefs['team'] = def_coefs['coef_name'].str.replace('defense_', '')
    # Multiply by -1 so positive values indicate strong defense (suppressing opponent EPA)
    def_coefs['adj_def_epa'] = def_coefs['val'] * -1 
    
    adjusted_team_metrics = pd.merge(
        off_coefs[['team', 'adj_off_epa']], 
        def_coefs[['team', 'adj_def_epa']], 
        on='team'
    )
    
    return adjusted_team_metrics