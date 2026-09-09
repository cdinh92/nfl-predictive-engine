import requests
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ODDS_API_KEY")

# 1. The Translation Dictionary
TEAM_MAP = {
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
    'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS'
}

def fetch_nfl_odds():
    print("Authenticating with The Odds API...")
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"
    params = {
        "api_key": API_KEY, "regions": "us", 
        "markets": "h2h,spreads", "oddsFormat": "american"
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"❌ API Error {response.status_code}: {response.text}")
        return None
    return response.json()

def parse_odds(odds_json):
    print("Parsing JSON and standardizing team names...")
    parsed_games = []
    
    for game in odds_json:
        # Standardize names using our map
        home_team = TEAM_MAP.get(game['home_team'], game['home_team'])
        away_team = TEAM_MAP.get(game['away_team'], game['away_team'])
        
        home_ml = None
        away_ml = None
        home_spread = None
        
        # Dig into the JSON to find DraftKings odds
        for bookie in game.get('bookmakers', []):
            if bookie['key'] == 'draftkings':
                for market in bookie['markets']:
                    if market['key'] == 'h2h':
                        for outcome in market['outcomes']:
                            if outcome['name'] == game['home_team']:
                                home_ml = outcome['price']
                            elif outcome['name'] == game['away_team']:
                                away_ml = outcome['price']
                    elif market['key'] == 'spreads':
                        for outcome in market['outcomes']:
                            if outcome['name'] == game['home_team']:
                                home_spread = outcome['point']
        
        parsed_games.append({
            'home_team': home_team,
            'away_team': away_team,
            'commence_time': game['commence_time'],
            'home_moneyline': home_ml,
            'away_moneyline': away_ml,
            'home_spread': home_spread
        })
        
    return pd.DataFrame(parsed_games)

if __name__ == "__main__":
    raw_odds = fetch_nfl_odds()
    
    if raw_odds:
        odds_df = parse_odds(raw_odds)
        
        print("\n--- Cleaned Odds Preview ---")
        print(odds_df.head())
        
        # Smart Pathing to save the data
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "src" else SCRIPT_DIR
        DATA_DIR = os.path.join(PROJECT_ROOT, "data")
        
        output_path = os.path.join(DATA_DIR, "upcoming_odds.csv")
        odds_df.to_csv(output_path, index=False)
        print(f"\n✅ Saved clean odds for {len(odds_df)} games to {output_path}")