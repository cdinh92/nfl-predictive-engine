# NFL Predictive Engine

An automated machine learning pipeline built with Python and XGBoost to predict American football match outcomes and compare them against live Vegas spreads.

## Architecture Overview

This project is structured as a continuous online learning system. It ingests historical data, engineers performance features, trains a regression model to predict score margins, and evaluates those predictions against live bookmaker odds to find statistical betting edges.

## Development Roadmap

### Phase 1: Infrastructure & Data Engineering
- [ ] Set up the Python virtual environment and core dependencies (`pandas`, `xgboost`, `scikit-learn`, `requests`).
- [ ] Build the data transformation script to convert wide-format game logs (Home/Away) into long-format team logs.
- [ ] Generate time-shifted 5-game rolling averages for key metrics (Net Yards Per Play, Turnover Differentials).

### Phase 2: Core Machine Learning
- [ ] Implement chronological train/test splitting to prevent data leakage.
- [ ] Train the `XGBRegressor` to predict the continuous point margin (`Home_Points - Away_Points`).
- [ ] Optimize the model for Apple Silicon (M-series) using `n_jobs=-1` for maximum CPU core utilization.

### Phase 3: Walk-Forward Validation & Backtesting
- [ ] Standardize historical bookmaker spreads to calculate the "Vegas Implied Margin".
- [ ] Build an expanding-window (walk-forward) validation loop to simulate real-world weekly betting.
- [ ] Develop a threshold optimizer to find the mathematically optimal Minimum Edge (e.g., model disagrees with Vegas by > 2.0 points).

### Phase 4: Live Production Pipeline
- [ ] Build the entity resolution dictionary (`TEAM_MAP`) to align local database names with external API names.
- [ ] Integrate **The Odds API** to fetch live Friday/Saturday point spreads.
- [ ] Merge live odds with historical rolling averages and execute the final inference script to generate the Sunday Betting Card.

## Local Setup

To run this project locally, ensure you are using a virtual environment:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt