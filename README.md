# 🏈 NFL Predictive Engine: Week 3 Update

Welcome to the **NFL Predictive Engine** weekly update! This document outlines our performance evaluation from Week 2 and provides the updated machine learning projections for Week 3 based on our refactored algorithm.

## 📊 1. Week 2 Prediction Performance Evaluation

During Week 2, the predictive engine produced an overall straight-up (SU) record of **9–7 (56.3%)**. Performance varied significantly across confidence tiers, highlighting specific vulnerabilities in road-underdog spots and over-reliance on preseason baseline weights.

### 🏈 Week 2 Slate Results Breakdown

| Matchup | Model Blend | Vegas Implied | Edge | Pick | Actual Score | Outcome | 
 | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **🟢 High Confidence** |  |  |  |  |  |  | 
| Lions @ Bills | 70.6% | 70.4% | +0.2% | Bills (+6.4) | Bills 41, Lions 31 | ✅ **Correct** | 
| Giants @ Rams | 75.6% | 78.0% | \-2.4% | Rams (+7.7) | Rams 28, Giants 6 | ✅ **Correct** | 
| **🟡 Moderate Confidence** |  |  |  |  |  |  | 
| Saints @ Ravens | 66.6% | 79.2% | \-12.5% | Ravens (+5.3) | Saints 24, Ravens 17 | ❌ **Incorrect** | 
| Bengals @ Texans | 62.7% | 58.7% | +4.0% | Texans (+4.3) | Bengals 20, Texans 6 | ❌ **Incorrect** | 
| Browns @ Buccaneers | 63.2% | 81.5% | \-18.3% | Buccaneers (+4.4) | Browns 23, Buccaneers 19 | ❌ **Incorrect** | 
| Raiders @ Chargers | 60.7% | 75.3% | \-14.6% | Chargers (+3.8) | Raiders 26, Chargers 14 | ❌ **Incorrect** | 
| Seahawks @ Cardinals | 37.1% | 35.7% | +1.3% | Seahawks (+4.4) | Seahawks 31, Cardinals 7 | ✅ **Correct** | 
| Dolphins @ 49ers | 69.4% | 90.0% | \-20.6% | 49ers (+6.0) | 49ers 35, Dolphins 13 | ✅ **Correct** | 
| **🔴 Low Confidence** |  |  |  |  |  |  | 
| Panthers @ Falcons | 63.4% | 43.5% | +19.9% | Falcons (+4.5) | Panthers 34, Falcons 3 | ❌ **Incorrect** | 
| Vikings @ Bears | 58.7% | 67.2% | \-8.5% | Bears (+3.3) | Vikings 9, Bears 3 | ❌ **Incorrect** | 
| Steelers @ Patriots | 59.1% | 69.2% | \-10.1% | Patriots (+3.4) | Patriots 20, Steelers 3 | ✅ **Correct** | 
| Packers @ Jets | 48.2% | 38.5% | +9.7% | Packers (+1.5) | Packers 20, Jets 17 (OT) | ✅ **Correct** | 
| Eagles @ Titans | 59.9% | 27.8% | +32.1% | Titans (+3.6) | Eagles 24, Titans 20 | ❌ **Incorrect** | 
| Jaguars @ Broncos | 59.0% | 59.7% | \-0.7% | Broncos (+3.3) | Broncos 20, Jaguars 13 | ✅ **Correct** | 
| Commanders @ Cowboys | 52.5% | 68.6% | \-16.1% | Cowboys (+1.6) | Cowboys 37, Commanders 20 | ✅ **Correct** | 
| Colts @ Chiefs | 57.4% | 73.5% | \-16.1% | Chiefs (+2.9) | Chiefs 33, Colts 30 (OT) | ✅ **Correct** | 

> **Key Diagnostic Takeaways:**
>
> * **High-Confidence Dominance (2–0, 100%):** Both top plays (Bills and Rams) cashed with double-digit cover margins.
>
> * **Moderate-Tier Collapse (2–4, 33.3%):** The model struggled heavily with road upsets (Saints, Bengals, Browns, Raiders). The original weighting allowed heavy home favorites to absorb too much market momentum without penalizing red-zone turnover variance.
>
> * **Low-Confidence Outperformance (5–3, 62.5%):** Correctly identified underdog edges in Packers @ Jets, while remaining conservative on close finishes.

## ⚙️ 2. Updated Model Architecture & Execution Workflow

To resolve the blind spots exposed in Week 2, the pipeline has been refactored across feature engineering, learning targets, and probability ensembling.

```
nflreadpy Raw Data ──> 02_feature_engineering.py (EMA Differentials)
                                │
                                ▼
                       03_model_training.py (XGBoost Regressor)
                                │
                                ▼
                       07_predict_weekly_slate.py (80/20 Ensemble & Odds Vig Strip)

```

### A. Dynamic EMA Feature Engineering (`02_feature_engineering.py`)

Rather than relying on unweighted seasonal aggregates, the model transitions to **Exponential Moving Averages (EMA)** with an active smoothing span of `span = 4`.
* **Leakage Prevention:** Every rolling window shifts backward by 1 week (`shift(1)`), ensuring the model never observes current game data during feature synthesis.
* **Efficiency Focus:** Incorporates advanced efficiency metrics from `nflreadpy`, including:
  * Expected Points Added (`off_epa`, `def_epa`)
  * Completion Percentage Over Expected (`off_cpoe`)
  * Net yards per play (`net_yds_per_play`)
* **Mathematical Compression (9 Differentials):** To optimize signal-to-noise ratio, features are condensed into 9 differential metrics:
  1. `diff_pts_scored` = Home EMA Pts Scored - Away EMA Pts Scored
  2. `diff_pts_allowed` = Away EMA Pts Allowed - Home EMA Pts Allowed
  3. `diff_pass_yds` = Home EMA Pass Yds - Away EMA Pass Yds
  4. `diff_rush_yds` = Home EMA Rush Yds - Away EMA Rush Yds
  5. `diff_comp_pct` = Home EMA Comp % - Away EMA Comp %
  6. `diff_net_yds_play` = Home EMA Net Yds/Play - Away EMA Net Yds/Play
  7. `diff_off_epa` = Home EMA Off EPA - Away EMA Off EPA
  8. `diff_def_epa` = Away EMA Def EPA - Home EMA Def EPA
  9. `diff_off_cpoe` = Home EMA CPOE - Away EMA CPOE

### B. Point Margin Regression Target (`03_model_training.py`)

Rather than treating game prediction as a noisy binary classification task (0 or 1), the engine trains an **XGBoost Regressor** (`xgb.XGBRegressor`) on the actual score margin:
`home_margin = home_score - away_score`

* **Hyperparameter Regularization:**
  * `learning_rate`: 0.01 (slowed from 0.05 to prevent overfitting early trends)
  * `n_estimators`: 250
  * `max_depth`: 3 (shallow trees reduce over-reliance on individual blowouts)
  * `subsample` & `colsample_bytree`: 0.8
* **Margin-to-Probability Transformation:**
  Predictions output a point margin rather than a direct probability. The predicted margin is converted to win probability using the cumulative standard normal distribution ($\Phi$), parameterized by the NFL's historical standard deviation of scoring margins ($\sigma \approx 13.5$):
  $$P(\text{Home Win}) = \Phi\left(\frac{\widehat{\text{margin}}}{13.5}\right)$$

### C. Slate Inference & Ensemble Architecture (`07_predict_weekly_slate.py`)
When executing predictions on the active week:
1. **Bookmaker Vig Removal:** Raw moneyline odds are stripped of the house edge to identify true market implied probability:
   $$P_{\text{raw}} = \begin{cases} \frac{-ML}{-ML + 100}, & ML < 0 \\ \frac{100}{ML + 100}, & ML > 0 \end{cases}$$
   $$P_{\text{vegas}} = \frac{P_{\text{raw, home}}}{P_{\text{raw, home}} + P_{\text{raw, away}}}$$
2. **80/20 Ensemble Weighting:**
   Predictions blend purely data-driven fundamentals with betting market consensus:
   $$\text{Model Win Prob} = 0.80 \times P(\text{Home Win}) + 0.20 \times P_{\text{vegas}}$$

---

## 🔮 3. Week 3 Projections Slate

Projections generated with the updated 9-differential XGBoost regression pipeline:

| Matchup | Gametime (MT) | Model Win Prob | Vegas Implied | Edge | Pick (Proj. Margin) | Tier | 
 | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **Falcons @ Packers** | Thu 5:15 PM | 68.4% | 63.1% | +5.3% | Packers (by 4.8 pts) | 🟡 Moderate | 
| **Bengals @ Steelers** | Sun 10:00 AM | 52.1% | 50.5% | +1.6% | Bengals (by 0.8 pts) | 🔴 Low | 
| **Patriots @ Jaguars** | Sun 10:00 AM | 61.5% | 65.0% | \-3.5% | Jaguars (by 3.2 pts) | 🟡 Moderate | 
| **Titans @ Giants** | Sun 10:00 AM | 48.3% | 42.1% | +6.2% | Titans (by 0.6 pts) | 🔴 Low | 
| **Seahawks @ Commanders** | Sun 10:00 AM | 57.0% | 53.8% | +3.2% | Seahawks (by 1.9 pts) | 🔴 Low | 
| **Jets @ Lions** | Sun 10:00 AM | 74.2% | 78.5% | \-4.3% | Lions (by 8.1 pts) | 🟢 High | 
| **Panthers @ Browns** | Sun 10:00 AM | 44.0% | 51.2% | \-7.2% | Browns (by 2.1 pts) | 🔴 Low | 
| **Chiefs @ Dolphins** | Sun 10:00 AM | 71.0% | 68.4% | +2.6% | Chiefs (by 6.7 pts) | 🟢 High | 
| **Chargers @ Bills** | Sun 10:00 AM | 31.5% | 24.0% | +7.5% | Bills (by 6.9 pts) | 🔴 Low | 
| **Texans @ Colts** | Sun 10:00 AM | 53.5% | 56.0% | \-2.5% | Colts (by 1.2 pts) | 🔴 Low | 
| **Cardinals @ 49ers** | Sun 1:05 PM | 22.8% | 17.5% | +5.3% | 49ers (by 10.4 pts) | 🔴 Low | 
| **Vikings @ Buccaneers** | Sun 1:05 PM | 50.4% | 49.0% | +1.4% | Vikings (by 0.2 pts) | 🔴 Low | 
| **Raiders @ Saints** | Sun 1:25 PM | 45.2% | 52.0% | \-6.8% | Saints (by 2.4 pts) | 🔴 Low | 
| **Ravens @ Cowboys** | Sun 1:25 PM | 54.8% | 51.0% | +3.8% | Ravens (by 1.5 pts) | 🔴 Low | 
| **Rams @ Broncos** | Sun 5:20 PM | 62.0% | 59.5% | +2.5% | Rams (by 3.8 pts) | 🟡 Moderate | 
| **Eagles @ Bears** | Mon 5:15 PM | 66.5% | 61.2% | +5.3% | Eagles (by 5.1 pts) | 🟡 Moderate | 