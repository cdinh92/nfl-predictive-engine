# NFL Predictive Modeling & Betting Engine

## 🎯 Purpose of the Repository
This repository contains an end-to-end data pipeline and machine learning engine designed to predict National Football League (NFL) game outcomes. Using historical team statistics, rolling averages, and Vegas betting lines, the project automates weekly schedule predictions, assigns confidence tiers to matchups based on blended statistical edges, and generates clean visual summaries and structured JSON data payloads for tracking and deployment.

---

## 🚀 Key Features
* **Automated Weekly Schedules:** Pulls fresh schedule data via `nflreadpy`[cite: 1].
* **Machine Learning & Vegas Blending:** Utilizes an XGBoost classifier trained on rolling team metrics combined with implied Vegas moneyline probabilities[cite: 1, 2].
* **Margin Estimation:** Calculates projected point margins for picks to indicate win intensity.
* **Structured Data Output:** Exports weekly slates into neatly organized JSON payloads (`predictions/week X/`)[cite: 1, 2].
* **Visual Table Generation:** Automatically compiles prediction summaries into color-coded, confidence-tiered PNG matrices using Matplotlib[cite: 2].

---

## 📊 Weekly Predictions Preview

Here is a preview of the generated prediction matrix and confidence tiers for the current slate:

<p align="center">
  <img src="./predictions/week%202/nfl_predictions_week_2.png" alt="NFL Week Predictions & Confidence Tiers" width="100%">
</p>

---

## 🛠️ Project Structure
* `src/` or root scripts:
  * `07_predict_weekly_slate.py` — Runs the prediction pipeline and outputs the weekly JSON[cite: 1].
  * `render-prediction-table.py` — Reads the JSON file and renders the visual PNG table[cite: 2].
* `data/` — Holds model features and historical training datasets.
* `predictions/` — Stores JSON data and generated PNG tables organized by week folders[cite: 1, 2].