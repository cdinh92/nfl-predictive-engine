# NFL Predictive Engine - Project Diary

**Date:** September 8, 2026

**Summary:** Today was a massive day for the ML pipeline. Built the end-to-end local architecture on the M-series Mac and proved the baseline logic works.

## Environment & Architecture
* Upgraded to Python 3.12 via Homebrew.
* Spun up a dedicated virtual environment (`env`) and solved the VS Code Pylance pathing issue.
* Core Stack: `pandas`, `xgboost`, `scikit-learn`, `nflreadpy`, `pyarrow`, `requests`, `python-dotenv`.
* Solved the macOS C++ compiler issue for XGBoost by installing `libomp` via Homebrew.

## Pipeline Progress (Phases 1-5)
* **Ingestion:** Successfully pulling raw schedule data and team passing stats (yards, completions, attempts). 
* **Feature Engineering & Data Leakage Prevention:** In predictive modeling, data leakage occurs if the dataset includes future information. To combat this, we explicitly restricted the dataset to historical data up to the prediction point. 
* **The Shift Logic:** We used the `.shift(1)` pandas function to offset our 4-game rolling averages. This guarantees the model only knows what happened *prior* to kickoff, preventing any inadvertent leakage of information from the test set into the model.
* **The Algorithm:** We deployed XGBoost (Extreme Gradient Boosting), a powerful algorithm highly effective for binary classification tasks, such as predicting whether Team A will beat Team B. 
* **Training Method:** We used a strict Chronological Train/Test Split (trained on 2023-2024, tested on 2025). We set `n_estimators=100` and `learning_rate=0.1`, as tuning the learning rate alongside the number of trees is important to produce stable predictions. This yielded a 62.13% baseline accuracy.
* **Live Odds Integration:** Hooked up The Odds API to pull real-time DraftKings spreads and moneylines, translating full team names to 3-letter abbreviations.
* **Inference & Calibration:** We converted Vegas odds into implied probabilities and compared them to the XGBoost model's predicted probabilities to isolate the mathematical "Edge." 
* **Future Probability Calibration:** While XGBoost is a widely used and powerful algorithm for classification, the predicted probabilities it outputs may not always be perfectly calibrated out-of-the-box. Tree-based machine learning techniques are not inherently well-calibrated, meaning a predicted probability might not perfectly align with the true likelihood of the event occurring.

## Next Steps For Next Session
* Update the training and inference scripts to digest the newly engineered QB passing stats.
* Push the latest diary and `requirements.txt` to GitHub.
* Explore Scikit-Learn's `CalibratedClassifierCV` to adjust the model's predicted probabilities, improving the reliability of our edge calculations.
