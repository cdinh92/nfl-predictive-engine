# Frontend Handoff Guide: NFL Predictive Engine (V1)

Hey Khoa! 👋 
We have locked in the foundational machine learning pipeline and generated our first live JSON schema for the 2026 NFL Season Kickoff game (New England @ Seattle) [cite: 1.1.1]. Below is everything you need to start hooking up the UI. 

Since this is your domain, feel free to use whichever frontend framework, styling libraries, or tooling you prefer (Next.js, Vite, Tailwind CSS, shadcn/ui, etc.)—whatever lets you move the fastest and build the cleanest dashboard!

---

## 1. JSON Data Structure (`data/predictions_latest.json`)

The predictive engine automatically compiles model probabilities, Vegas market lines, and mathematical edges into a single structured file. Here is the exact schema your frontend will consume:

```json
{
  "last_updated": "2026-09-09T05:00:00Z",
  "season": 2026,
  "week": "Kickoff",
  "games": [
    {
      "game_id": "2026_NE_SEA",
      "home_team": {
        "abbr": "SEA",
        "moneyline": -170
      },
      "away_team": {
        "abbr": "NE"
      },
      "prediction": {
        "model_home_win_prob": 0.323,
        "vegas_home_implied_prob": 0.630,
        "edge": -0.306,
        "pick": "NE",
        "confidence_tier": "Standard"
      }
    }
  ]
}
```

### Field Explanations for UI Binding:
* `model_home_win_prob`: The XGBoost model's true win probability for the home team (e.g., `0.323` = 32.3%).
* `vegas_home_implied_prob`: The converted American odds moneyline into an implied market probability (e.g., `-170` = 63.0%).
* `edge`: The mathematical difference (`model_win_prob - vegas_prob`). A negative value on the home team indicates a strong value opportunity on the away underdog (`NE`).
* `pick`: The automated algorithmic recommendation based on where the positive edge lies.
* `confidence_tier`: Categorized as `"High"` if the absolute edge $\ge 5\%$ (`0.05`), otherwise `"Standard"`.

---

## 2. Recommended Frontend Components to Build

To display this data effectively on the dashboard, consider building these core modular components:
1. **Matchup Card Grid:** Displays team abbreviations, moneyline odds, and a visual probability bar comparing Vegas vs. Model outlooks.
2. **Value Badge Indicator:** A conditional badge that highlights games where the model spots a mathematical edge over the sportsbooks.
3. **Prediction Ledger Table:** A historical table reading from `data/prediction_ledger.csv` to track past model accuracy and performance over time.

---

## 3. Next Steps & Estimated Timeline

Here is a proposed roadmap for the UI development phase. Adjust the timeline based on what fits your schedule best:

| Task / Milestone | Description | Estimated Timeline |
| :--- | :--- | :--- |
| **Phase 1: Project Setup** | Initialize your repo/app using your preferred tech stack and set up local JSON ingestion. | 1–2 Hours |
| **Phase 2: Core UI Components** | Build the matchup cards, probability comparison bars, and value edge indicators. | 3–4 Hours |
| **Phase 3: Ledger & History View** | Build a data table to ingest past predictions and track wins/losses post-game. | 2–3 Hours |
| **Phase 4: Polish & Deploy** | Style the dashboard, ensure responsive design, and deploy to Vercel/Netlify. | 2 Hours |

Let me know what tools or libraries you decide to spin up for this. Happy coding! 🚀
