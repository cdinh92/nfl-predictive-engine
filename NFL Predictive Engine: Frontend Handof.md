# NFL Predictive Engine: Frontend Handoff & Developer Guide

## 1. Understanding the Data Payload (`predictions_latest.json`)
The ML pipeline outputs a static, flat JSON file every time it runs. This makes it incredibly lightweight to consume—no database or backend required. 

* **`last_updated`:** An ISO-8601 timestamp so the UI can display "Last Updated: [Time]" to reassure users they are looking at fresh odds.
* **`season` / `week`:** Top-level metadata for page headers and routing.
* **`games`:** An array of objects where every single matchup lives.
* **`game_id`:** A unique string (e.g., `"2025_BAL_KC"`) perfect for React/Next.js `key` props when mapping components.
* **`home_team` / `away_team`:** Contains the 3-letter abbreviation and the current Vegas Moneyline odds. 
* **`prediction`:** The core engine output. It provides the model's exact win probability, Vegas's implied probability, the mathematical Edge, and a string classifying the pick as "High" or "Standard" confidence to make styling CSS classes easier.

---

## 2. Project Timeline & Tasks
This timeline is completely open to adjustment, giving full control over the stack (React, Vue, Svelte) and deployment choices.

**Phase 1: Architecture & Ingestion (Days 1-2)**
* Initialize the frontend repository and select a UI framework.
* Establish the data-fetching strategy. Decide whether to fetch the JSON file directly from the GitHub repository raw link, or set up a pipeline where the JSON is pushed to an S3/Cloudflare bucket on every ML run.
* Create a local mock server or simply import the JSON directly to begin shaping the layout.

**Phase 2: Component Development (Days 3-4)**
* Build the main `GameCard` component to display team abbreviations side-by-side.
* Implement the "Odds Board" table view to cleanly present the Moneyline, Win Probability, and Edge metrics.
* Create dynamic CSS classes that change color based on the `confidence_tier` flag (e.g., highlighting positive edges in green).

**Phase 3: Refinement & Launch (Days 5-6)**
* Ensure the dashboard is mobile-responsive, as most users check sports odds on their phones.
* Configure hosting via Vercel, Netlify, or a similar platform.
* Connect a custom domain and finalize SEO metadata for the launch.

---

## 3. NFL Rules & Betting Basics: A Quick Guide

### NFL Game Mechanics
* **Objective:** The goal is to score more points than the opposing team.
* **Field & Timing:** A standard football field is 120 yards long, with two 10-yard end zones. The game consists of four 15-minute quarters.
* **Possession & Downs:** The offensive team has four attempts, called "downs", to move the ball at least 10 yards forward. If they succeed, they get a new set of four downs; if they fail, possession turns over to the defense. 
* **Scoring Points:**
  * **Touchdown:** 6 points. Scored by carrying or catching the ball in the opponent's end zone.
  * **Field Goal:** 3 points. Scored by kicking the ball through the opponent's goalposts.
  * **Extra Points:** After a touchdown, teams can kick for 1 point or run/pass into the end zone for 2 points.

### Sports Betting Concepts
* **Moneyline:** Betting on which team will win the game outright, without any point spreads.
* **Spread:** A handicap placed on the stronger team to even the playing field. If a team is -3.0, they must win by more than 3 points for the bet to pay out.
* **Implied Probability:** The conversion of betting odds into a percentage that represents the sportsbook's estimated likelihood of a particular outcome. 
* **Edge:** The mathematical difference between our model's predicted win probability and Vegas's implied probability. A positive edge means the model sees value the sportsbook is missing.