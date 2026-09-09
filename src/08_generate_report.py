import os
import matplotlib.pyplot as plt
import numpy as np

def generate_visual_report(away_team, home_team, date_str, vegas_prob, model_prob_home, edge, home_scored, home_allowed, away_scored, away_allowed, pick):
    """
    Generates a Markdown summary and a JPG comparison chart, saving both to the predictions/ folder.
    """
    # Smart Pathing
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "src" else SCRIPT_DIR
    PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, "predictions")
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    
    file_prefix = f"{away_team}_vs_{home_team}_{date_str}"
    md_path = os.path.join(PREDICTIONS_DIR, f"{file_prefix}.md")
    jpg_path = os.path.join(PREDICTIONS_DIR, f"{file_prefix}.jpg")
    
    model_prob_away = 1 - model_prob_home
    vegas_prob_away = 1 - vegas_prob
    
    # 1. Generate Markdown
    md_content = f"""# 🏈 NFL Matchup Preview: {away_team} @ {home_team} ({date_str})

---

## 📊 Matchup Overview & Analytics

| Metric | Vegas Market | Model Prediction (XGBoost) | Edge & Outlook |
| :--- | :--- | :--- | :--- |
| **Matchup** | {away_team} @ {home_team} | {away_team} @ {home_team} | **Model vs. Market** |
| **Home Win Probability ({home_team})** | {vegas_prob * 100:.1f}% | {model_prob_home * 100:.1f}% | {edge * 100:.1f}% |
| **Away Win Probability ({away_team})** | {vegas_prob_away * 100:.1f}% | {model_prob_away * 100:.1f}% | {-edge * 100:.1f}% |
| **Rolling Offense (Pts Scored)** | N/A | {home_team}: {home_scored:.1f} \\| {away_team}: {away_scored:.1f} | **Postseason Stats Included** |
| **Rolling Defense (Pts Allowed)**| N/A | {home_team}: {home_allowed:.1f} \\| {away_team}: {away_allowed:.1f} | **4-Game Rolling Average** |
| **Final Recommendation** | Market Favors {home_team if vegas_prob > 0.5 else away_team} | Value on {pick} | **✅ Model Backs {pick}** |

---
"""
    with open(md_path, "w") as f:
        f.write(md_content)
        
    # 2. Generate JPG Chart
    categories = [f'Home Win Prob ({home_team})', f'Away Win Prob ({away_team})']
    vegas_probs = [vegas_prob * 100, vegas_prob_away * 100]
    model_probs = [model_prob_home * 100, model_prob_away * 100]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6), facecolor='#0d1117')
    ax.set_facecolor('#161b22')

    rects1 = ax.bar(x - width/2, vegas_probs, width, label='Vegas Implied Prob', color='#238636')
    rects2 = ax.bar(x + width/2, model_probs, width, label='Model Win Prob', color='#1f6feb')

    ax.set_ylabel('Probability (%)', color='white', fontsize=12, fontweight='bold')
    ax.set_title(f'{away_team} @ {home_team} ({date_str}): Model vs. Vegas', color='white', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, color='white', fontsize=11)
    ax.tick_params(colors='white', which='both')
    ax.legend(facecolor='#21262d', edgecolor='none', labelcolor='white', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.2, color='white')

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', color='white', fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(jpg_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    
    print(f"✅ Generated report and chart in: {PREDICTIONS_DIR}/")

if __name__ == "__main__":
    # Example usage for tomorrow's game
    generate_visual_report(
        away_team="NE",
        home_team="SEA",
        date_str="2026-09-09",
        vegas_prob=0.630,
        model_prob_home=0.323,
        edge=-0.306,
        home_scored=28.0,
        home_allowed=11.5,
        away_scored=23.0,
        away_allowed=9.0,
        pick="NE"
    )