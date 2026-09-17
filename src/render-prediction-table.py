from datetime import datetime
import json
import os
import matplotlib.pyplot as plt

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = (
    os.path.dirname(SCRIPT_DIR)
    if os.path.basename(SCRIPT_DIR) == 'src'
    else SCRIPT_DIR
)
PREDICTIONS_BASE = os.path.join(PROJECT_ROOT, 'predictions')

# Target configuration for week and file lookups
target_week = 2
WEEK_FOLDER_NAME = f'week {target_week}'

# Point directly to predictions/week 2/ to read JSON and save PNG
PREDICTIONS_DIR = os.path.join(PREDICTIONS_BASE, WEEK_FOLDER_NAME)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

JSON_FILENAME = f'predictions_2026_week_{target_week}.json'
# Fixed path: now correctly looks inside predictions/week 2/
json_path = os.path.join(PREDICTIONS_DIR, JSON_FILENAME)

if not os.path.exists(json_path):
  print(f'Error: Could not find prediction file at {json_path}')
  exit()

# Load JSON data[cite: 2]
with open(json_path, 'r') as f:
  data = json.load(f)

week_num = data.get('week', target_week)
games_by_date = data.get('games_by_date', {})

# --- CATEGORIZE GAMES BY CONFIDENCE TIER ---
high_tier = []
mod_tier = []
low_tier = []

for gameday, games in games_by_date.items():
  clean_date_str = str(gameday).split('T')[0].split(' ')[0]
  try:
    parsed_date = datetime.strptime(clean_date_str, '%Y-%m-%d')
    formatted_date_str = parsed_date.strftime('%b %d')
  except ValueError:
    formatted_date_str = clean_date_str

  for g in games:
    away_full = g['away_team']['name']
    home_full = g['home_team']['name']
    matchup_str = f'{away_full} @ {home_full}'
    
    raw_time = g['gametime']
    datetime_display = f'{formatted_date_str} • {raw_time}'

    p = g['prediction']
    model_prob = f"{p['model_home_win_prob']*100:.1f}%"
    vegas_prob = f"{p['vegas_home_implied_prob']*100:.1f}%"
    edge_val = f"{p['edge']*100:+.1f}%"
    
    # Include projected margin alongside the pick name
    pick_name = p['pick_name']
    projected_margin = p.get('projected_margin', 0.0)
    pick = f"{pick_name} (+{projected_margin})" if projected_margin > 0 else pick_name
    
    tier = p['confidence_tier']

    row_item = (matchup_str, datetime_display, model_prob, vegas_prob, edge_val, pick)

    if tier == 'High':
      high_tier.append(row_item)
    elif tier == 'Moderate':
      mod_tier.append(row_item)
    else:
      low_tier.append(row_item)

# Define tier titles, contents, and styling colors[cite: 2]
tiers_data = [
    ('HIGH CONFIDENCE', high_tier, '#d4edda', '#155724'),
    ('MODERATE CONFIDENCE', mod_tier, '#fff3cd', '#856404'),
    ('LOW CONFIDENCE', low_tier, '#f8d7da', '#721c24'),
]

# --- RENDER TABLE IMAGE USING MATPLOTLIB ---
fig, ax = plt.subplots(figsize=(11, 10))
ax.axis('off')

table_data = []
row_styles = []

for tier_title, matches, bg_color, text_color in tiers_data:
  table_data.append([tier_title, '', '', '', '', ''])
  row_styles.append(('header', bg_color, text_color))

  if not matches:
    table_data.append(['No games in this tier', '', '', '', '', ''])
    row_styles.append(('data', '#ffffff', '#6c757d'))
  else:
    for match, datetime_str, model, vegas, edge, pick in matches:
      table_data.append([match, datetime_str, model, vegas, edge, pick])
      row_styles.append(('data', bg_color, '#ffffff'))

table = ax.table(
    cellText=table_data,
    colLabels=['Matchup', 'Date & Time', 'Model Blend', 'Vegas', 'Edge', 'Pick'],
    cellLoc='center',
    loc='center',
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.3)

# Strict and balanced column widths to keep layout completely stable[cite: 2]
col_widths = [0.24, 0.26, 0.12, 0.12, 0.11, 0.15]
for i, col in enumerate(col_widths):
  for row in range(len(table_data) + 1):
    table[(row, i)].set_width(col)

# Style main column headers[cite: 2]
for j in range(6):
  header_cell = table[(0, j)]
  header_cell.set_facecolor('#343a40')
  header_cell.set_text_props(weight='bold', color='#ffffff', size=11)

# Style rows and tier headers cleanly[cite: 2]
for idx, (row_type, bg_color, color_val) in enumerate(row_styles):
  row_idx = idx + 1
  if row_type == 'header':
    for j in range(6):
      cell = table[(row_idx, j)]
      cell.set_facecolor(bg_color)
      if j == 0:
        cell.set_text_props(weight='bold', color=color_val, size=10.5, ha='left')
        cell.get_text().set_x(0.05)
      else:
        cell.get_text().set_text('')
  else:
    for j in range(6):
      cell = table[(row_idx, j)]
      cell.set_facecolor('#ffffff')
      cell.set_text_props(color='#212529', size=10)

# Lock title firmly in place right above the table header[cite: 2]
ax.set_title(
    f'NFL Week {week_num} Predictions & Confidence Tiers',
    weight='bold',
    size=15,
    pad=10,
    y=1.02,
)

output_image_name = f'nfl_predictions_week_{week_num}.png'
output_image_path = os.path.join(PREDICTIONS_DIR, output_image_name)
plt.savefig(output_image_path, dpi=300, bbox_inches='tight', pad_inches=0.15)

print(f'Successfully generated and saved table image to: {os.path.abspath(output_image_path)}')