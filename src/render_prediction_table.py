from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import json
import os
import re
import matplotlib.pyplot as plt

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = (
    os.path.dirname(SCRIPT_DIR)
    if os.path.basename(SCRIPT_DIR) == 'src'
    else SCRIPT_DIR
)
PREDICTIONS_BASE = os.path.join(PROJECT_ROOT, 'predictions')

# --- DYNAMIC TARGET CONFIGURATION ---
target_week = 2 # Default fallback
predictor_path = os.path.join(SCRIPT_DIR, '07_predict_weekly_slate.py')

if os.path.exists(predictor_path):
    with open(predictor_path, 'r') as f:
        content = f.read()
        # Regex to find TARGET_WEEK = (any number)
        match = re.search(r'TARGET_WEEK\s*=\s*(\d+)', content)
        if match:
            target_week = int(match.group(1))
            print(f"✅ Automatically synced target week to {target_week} from 07_predict_weekly_slate.py")
else:
    print(f"⚠️ Warning: Could not find {predictor_path} to sync week. Defaulting to {target_week}.")

WEEK_FOLDER_NAME = f'week_{target_week}'

# Point directly to predictions/week_X/ to read JSON and save PNG
PREDICTIONS_DIR = os.path.join(PREDICTIONS_BASE, WEEK_FOLDER_NAME)

# Fallback in case the older folder naming convention ('week X') is used
if not os.path.exists(PREDICTIONS_DIR):
    alt_dir = os.path.join(PREDICTIONS_BASE, f'week {target_week}')
    if os.path.exists(alt_dir):
        PREDICTIONS_DIR = alt_dir

os.makedirs(PREDICTIONS_DIR, exist_ok=True)

JSON_FILENAME = f'predictions_2026_week_{target_week}.json'
json_path = os.path.join(PREDICTIONS_DIR, JSON_FILENAME)

if not os.path.exists(json_path):
  print(f'❌ Error: Could not find prediction file at {json_path}')
  exit()

# Load JSON data
with open(json_path, 'r') as f:
  data = json.load(f)

week_num = data.get('week', target_week)
games_by_date = data.get('games_by_date', {})

# --- CATEGORIZE GAMES BY CONFIDENCE TIER ---
high_tier = []
mod_tier = []
low_tier = []

# Using America/Denver to represent Mountain Time (your timezone when exported)
local_tz = ZoneInfo("America/Denver")

# --- ALTERNATIVE TIMEZONES (Uncomment the one you need) ---
# local_tz = ZoneInfo("America/New_York")      # Eastern Time (ET)
# local_tz = ZoneInfo("America/Chicago")       # Central Time (CT)
# local_tz = ZoneInfo("America/Los_Angeles")   # Pacific Time (PT)
# local_tz = ZoneInfo("UTC")                   # Coordinated Universal Time (UTC)
# local_tz = ZoneInfo("Europe/London")         # British Summer Time / Greenwich Mean Time (BST/GMT)
# ----------------------------------------------------------

for gameday, games in games_by_date.items():
  for g in games:
    away_full = g['away_team']['name']
    home_full = g['home_team']['name']
    matchup_str = f'{away_full} @ {home_full}'
    
    raw_time = g['gametime']
    
    # Check if raw_time is a valid ISO 8601 UTC string (ends with 'Z')
    if raw_time and raw_time.endswith('Z'):
        try:
            # Parse the UTC time
            dt_utc = datetime.strptime(raw_time, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
            # Convert to local Mountain Time
            dt_local = dt_utc.astimezone(local_tz)
            
            # Format date and time for display
            formatted_date_str = dt_local.strftime('%b %d')
            formatted_time_str = dt_local.strftime('%I:%M %p').lstrip('0')
            datetime_display = f'{formatted_date_str} • {formatted_time_str}'
        except ValueError:
            datetime_display = f'{gameday} • {raw_time}'
    else:
        # Fallback if the time isn't in the expected UTC format
        clean_date_str = str(gameday).split('T')[0].split(' ')[0]
        try:
            parsed_date = datetime.strptime(clean_date_str, '%Y-%m-%d')
            formatted_date_str = parsed_date.strftime('%b %d')
        except ValueError:
            formatted_date_str = clean_date_str
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

# Define tier titles, contents, and styling colors
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
    colLabels=['Matchup', 'Date & Time', 'Model', 'Vegas', 'Edge', 'Pick'],
    cellLoc='center',
    loc='center',
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.3)

# Strict and balanced column widths to keep layout completely stable
col_widths = [0.24, 0.26, 0.12, 0.12, 0.11, 0.15]
for i, col in enumerate(col_widths):
  for row in range(len(table_data) + 1):
    table[(row, i)].set_width(col)

# Style main column headers
for j in range(6):
  header_cell = table[(0, j)]
  header_cell.set_facecolor('#343a40')
  header_cell.set_text_props(weight='bold', color='#ffffff', size=11)

# Style rows and tier headers cleanly
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

# Lock title firmly in place right above the table header
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