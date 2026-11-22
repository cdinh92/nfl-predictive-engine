from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import json
import math
import os
import re
import requests
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = (
    os.path.dirname(SCRIPT_DIR)
    if os.path.basename(SCRIPT_DIR) == 'src'
    else SCRIPT_DIR
)
PREDICTIONS_BASE = os.path.join(PROJECT_ROOT, 'predictions')
ASSETS_DIR = os.path.join(PROJECT_ROOT, 'assets')
LOGOS_DIR = os.path.join(ASSETS_DIR, 'logos')
os.makedirs(LOGOS_DIR, exist_ok=True)

# --- DYNAMIC TARGET CONFIGURATION ---
target_week = 11
predictor_path = os.path.join(SCRIPT_DIR, '07_predict_weekly_slate.py')

if os.path.exists(predictor_path):
    with open(predictor_path, 'r') as f:
        content = f.read()
        match = re.search(r'TARGET_WEEK\s*=\s*(\d+)', content)
        if match:
            target_week = int(match.group(1))
            print(f"✅ Automatically synced target week to {target_week} from 07_predict_weekly_slate.py")

WEEK_FOLDER_NAME = f'week_{target_week}'
PREDICTIONS_DIR = os.path.join(PREDICTIONS_BASE, WEEK_FOLDER_NAME)
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


def get_team_logo(team_abbr: str, max_px: int = 72) -> OffsetImage:
    """Fetches, caches, crops transparent whitespace, and normalizes NFL logos."""
    espn_map = {'LA': 'lar', 'LAR': 'lar', 'WAS': 'wsh'}
    code = espn_map.get(team_abbr.upper(), team_abbr.lower())
    cache_path = os.path.join(LOGOS_DIR, f'{code}.png')

    if not os.path.exists(cache_path):
        try:
            url = f'https://a.espncdn.com/i/teamlogos/nfl/500/{code}.png'
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            with open(cache_path, 'wb') as f:
                f.write(resp.content)
        except Exception:
            placeholder = Image.new('RGBA', (100, 100), (30, 41, 59, 255))
            placeholder.save(cache_path)

    img = Image.open(cache_path).convert('RGBA')
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    img.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
    return OffsetImage(img, zoom=0.50)


def format_mt_kickoff(gameday: str, raw_time: str, local_tz: ZoneInfo) -> str:
    if raw_time and raw_time.endswith('Z'):
        try:
            dt_utc = datetime.strptime(raw_time, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
            dt_local = dt_utc.astimezone(local_tz)
            return f"{dt_local.strftime('%a %b %d').upper()} • {dt_local.strftime('%I:%M %p').lstrip('0')} MT"
        except ValueError:
            pass
    return f"{gameday} • {raw_time}"


def draw_market_cell(ax, x, y, w, h, main_text, odds_text, is_pick=False, edge_tag=None):
    """Draws a clean, book-agnostic dark-mode odds tile with emerald model-edge highlight."""
    bg_color = '#0F2E28' if is_pick else '#1A2336'
    edge_color = '#10B981' if is_pick else '#28354D'
    lw = 1.6 if is_pick else 0.8

    tile = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.0,rounding_size=0.018',
        facecolor=bg_color,
        edgecolor=edge_color,
        linewidth=lw,
        transform=ax.transAxes,
        zorder=2,
    )
    ax.add_patch(tile)

    # Centered primary line (+15.5, U 44.5, -1100)
    main_color = '#ECFDF5' if is_pick else '#E2E8F0'
    main_y = y + h * (0.50 if odds_text is None else 0.64)
    ax.text(
        x + w / 2, main_y, main_text,
        ha='center', va='center',
        fontsize=11.2, fontweight='bold', color=main_color,
        transform=ax.transAxes, zorder=3,
    )

    # Centered standard odds sub-label (-110)
    if odds_text is not None:
        sub_color = '#6EE7B7' if is_pick else '#94A3B8'
        ax.text(
            x + w / 2, y + h * 0.25, odds_text,
            ha='center', va='center',
            fontsize=8.8, fontweight='normal', color=sub_color,
            transform=ax.transAxes, zorder=3,
        )

    # Top-right mini edge pill when selected by model
    if is_pick and edge_tag:
        ax.text(
            x + w - 0.012, y + h - 0.014, edge_tag,
            ha='right', va='top',
            fontsize=6.6, fontweight='bold', color='#052E16',
            bbox=dict(boxstyle='round,pad=0.16', facecolor='#34D399', edgecolor='none'),
            transform=ax.transAxes, zorder=4,
        )


def draw_custom_matchup_card(ax, game_obj: dict, kickoff_label: str):
    ax.set_facecolor('#0B0F19')
    ax.axis('off')

    # Outer elevated card container
    card_bg = patches.FancyBboxPatch(
        (0.015, 0.02), 0.97, 0.96,
        boxstyle='round,pad=0.0,rounding_size=0.03',
        facecolor='#131A29',
        edgecolor='#232E42',
        linewidth=1.3,
        transform=ax.transAxes,
        zorder=1,
    )
    ax.add_patch(card_bg)

    away_abbr = game_obj['away_team']['abbr']
    home_abbr = game_obj['home_team']['abbr']
    p = game_obj['prediction']
    # Support both new 'market_lines' key and older 'betmgm_lines' key
    mkt = game_obj.get('market_lines') or game_obj.get('betmgm_lines', {})

    away_score = p.get('away_projected_score', 0.0)
    home_score = p.get('home_projected_score', 0.0)
    home_prob = float(p.get('model_home_win_prob', 0.5))
    away_prob = 1.0 - home_prob

    # --- TOP BAR: Kickoff & Confidence Pill ---
    ax.text(0.045, 0.925, kickoff_label, ha='left', va='center',
            fontsize=8.2, fontweight='bold', color='#94A3B8', transform=ax.transAxes, zorder=3)

    tier = p.get('confidence_tier', 'Low')
    tier_palette = {
        'High': ('#064E3B', '#34D399'),
        'Moderate': ('#451A03', '#FBBF24'),
        'Low': ('#1E293B', '#94A3B8'),
    }
    t_bg, t_fg = tier_palette.get(tier, ('#1E293B', '#94A3B8'))
    ax.text(
        0.955, 0.925, f"● {tier.upper()} CONFIDENCE",
        ha='right', va='center', fontsize=7.6, fontweight='bold', color=t_fg,
        bbox=dict(boxstyle='round,pad=0.25', facecolor=t_bg, edgecolor='none'),
        transform=ax.transAxes, zorder=3,
    )

    # --- SCOREBOARD HERO HEADER ---
    ax.text(0.50, 0.855, 'PROJECTED SCORE', ha='center', va='center',
            fontsize=7.8, fontweight='bold', color='#38BDF8', transform=ax.transAxes, zorder=3)

    # Away Team Block (Left)
    ax.add_artist(AnnotationBbox(get_team_logo(away_abbr, max_px=74), (0.14, 0.745), frameon=False, xycoords='axes fraction', zorder=4))
    ax.text(0.24, 0.77, away_abbr, ha='left', va='center',
            fontsize=13.5, fontweight='bold', color='#F8FAFC', transform=ax.transAxes, zorder=3)
    away_score_color = '#34D399' if away_score > home_score else '#CBD5E1'
    ax.text(0.24, 0.695, f"{away_score:.1f}", ha='left', va='center',
            fontsize=15.5, fontweight='bold', color=away_score_color, transform=ax.transAxes, zorder=3)

    # Center @ Divider
    ax.text(0.50, 0.735, '@', ha='center', va='center',
            fontsize=12.0, fontweight='bold', color='#475569', transform=ax.transAxes, zorder=3)

    # Home Team Block (Right)
    ax.add_artist(AnnotationBbox(get_team_logo(home_abbr, max_px=74), (0.86, 0.745), frameon=False, xycoords='axes fraction', zorder=4))
    ax.text(0.76, 0.77, home_abbr, ha='right', va='center',
            fontsize=13.5, fontweight='bold', color='#F8FAFC', transform=ax.transAxes, zorder=3)
    home_score_color = '#34D399' if home_score >= away_score else '#CBD5E1'
    ax.text(0.76, 0.695, f"{home_score:.1f}", ha='right', va='center',
            fontsize=15.5, fontweight='bold', color=home_score_color, transform=ax.transAxes, zorder=3)

    # --- WIN PROBABILITY SPLIT BAR ---
    bar_x, bar_y, bar_w, bar_h = 0.045, 0.585, 0.91, 0.026
    away_w = max(0.08, min(0.92, away_prob)) * bar_w
    home_w = bar_w - away_w

    away_bar_color = '#10B981' if away_prob > home_prob else '#334155'
    home_bar_color = '#10B981' if home_prob >= away_prob else '#334155'

    ax.add_patch(patches.Rectangle((bar_x, bar_y), away_w, bar_h, facecolor=away_bar_color, edgecolor='none', transform=ax.transAxes, zorder=3))
    ax.add_patch(patches.Rectangle((bar_x + away_w, bar_y), home_w, bar_h, facecolor=home_bar_color, edgecolor='none', transform=ax.transAxes, zorder=3))

    ax.text(bar_x, bar_y + 0.042, f"{away_abbr} {away_prob*100:.1f}%", ha='left', va='center',
            fontsize=7.8, fontweight='bold', color='#94A3B8', transform=ax.transAxes, zorder=3)
    ax.text(0.50, bar_y + 0.042, "WIN PROBABILITY", ha='center', va='center',
            fontsize=7.2, fontweight='bold', color='#64748B', transform=ax.transAxes, zorder=3)
    ax.text(bar_x + bar_w, bar_y + 0.042, f"{home_prob*100:.1f}% {home_abbr}", ha='right', va='center',
            fontsize=7.8, fontweight='bold', color='#94A3B8', transform=ax.transAxes, zorder=3)

    # --- MARKET MATRIX HEADERS ---
    col_x = [0.125, 0.405, 0.685]
    col_w = 0.270
    for cx, title in zip(col_x, ['SPREAD', 'TOTAL', 'MONEYLINE']):
        ax.text(cx + col_w / 2, 0.525, title, ha='center', va='center',
                fontsize=8.2, fontweight='bold', color='#64748B', transform=ax.transAxes, zorder=3)

    # Left-side Row Team Logos
    ax.add_artist(AnnotationBbox(get_team_logo(away_abbr, max_px=56), (0.075, 0.39), frameon=False, xycoords='axes fraction', zorder=4))
    ax.add_artist(AnnotationBbox(get_team_logo(home_abbr, max_px=56), (0.075, 0.165), frameon=False, xycoords='axes fraction', zorder=4))

    row_h = 0.185
    away_y, home_y = 0.298, 0.072

    # Standard market lines & model edge labels
    away_spread = mkt.get('away_spread')
    home_spread = mkt.get('home_spread')
    away_sp_odds = mkt.get('away_spread_odds', -110)
    home_sp_odds = mkt.get('home_spread_odds', -110)

    total_line = mkt.get('total_line')
    over_odds = mkt.get('over_odds', -110)
    under_odds = mkt.get('under_odds', -110)

    away_ml = mkt.get('away_ml', game_obj['away_team'].get('moneyline', -110))
    home_ml = mkt.get('home_ml', game_obj['home_team'].get('moneyline', -110))

    ats_edge = p.get('ats_edge')
    tot_edge = p.get('total_edge')
    sp_tag = f"+{abs(ats_edge):.1f}" if ats_edge is not None else None
    tot_tag = f"+{abs(tot_edge):.1f}" if tot_edge is not None else None

    # 1. Spread Cells
    draw_market_cell(
        ax, col_x[0], away_y, col_w, row_h,
        f"{away_spread:+.1f}" if away_spread is not None else "N/A",
        f"{int(away_sp_odds):+d}" if away_spread is not None else None,
        is_pick=(p.get('spread_pick_side') == 'away'),
        edge_tag=sp_tag,
    )
    draw_market_cell(
        ax, col_x[0], home_y, col_w, row_h,
        f"{home_spread:+.1f}" if home_spread is not None else "N/A",
        f"{int(home_sp_odds):+d}" if home_spread is not None else None,
        is_pick=(p.get('spread_pick_side') == 'home'),
        edge_tag=sp_tag,
    )

    # 2. Total Cells
    draw_market_cell(
        ax, col_x[1], away_y, col_w, row_h,
        f"O {total_line:.1f}" if total_line is not None else "N/A",
        f"{int(over_odds):+d}" if total_line is not None else None,
        is_pick=(p.get('total_pick_side') == 'over'),
        edge_tag=tot_tag,
    )
    draw_market_cell(
        ax, col_x[1], home_y, col_w, row_h,
        f"U {total_line:.1f}" if total_line is not None else "N/A",
        f"{int(under_odds):+d}" if total_line is not None else None,
        is_pick=(p.get('total_pick_side') == 'under'),
        edge_tag=tot_tag,
    )

    # 3. Moneyline Cells
    draw_market_cell(
        ax, col_x[2], away_y, col_w, row_h,
        f"{int(away_ml):+d}", None,
        is_pick=(p.get('ml_pick_side') == 'away'),
        edge_tag="PICK",
    )
    draw_market_cell(
        ax, col_x[2], home_y, col_w, row_h,
        f"{int(home_ml):+d}", None,
        is_pick=(p.get('ml_pick_side') == 'home'),
        edge_tag="PICK",
    )


# --- LOAD DATA & RENDER 2-COLUMN GRID ---
with open(json_path, 'r') as f:
    data = json.load(f)

week_num = data.get('week', target_week)
scale_factor = data.get('scale_factor', 1.0)
games_by_date = data.get('games_by_date', {})
local_tz = ZoneInfo('America/Denver')

all_games = []
for gameday, games in sorted(games_by_date.items()):
    for g in games:
        kickoff_str = format_mt_kickoff(gameday, g.get('gametime', 'TBD'), local_tz)
        all_games.append((kickoff_str, g))

if not all_games:
    print('⚠️ No games found in JSON to render.')
    exit()

num_games = len(all_games)
ncols = 2
nrows = math.ceil(num_games / ncols)

fig_width = 15.5
fig_height = max(4.5, 4.15 * nrows + 1.0)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), dpi=220)
fig.patch.set_facecolor('#0B0F19')

if nrows == 1:
    axes_flat = list(axes)
else:
    axes_flat = [ax for row in axes for ax in row]

fig.suptitle(
    f"NFL PREDICTIVE ENGINE  •  WEEK {week_num} MODEL vs. MARKET LINES",
    fontsize=16.5, fontweight='bold', color='#F8FAFC', y=0.985,
)
fig.text(
    0.50, 0.962,
    f"XGBoost Pure-Stats Score Projections (Variance Scale: {scale_factor:.2f}x)  •  All Kickoffs Mountain Time (MT)",
    ha='center', va='center', fontsize=10.0, fontweight='normal', color='#94A3B8',
)

for idx, (kickoff_str, game_obj) in enumerate(all_games):
    draw_custom_matchup_card(axes_flat[idx], game_obj, kickoff_str)

for empty_idx in range(num_games, len(axes_flat)):
    axes_flat[empty_idx].set_facecolor('#0B0F19')
    axes_flat[empty_idx].axis('off')

plt.subplots_adjust(left=0.015, right=0.985, top=0.94, bottom=0.015, wspace=0.04, hspace=0.06)

output_image_name = f'nfl_predictions_week_{week_num}.png'
output_image_path = os.path.join(PREDICTIONS_DIR, output_image_name)
plt.savefig(output_image_path, dpi=220, facecolor='#0B0F19')
plt.close(fig)

print(f'Generated book-agnostic NFL Predictive Engine board at: {os.path.abspath(output_image_path)}')