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
FULL_TEAM_NAMES = {
    'ARI': 'Arizona Cardinals', 'ATL': 'Atlanta Falcons', 'BAL': 'Baltimore Ravens',
    'BUF': 'Buffalo Bills', 'CAR': 'Carolina Panthers', 'CHI': 'Chicago Bears',
    'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns', 'DAL': 'Dallas Cowboys',
    'DEN': 'Denver Broncos', 'DET': 'Detroit Lions', 'GB': 'Green Bay Packers',
    'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars',
    'KC': 'Kansas City Chiefs', 'LAC': 'Los Angeles Chargers', 'LAR': 'Los Angeles Rams',
    'LA': 'Los Angeles Rams', 'LV': 'Las Vegas Raiders', 'MIA': 'Miami Dolphins',
    'MIN': 'Minnesota Vikings', 'NE': 'New England Patriots', 'NO': 'New Orleans Saints',
    'NYG': 'New York Giants', 'NYJ': 'New York Jets', 'PHI': 'Philadelphia Eagles',
    'PIT': 'Pittsburgh Steelers', 'SEA': 'Seattle Seahawks', 'SF': 'San Francisco 49ers',
    'TB': 'Tampa Bay Buccaneers', 'TEN': 'Tennessee Titans', 'WAS': 'Washington Commanders',
}
os.makedirs(LOGOS_DIR, exist_ok=True)


def get_team_nickname(abbr: str, fallback: str = None) -> str:
    full = FULL_TEAM_NAMES.get(abbr, fallback or abbr)
    return full.rsplit(' ', 1)[-1]


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


def get_team_logo(team_abbr: str, max_px: int = 72, zoom: float = 0.50) -> OffsetImage:
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
    return OffsetImage(img, zoom=zoom)


def format_mt_kickoff(gameday: str, raw_time: str, local_tz: ZoneInfo, compact: bool = False) -> str:
    if raw_time and raw_time.endswith('Z'):
        try:
            dt_utc = datetime.strptime(raw_time, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
            dt_local = dt_utc.astimezone(local_tz)
            if compact:
                return f"{dt_local.strftime('%b %d')} • {dt_local.strftime('%I:%M %p').lstrip('0')}"
            return f"{dt_local.strftime('%a %b %d').upper()} • {dt_local.strftime('%I:%M %p').lstrip('0')} MT"
        except ValueError:
            pass
    return f"{gameday} • {raw_time}"


def normalize_tier_key(raw_tier: str) -> str:
    val = str(raw_tier).strip().lower()
    if val == 'high':
        return 'HIGH'
    if val in ('moderate', 'medium'):
        return 'MEDIUM'
    return 'LOW'


def render_simple_table(all_games: list, week_num: int, scale_factor: float, local_tz: ZoneInfo, output_path: str):
    """Renders a 9-column summary table with team nicknames and tier sections."""
    tier_order = ['HIGH', 'MEDIUM', 'LOW']
    tier_meta = {
        'HIGH': {
            'title': '● HIGH CONFIDENCE PICKS',
            'bg': '#331D08',
            'accent': '#FBBF24',
            'border': '#D97706',
            'row_even': '#191926',
            'row_odd': '#202030',
        },
        'MEDIUM': {
            'title': '● MEDIUM CONFIDENCE PICKS',
            'bg': '#261844',
            'accent': '#C084FC',
            'border': '#9333EA',
            'row_even': '#16172B',
            'row_odd': '#1C1D36',
        },
        'LOW': {
            'title': '● LOW CONFIDENCE / TOSS-UP PICKS',
            'bg': '#0A2E46',
            'accent': '#38BDF8',
            'border': '#0284C7',
            'row_even': '#111B2E',
            'row_odd': '#152238',
        },
    }

    grouped = {'HIGH': [], 'MEDIUM': [], 'LOW': []}
    for item in all_games:
        _, g = item
        t_key = normalize_tier_key(g.get('prediction', {}).get('confidence_tier', 'Low'))
        grouped[t_key].append(item)

    slots = [('column_header', None)]
    for t_key in tier_order:
        if grouped[t_key]:
            slots.append(('tier_banner', t_key))
            for game_item in grouped[t_key]:
                slots.append(('game_row', (t_key, game_item)))

    slot_weights = [
        0.92 if s_type == 'column_header' else (0.72 if s_type == 'tier_banner' else 1.04)
        for s_type, _ in slots
    ]
    total_weight = sum(slot_weights)

    header_height_in = 1.32
    footer_height_in = 0.25
    unit_row_in = 0.60
    fig_width = 15.2
    fig_height = max(4.0, header_height_in + footer_height_in + total_weight * unit_row_in)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=220)
    fig.patch.set_facecolor('#0B0F19')
    ax.set_facecolor('#0B0F19')
    ax.axis('off')

    fig.suptitle(
        f"NFL PREDICTIVE ENGINE  •  WEEK {week_num} QUICK PICKS BY TIER",
        fontsize=16.5, fontweight='bold', color='#F8FAFC', y=1.0 - (0.34 / fig_height),
    )
    fig.text(
        0.50, 1.0 - (0.68 / fig_height),
        f"Model Summary (Variance Scale: {scale_factor:.2f}x)  •  Grouped by Confidence Tier  •  Mountain Time (MT)",
        ha='center', va='center', fontsize=10.0, color='#94A3B8',
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.subplots_adjust(left=0.014, right=0.986, top=1.0 - (0.95 / fig_height), bottom=0.02)

    headers = [
        'Match up',
        'Date & Time',
        'Model Blend',
        'Vegas',
        'Winner Pick',
        'Margin',
        'Home Score',
        'Away Score',
        'Total',
    ]
    col_widths = [0.215, 0.130, 0.095, 0.085, 0.130, 0.080, 0.090, 0.090, 0.085]
    col_starts = []
    acc = 0.0
    for w in col_widths:
        col_starts.append(acc)
        acc += w

    outer_box = patches.FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0,
        boxstyle='round,pad=0.0,rounding_size=0.012',
        facecolor='#131A29',
        edgecolor='#263349',
        linewidth=1.2,
        zorder=1,
    )
    ax.add_patch(outer_box)

    cursor_y = 1.0
    tier_row_counter = 0

    for idx, ((s_type, payload), weight) in enumerate(zip(slots, slot_weights)):
        slot_h = weight / total_weight
        row_y = cursor_y - slot_h
        row_mid_y = row_y + slot_h / 2
        cursor_y = row_y

        if s_type == 'column_header':
            hdr_bg = patches.Rectangle((0.0, row_y), 1.0, slot_h, facecolor='#1E293B', edgecolor='none', zorder=2)
            ax.add_patch(hdr_bg)
            ax.plot([0, 1], [row_y, row_y], color='#334155', linewidth=1.2, zorder=3)

            for cx, cw, label in zip(col_starts, col_widths, headers):
                ax.text(
                    cx + cw / 2, row_mid_y, label.upper(),
                    ha='center', va='center',
                    fontsize=8.6, fontweight='bold', color='#94A3B8', zorder=4,
                )

        elif s_type == 'tier_banner':
            tier_row_counter = 0
            meta = tier_meta[payload]
            count = len(grouped[payload])
            game_word = 'GAME' if count == 1 else 'GAMES'

            banner_bg = patches.Rectangle((0.0, row_y), 1.0, slot_h, facecolor=meta['bg'], edgecolor='none', zorder=2)
            ax.add_patch(banner_bg)
            accent_bar = patches.Rectangle((0.0, row_y), 0.006, slot_h, facecolor=meta['accent'], edgecolor='none', zorder=3)
            ax.add_patch(accent_bar)
            ax.plot([0, 1], [row_y + slot_h, row_y + slot_h], color=meta['border'], linewidth=1.1, zorder=3)
            ax.plot([0, 1], [row_y, row_y], color=meta['border'], linewidth=1.0, zorder=3)

            ax.text(
                0.016, row_mid_y, meta['title'],
                ha='left', va='center',
                fontsize=9.0, fontweight='bold', color=meta['accent'], zorder=4,
            )
            ax.text(
                0.984, row_mid_y, f"{count} {game_word}",
                ha='right', va='center',
                fontsize=8.2, fontweight='bold', color=meta['accent'], alpha=0.90, zorder=4,
            )

        elif s_type == 'game_row':
            tier_key, (_, g) = payload
            meta = tier_meta[tier_key]

            row_bg_color = meta['row_odd'] if (tier_row_counter % 2 == 1) else meta['row_even']
            row_patch = patches.Rectangle((0.0, row_y), 1.0, slot_h, facecolor=row_bg_color, edgecolor='none', zorder=2)
            ax.add_patch(row_patch)
            tier_row_counter += 1

            next_is_game = (idx + 1 < len(slots)) and (slots[idx + 1][0] == 'game_row')
            if next_is_game:
                ax.plot([0.008, 0.992], [row_y, row_y], color='#243047', linewidth=0.8, zorder=3)

            away_abbr = g['away_team']['abbr']
            home_abbr = g['home_team']['abbr']
            away_nick = get_team_nickname(away_abbr, g['away_team'].get('name'))
            home_nick = get_team_nickname(home_abbr, g['home_team'].get('name'))

            p = g['prediction']
            home_prob = float(p.get('model_home_win_prob', 0.5))
            vegas_home_prob = float(p.get('vegas_home_implied_prob', 0.5))
            pick_abbr = p.get('pick_abbr', home_abbr if home_prob >= 0.5 else away_abbr)
            pick_nick = get_team_nickname(pick_abbr, p.get('pick_name'))

            pick_is_home = (pick_abbr == home_abbr)
            blend_pick_pct = (home_prob if pick_is_home else (1.0 - home_prob)) * 100.0
            vegas_pick_pct = (vegas_home_prob if pick_is_home else (1.0 - vegas_home_prob)) * 100.0

            projected_margin = float(p.get('projected_margin', abs(p.get('calibrated_home_margin', 0.0))))
            home_score = float(p.get('home_projected_score', 0.0))
            away_score = float(p.get('away_projected_score', 0.0))
            total_pts = float(p.get('projected_total_points', home_score + away_score))

            raw_iso = g.get('gametime', 'TBD')
            compact_dt = format_mt_kickoff('', raw_iso, local_tz, compact=True) if raw_iso != 'TBD' else 'TBD'

            # 1. Match up
            c0_left = col_starts[0]
            c0_w = col_widths[0]
            ax.add_artist(AnnotationBbox(get_team_logo(away_abbr, max_px=40, zoom=0.45), (c0_left + 0.020, row_mid_y), frameon=False, zorder=4))
            ax.text(
                c0_left + c0_w / 2, row_mid_y, f"{away_nick}  @  {home_nick}",
                ha='center', va='center', fontsize=9.1, fontweight='bold', color='#F8FAFC', zorder=4,
            )
            ax.add_artist(AnnotationBbox(get_team_logo(home_abbr, max_px=40, zoom=0.45), (c0_left + c0_w - 0.020, row_mid_y), frameon=False, zorder=4))

            # 2. Date & Time
            ax.text(col_starts[1] + col_widths[1] / 2, row_mid_y, compact_dt,
                    ha='center', va='center', fontsize=9.0, color='#CBD5E1', zorder=4)

            # 3. Model Blend
            ax.text(col_starts[2] + col_widths[2] / 2, row_mid_y, f"{blend_pick_pct:.1f}%",
                    ha='center', va='center', fontsize=9.4, fontweight='bold', color='#38BDF8', zorder=4)

            # 4. Vegas
            ax.text(col_starts[3] + col_widths[3] / 2, row_mid_y, f"{vegas_pick_pct:.1f}%",
                    ha='center', va='center', fontsize=9.2, color='#94A3B8', zorder=4)

            # 5. Winner Pick
            ax.text(
                col_starts[4] + col_widths[4] / 2, row_mid_y, pick_nick,
                ha='center', va='center', fontsize=9.1, fontweight='bold', color='#ECFDF5',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#0F2E28', edgecolor='#10B981', linewidth=1.0),
                zorder=4,
            )

            # 6. Margin
            ax.text(col_starts[5] + col_widths[5] / 2, row_mid_y, f"+{projected_margin:.1f}",
                    ha='center', va='center', fontsize=9.4, fontweight='bold', color='#34D399', zorder=4)

            # 7. Home Score
            home_color = '#F8FAFC' if home_score >= away_score else '#94A3B8'
            ax.text(col_starts[6] + col_widths[6] / 2, row_mid_y, f"{home_score:.1f}",
                    ha='center', va='center', fontsize=9.4, fontweight='bold' if home_score >= away_score else 'normal',
                    color=home_color, zorder=4)

            # 8. Away Score
            away_color = '#F8FAFC' if away_score > home_score else '#94A3B8'
            ax.text(col_starts[7] + col_widths[7] / 2, row_mid_y, f"{away_score:.1f}",
                    ha='center', va='center', fontsize=9.4, fontweight='bold' if away_score > home_score else 'normal',
                    color=away_color, zorder=4)

            # 9. Total
            ax.text(col_starts[8] + col_widths[8] / 2, row_mid_y, f"{total_pts:.1f}",
                    ha='center', va='center', fontsize=9.4, fontweight='bold', color='#E2E8F0', zorder=4)

    plt.savefig(output_path, dpi=220, facecolor='#0B0F19')
    plt.close(fig)


def format_odds_label(odds_val, has_live_odds: bool):
    """Returns formatted American odds only when real sportsbook pricing exists (never fake -110)."""
    if odds_val is None or not has_live_odds:
        return None
    return f"{int(odds_val):+d}"


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

    main_color = '#ECFDF5' if is_pick else '#E2E8F0'
    main_y = y + h * (0.50 if odds_text is None else 0.64)
    ax.text(
        x + w / 2, main_y, main_text,
        ha='center', va='center',
        fontsize=11.6, fontweight='bold', color=main_color,
        transform=ax.transAxes, zorder=3,
    )

    if odds_text is not None:
        sub_color = '#6EE7B7' if is_pick else '#94A3B8'
        ax.text(
            x + w / 2, y + h * 0.25, odds_text,
            ha='center', va='center',
            fontsize=8.8, fontweight='normal', color=sub_color,
            transform=ax.transAxes, zorder=3,
        )

    if is_pick and edge_tag:
        ax.text(
            x + w - 0.012, y + h - 0.014, edge_tag,
            ha='right', va='top',
            fontsize=6.8, fontweight='bold', color='#052E16',
            bbox=dict(boxstyle='round,pad=0.16', facecolor='#34D399', edgecolor='none'),
            transform=ax.transAxes, zorder=4,
        )


def draw_custom_matchup_card(ax, game_obj: dict, kickoff_label: str):
    ax.set_facecolor('#0B0F19')
    ax.axis('off')

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
    away_full = FULL_TEAM_NAMES.get(away_abbr, game_obj['away_team'].get('name', away_abbr))
    home_full = FULL_TEAM_NAMES.get(home_abbr, game_obj['home_team'].get('name', home_abbr))

    away_city, away_nick = away_full.rsplit(' ', 1)
    home_city, home_nick = home_full.rsplit(' ', 1)

    p = game_obj['prediction']
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
        'High': ('#331D08', '#FBBF24'),
        'Moderate': ('#261844', '#C084FC'),
        'Low': ('#0A2E46', '#38BDF8'),
    }
    t_bg, t_fg = tier_palette.get(tier, ('#1E293B', '#94A3B8'))
    tier_text = 'MEDIUM' if tier.lower() in ('moderate', 'medium') else tier.upper()
    ax.text(
        0.955, 0.925, f"● {tier_text} CONFIDENCE",
        ha='right', va='center', fontsize=7.6, fontweight='bold', color=t_fg,
        bbox=dict(boxstyle='round,pad=0.25', facecolor=t_bg, edgecolor='none'),
        transform=ax.transAxes, zorder=3,
    )

    # --- SCOREBOARD HERO HEADER ---
    ax.text(0.50, 0.862, 'PROJECTED SCORE', ha='center', va='center',
            fontsize=7.6, fontweight='bold', color='#38BDF8', transform=ax.transAxes, zorder=3)

    ax.add_artist(AnnotationBbox(get_team_logo(away_abbr, max_px=74), (0.11, 0.745), frameon=False, xycoords='axes fraction', zorder=4))
    ax.text(0.195, 0.795, away_city.upper(), ha='left', va='center',
            fontsize=7.4, fontweight='bold', color='#94A3B8', transform=ax.transAxes, zorder=3)
    ax.text(0.195, 0.742, away_nick, ha='left', va='center',
            fontsize=12.0, fontweight='bold', color='#F8FAFC', transform=ax.transAxes, zorder=3)
    away_score_color = '#34D399' if away_score > home_score else '#CBD5E1'
    ax.text(0.195, 0.678, f"{away_score:.1f}", ha='left', va='center',
            fontsize=14.5, fontweight='bold', color=away_score_color, transform=ax.transAxes, zorder=3)

    ax.text(0.50, 0.735, '@', ha='center', va='center',
            fontsize=11.5, fontweight='bold', color='#475569', transform=ax.transAxes, zorder=3)

    ax.add_artist(AnnotationBbox(get_team_logo(home_abbr, max_px=74), (0.89, 0.745), frameon=False, xycoords='axes fraction', zorder=4))
    ax.text(0.805, 0.795, home_city.upper(), ha='right', va='center',
            fontsize=7.4, fontweight='bold', color='#94A3B8', transform=ax.transAxes, zorder=3)
    ax.text(0.805, 0.742, home_nick, ha='right', va='center',
            fontsize=12.0, fontweight='bold', color='#F8FAFC', transform=ax.transAxes, zorder=3)
    home_score_color = '#34D399' if home_score >= away_score else '#CBD5E1'
    ax.text(0.805, 0.678, f"{home_score:.1f}", ha='right', va='center',
            fontsize=14.5, fontweight='bold', color=home_score_color, transform=ax.transAxes, zorder=3)

    # --- WIN PROBABILITY SPLIT BAR ---
    bar_x, bar_y, bar_w, bar_h = 0.045, 0.580, 0.91, 0.024
    away_w = max(0.08, min(0.92, away_prob)) * bar_w
    home_w = bar_w - away_w

    away_bar_color = '#10B981' if away_prob > home_prob else '#334155'
    home_bar_color = '#10B981' if home_prob >= away_prob else '#334155'

    ax.add_patch(patches.Rectangle((bar_x, bar_y), away_w, bar_h, facecolor=away_bar_color, edgecolor='none', transform=ax.transAxes, zorder=3))
    ax.add_patch(patches.Rectangle((bar_x + away_w, bar_y), home_w, bar_h, facecolor=home_bar_color, edgecolor='none', transform=ax.transAxes, zorder=3))

    ax.text(bar_x, bar_y + 0.040, f"{away_abbr} {away_prob*100:.1f}%", ha='left', va='center',
            fontsize=7.6, fontweight='bold', color='#94A3B8', transform=ax.transAxes, zorder=3)
    ax.text(0.50, bar_y + 0.040, "WIN PROBABILITY", ha='center', va='center',
            fontsize=7.0, fontweight='bold', color='#64748B', transform=ax.transAxes, zorder=3)
    ax.text(bar_x + bar_w, bar_y + 0.040, f"{home_prob*100:.1f}% {home_abbr}", ha='right', va='center',
            fontsize=7.6, fontweight='bold', color='#94A3B8', transform=ax.transAxes, zorder=3)

    # --- 2-COLUMN MARKET MATRIX HEADERS (SPREAD & TOTAL ONLY) ---
    col_x = [0.130, 0.550]
    col_w = 0.405
    for cx, title in zip(col_x, ['SPREAD', 'TOTAL']):
        ax.text(cx + col_w / 2, 0.525, title, ha='center', va='center',
                fontsize=8.4, fontweight='bold', color='#64748B', transform=ax.transAxes, zorder=3)

    ax.add_artist(AnnotationBbox(get_team_logo(away_abbr, max_px=56), (0.078, 0.39), frameon=False, xycoords='axes fraction', zorder=4))
    ax.add_artist(AnnotationBbox(get_team_logo(home_abbr, max_px=56), (0.078, 0.165), frameon=False, xycoords='axes fraction', zorder=4))

    row_h = 0.185
    away_y, home_y = 0.298, 0.072

    has_live_odds = bool(mkt.get('odds_fetched_at'))
    away_spread = mkt.get('away_spread')
    home_spread = mkt.get('home_spread')
    away_sp_odds = format_odds_label(mkt.get('away_spread_odds'), has_live_odds)
    home_sp_odds = format_odds_label(mkt.get('home_spread_odds'), has_live_odds)

    total_line = mkt.get('total_line')
    over_odds = format_odds_label(mkt.get('over_odds'), has_live_odds)
    under_odds = format_odds_label(mkt.get('under_odds'), has_live_odds)

    ats_edge = p.get('ats_edge')
    tot_edge = p.get('total_edge')
    sp_tag = f"+{abs(ats_edge):.1f}" if ats_edge is not None else None
    tot_tag = f"+{abs(tot_edge):.1f}" if tot_edge is not None else None

    # 1. Spread Cells
    draw_market_cell(
        ax, col_x[0], away_y, col_w, row_h,
        f"{away_spread:+.1f}" if away_spread is not None else "N/A",
        away_sp_odds if away_spread is not None else None,
        is_pick=(p.get('spread_pick_side') == 'away'),
        edge_tag=sp_tag,
    )
    draw_market_cell(
        ax, col_x[0], home_y, col_w, row_h,
        f"{home_spread:+.1f}" if home_spread is not None else "N/A",
        home_sp_odds if home_spread is not None else None,
        is_pick=(p.get('spread_pick_side') == 'home'),
        edge_tag=sp_tag,
    )

    # 2. Total Cells
    draw_market_cell(
        ax, col_x[1], away_y, col_w, row_h,
        f"O {total_line:.1f}" if total_line is not None else "N/A",
        over_odds if total_line is not None else None,
        is_pick=(p.get('total_pick_side') == 'over'),
        edge_tag=tot_tag,
    )
    draw_market_cell(
        ax, col_x[1], home_y, col_w, row_h,
        f"U {total_line:.1f}" if total_line is not None else "N/A",
        under_odds if total_line is not None else None,
        is_pick=(p.get('total_pick_side') == 'under'),
        edge_tag=tot_tag,
    )


def render_detailed_cards(all_games: list, week_num: int, scale_factor: float, output_path: str):
    """Renders the detailed 2-column matchup card board PNG (Spread & Total only)."""
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
    plt.savefig(output_path, dpi=220, facecolor='#0B0F19')
    plt.close(fig)


# --- LOAD DATA & RENDER BOTH PNGS ---
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

simple_image_path = os.path.join(PREDICTIONS_DIR, f'nfl_predictions_week_{week_num}_simple.png')
detailed_image_path = os.path.join(PREDICTIONS_DIR, f'nfl_predictions_week_{week_num}.png')

render_simple_table(all_games, week_num, scale_factor, local_tz, simple_image_path)
render_detailed_cards(all_games, week_num, scale_factor, detailed_image_path)

print(f'Generated simple summary table at: {os.path.abspath(simple_image_path)}')
print(f'Generated detailed matchup cards at: {os.path.abspath(detailed_image_path)}')