"""
Streamlit UI for the shipping cost predictor.

Usage:
    streamlit run app.py
"""

import datetime
import streamlit as st

from data_prep import load_item_dims
from predict import predict_options_multi

SHIP_FROM_LOCATIONS = ['KT PA', 'KT PHX', 'KT New KC']

st.set_page_config(page_title='Shipping Cost Predictor', page_icon='📦', layout='centered')

CSS = """
<style>
:root {
  --bg-page: #FFFFFF;
  --bg-surface: #FFFFFF;
  --bg-surface-alt: #F8FAFC;
  --border: #E2E8F0;
  --border-strong: #CBD5E1;
  --text-primary: #0F172A;
  --text-secondary: #334155;
  --text-muted: #64748B;
  --text-faint: #94A3B8;
  --accent: #6366F1;
  --accent-hover: #4F46E5;
  --accent-glow: rgba(99, 102, 241, 0.18);
  --danger: #DC2626;
  --danger-bg: #FEE2E2;
  --insight-bg: #F0FDFA;
  --insight-border: #99F6E4;
  --insight-text: #115E59;
  --radius-sm: 8px;
  --radius-lg: 12px;
}

#MainMenu, footer, [data-testid="stHeader"] { visibility: hidden; height: 0; }

html, body, [class*="css"] { font-variant-numeric: tabular-nums; }

[data-testid="stAppViewContainer"], .stApp, body {
  background: var(--bg-page) !important;
}

.block-container {
  max-width: 720px;
  padding-top: 2.75rem;
  padding-bottom: 3rem;
}

h1 {
  font-size: 1.6rem !important;
  font-weight: 700 !important;
  color: var(--text-primary) !important;
  letter-spacing: -0.01em;
  margin-bottom: 0.15rem !important;
}

[data-testid="stCaptionContainer"] {
  color: var(--text-muted) !important;
  font-size: 0.9rem !important;
  margin-bottom: 1.75rem !important;
}

.section-label {
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: var(--text-muted);
  margin: 0.9rem 0 0.5rem 0;
}
.section-label:first-child { margin-top: 0; }

/* Outer form card */
div[data-testid="stVerticalBlockBorderWrapper"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  background: var(--bg-surface) !important;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
div[data-testid="stVerticalBlockBorderWrapper"] > div {
  padding: 0.35rem 0.15rem;
}

/* Nested item-row cards read as a lighter, tighter grouping than the outer card */
div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlockBorderWrapper"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  background: var(--bg-surface-alt) !important;
  box-shadow: none !important;
  margin-bottom: 0.55rem;
}
div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlockBorderWrapper"] > div {
  padding: 0.3rem 0.3rem;
}

/* Text / number / date inputs */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input {
  border-radius: var(--radius-sm) !important;
  border: 1px solid var(--border) !important;
  background: var(--bg-page) !important;
  color: var(--text-primary) !important;
}
[data-testid="stTextInput"] input::placeholder { color: var(--text-faint) !important; }
div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stNumberInput"] input {
  background: var(--bg-page) !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus,
[data-testid="stDateInput"] input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-glow) !important;
}
[data-testid="stNumberInput"] button {
  border-color: var(--border) !important;
  background: var(--bg-page) !important;
  border-radius: var(--radius-sm) !important;
}

/* Selectbox (BaseWeb under the hood) */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
  border-radius: var(--radius-sm) !important;
  border: 1px solid var(--border) !important;
  background: var(--bg-page) !important;
  color: var(--text-primary) !important;
}
[data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within > div {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-glow) !important;
}
[data-testid="stSelectbox"] svg { fill: var(--text-muted) !important; }

/* Buttons */
[data-testid="stButton"] button {
  border-radius: var(--radius-sm) !important;
  font-weight: 500 !important;
  transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease;
}
[data-testid="stButton"] button[kind="primary"] {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
  color: #FFFFFF !important;
  font-weight: 600 !important;
  padding: 0.6rem 1rem !important;
}
[data-testid="stButton"] button[kind="primary"]:hover {
  background: var(--accent-hover) !important;
  border-color: var(--accent-hover) !important;
}
[data-testid="stButton"] button[kind="secondary"] {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-secondary) !important;
}
[data-testid="stButton"] button[kind="secondary"]:hover {
  border-color: var(--border-strong) !important;
  background: var(--bg-surface-alt) !important;
  color: var(--text-primary) !important;
}

/* Remove-row (X) buttons: nested two card-levels deep -> ghost icon style */
div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stButton"] button {
  border: none !important;
  background: transparent !important;
  color: var(--text-faint) !important;
  padding: 0.3rem 0.5rem !important;
  min-height: 2.4rem;
}
div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stButton"] button:hover:not(:disabled) {
  color: var(--danger) !important;
  background: var(--danger-bg) !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stButton"] button:disabled {
  color: var(--border-strong) !important;
}

[data-testid="stCheckbox"] { margin-top: 0.25rem; }
[data-testid="stCheckbox"] label p { color: var(--text-secondary) !important; }

/* Validation errors */
[data-testid="stAlertContentError"], div[data-testid="stAlert"] {
  border-radius: var(--radius-sm) !important;
}

/* ---- Results ---- */
.results-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  margin-top: 1.5rem;
}
.mode-card {
  position: relative;
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  background: var(--bg-surface);
  padding: 1.3rem 1.25rem 1.1rem;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
  display: flex;
  flex-direction: column;
  gap: 0.7rem;
  min-height: 100%;
}
.mode-card.recommended {
  border-color: var(--accent);
  box-shadow: 0 0 0 1px var(--accent), 0 6px 16px rgba(99, 102, 241, 0.14);
}
.mode-card .badge {
  position: absolute;
  top: -0.65rem;
  right: 1.1rem;
  background: var(--accent);
  color: #FFFFFF;
  font-size: 0.62rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  padding: 0.22rem 0.6rem;
  border-radius: 999px;
}
.mode-label {
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-muted);
}
.headline-price {
  font-size: 2.1rem;
  font-weight: 700;
  color: var(--text-primary);
  line-height: 1.1;
}
.headline-tier {
  font-size: 0.75rem;
  color: var(--text-faint);
  margin-top: -0.5rem;
}
.tier-list {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
  border-top: 1px solid var(--border);
  padding-top: 0.65rem;
  margin-top: 0.1rem;
}
.tier-row {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  font-size: 0.85rem;
  color: var(--text-muted);
}
.tier-row .tier-price {
  font-weight: 600;
  color: var(--text-secondary);
}
.steer-wrap {
  flex-grow: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 1.5rem 0;
}
.steer {
  font-size: 1rem;
  font-weight: 600;
  color: var(--text-faint);
}

.rec-banner {
  display: flex;
  gap: 0.65rem;
  align-items: flex-start;
  background: var(--insight-bg);
  border: 1px solid var(--insight-border);
  border-radius: var(--radius-lg);
  padding: 0.9rem 1.05rem;
  margin-top: 1rem;
  font-size: 0.85rem;
  color: var(--insight-text);
  line-height: 1.5;
}
.rec-banner .icon { flex-shrink: 0; font-size: 1.05rem; line-height: 1.4; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def known_items():
    """All item_ids in the item master (Item Unit Dims and Cartons tab), for a searchable
    dropdown instead of free text. Items without shipment history still price fine —
    predict.py falls back to global median weight/cbft for them."""
    return sorted(load_item_dims().index)


def render_results(result: dict, recommendation: dict):
    """Build the PARCEL/LTL side-by-side comparison as one HTML block."""
    parcel, ltl = result['PARCEL'], result['LTL']

    if isinstance(parcel, str):
        parcel_best = False   # PARCEL is flagged -> LTL is the real option
    elif isinstance(ltl, str):
        parcel_best = True    # LTL is flagged -> PARCEL is the real option
    else:
        parcel_best = parcel['Ground'] <= ltl['Ground']

    def card(mode_label, tiers, is_best):
        classes = 'mode-card recommended' if is_best else 'mode-card'
        badge = '<div class="badge">Recommended</div>' if is_best else ''
        if isinstance(tiers, str):
            body = f'<div class="steer-wrap"><div class="steer">{tiers}</div></div>'
        else:
            headline_tier, headline_price = next(iter(tiers.items()))
            rest = list(tiers.items())[1:]
            rows = ''.join(
                f'<div class="tier-row"><span>{tier}</span>'
                f'<span class="tier-price">${cost:,.2f}</span></div>'
                for tier, cost in rest
            )
            tier_list = f'<div class="tier-list">{rows}</div>' if rows else ''
            body = (
                f'<div class="headline-price">${headline_price:,.2f}</div>'
                f'<div class="headline-tier">{headline_tier}</div>'
                f'{tier_list}'
            )
        return f'<div class="{classes}">{badge}<div class="mode-label">{mode_label}</div>{body}</div>'

    html = '<div class="results-grid">'
    html += card('PARCEL', parcel, parcel_best is True)
    html += card('LTL', ltl, parcel_best is False)
    html += '</div>'

    if recommendation['flag']:
        html += (
            '<div class="rec-banner"><span class="icon">💡</span>'
            f'<span>{recommendation["message"]}</span></div>'
        )

    st.markdown(html, unsafe_allow_html=True)


RESETTABLE_PREFIXES = ('item_', 'qty_', 'remove_')
RESETTABLE_KEYS = ('ship_from', 'ship_to_zip', 'specify_date', 'ship_date_input')


def clear_all():
    for key in list(st.session_state.keys()):
        if key.startswith(RESETTABLE_PREFIXES) or key in RESETTABLE_KEYS:
            del st.session_state[key]
    st.session_state.line_items = [{'item_id': None, 'qty': 1}]


if 'line_items' not in st.session_state:
    st.session_state.line_items = [{'item_id': None, 'qty': 1}]

st.title('Shipping Cost Predictor')
st.caption('Estimate PARCEL vs. LTL cost across speed tiers for one or more line items.')

item_options = known_items()

with st.container(border=True):
    st.markdown('<div class="section-label">Shipment</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        ship_from = st.selectbox('Ship from', SHIP_FROM_LOCATIONS, key='ship_from')
    with col2:
        ship_to_zip = st.text_input(
            'Ship to zip', max_chars=10, placeholder='e.g. 10001', key='ship_to_zip',
        )

    ship_date = None
    if st.checkbox('Specify a ship date (defaults to today)', key='specify_date'):
        ship_date = st.date_input(
            'Ship date', value=datetime.date.today(),
            label_visibility='collapsed', key='ship_date_input',
        )

    st.markdown('<div class="section-label">Line items</div>', unsafe_allow_html=True)

    for i, line in enumerate(st.session_state.line_items):
        with st.container(border=True):
            row = st.columns([5, 2, 1], vertical_alignment='center')
            line['item_id'] = row[0].selectbox(
                'Item', item_options, index=None, placeholder='Search item ID...',
                key=f'item_{i}', label_visibility='collapsed',
            )
            line['qty'] = row[1].number_input(
                'Qty', min_value=1, value=line['qty'], step=1,
                key=f'qty_{i}', label_visibility='collapsed',
            )
            if row[2].button('✕', key=f'remove_{i}', disabled=len(st.session_state.line_items) == 1):
                st.session_state.line_items.pop(i)
                st.rerun()

    if st.button('+ Add another item'):
        st.session_state.line_items.append({'item_id': None, 'qty': 1})
        st.rerun()

clear_col, predict_col = st.columns([1, 3])
with clear_col:
    st.button('Clear', use_container_width=True, on_click=clear_all)
with predict_col:
    predict_clicked = st.button('Predict cost', type='primary', use_container_width=True)

if predict_clicked:
    items = [(line['item_id'], line['qty']) for line in st.session_state.line_items]
    missing = [i + 1 for i, (item_id, _) in enumerate(items) if not item_id]
    if not ship_to_zip.strip():
        st.error('Enter a destination zip.')
    elif missing:
        st.error(f'Select an item for line {", ".join(map(str, missing))}.')
    else:
        with st.spinner('Predicting...'):
            result = predict_options_multi(ship_from, ship_to_zip.strip(), items, ship_date=ship_date)
        recommendation = result.pop('recommendation')
        render_results(result, recommendation)
