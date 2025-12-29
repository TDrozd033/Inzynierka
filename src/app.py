# src/app.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

from src.football_data_client import LEAGUES



STANDINGS_DIR = PROJECT_ROOT / "data_app" / "standings"
PREDICTIONS_DIR = PROJECT_ROOT / "data_app" / "predictions"

TABLE_PATHS = {
    "premier_league": STANDINGS_DIR / "premier_league_table.csv",
    "la_liga": STANDINGS_DIR / "la_liga_table.csv",
    "serie_a": STANDINGS_DIR / "serie_a_table.csv",
    "bundesliga": STANDINGS_DIR / "bundesliga_table.csv",
    "ligue_1": STANDINGS_DIR / "ligue_1_table.csv",
}

PRED_PATHS = {
    "premier_league": PREDICTIONS_DIR / "premier_league_next_matchday_predictions.csv",
    "la_liga": PREDICTIONS_DIR / "la_liga_next_matchday_predictions.csv",
    "serie_a": PREDICTIONS_DIR / "serie_a_next_matchday_predictions.csv",
    "bundesliga": PREDICTIONS_DIR / "bundesliga_next_matchday_predictions.csv",
    "ligue_1": PREDICTIONS_DIR / "ligue_1_next_matchday_predictions.csv",
}

LEAGUE_LABELS = {
    "premier_league": "Premier League",
    "la_liga": "La Liga",
    "serie_a": "Serie A",
    "bundesliga": "Bundesliga",
    "ligue_1": "Ligue 1",
}

COLORS = {
    "ucl_bg": "#E8F1FB",
    "ucl_border": "#1F7AE0",
    "ucl_text": "#0A2540",
    "uel_bg": "#FFF4E5",
    "uel_border": "#F59E0B",
    "uel_text": "#7C2D12",
    "uecl_bg": "#ECFDF5",
    "uecl_border": "#22C55E",
    "uecl_text": "#14532D",
    "playoff_bg": "#F3F4F6",
    "playoff_border": "#9CA3AF",
    "playoff_text": "#374151",
    "rel_bg": "#FEE2E2",
    "rel_border": "#DC2626",
    "rel_text": "#7F1D1D",
}

LEAGUE_ZONES = {
    "premier_league": {"ucl": range(1, 5), "uel": [5], "relegation": range(18, 21)},
    "la_liga": {"ucl": range(1, 5), "uel": [5], "uecl": [6], "relegation": range(18, 21)},
    "serie_a": {"ucl": range(1, 5), "uel": [5], "uecl": [6], "relegation": range(18, 21)},
    "bundesliga": {"ucl": range(1, 5), "uel": [5], "uecl": [6], "playoff": [16], "relegation": range(17, 19)},
    "ligue_1": {"ucl": range(1, 4), "uel": [4], "uecl": [5], "playoff": [16], "relegation": range(17, 19)},
}

LEAGUE_CENTER_COLS = ["Played", "Won", "Draw", "Lost", "GoalsFor", "GoalsAgainst", "GoalDiff", "Points"]
PRED_CENTER_COLS = ["Confidence", "P(H)", "P(D)", "P(A)"]

# UI / STYLE

st.set_page_config(page_title="Football Match Predictor", layout="wide")

st.markdown(
    """
    <style>
    div.stButton > button {
        width:100%;
        border-radius:10px;
        height:42px;
        font-weight:600;
        border:none;
    }

    @media (prefers-color-scheme: dark) {
        div.stButton > button {
            background:#020617;
            color:#E5E7EB;
        }
        div.stButton > button:hover {
            filter:brightness(1.2);
        }
    }

    @media (prefers-color-scheme: light) {
        div.stButton > button {
            background:#111827;
            color:white;
        }
        div.stButton > button:hover {
            background:#1F2937;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<hr style='margin:24px 0; border:none; height:1px; background:rgba(148,163,184,0.3);'>",
    unsafe_allow_html=True,
)

st.markdown("<h1 style='text-align:center;'>⚽ Football Match Predictor</h1>", unsafe_allow_html=True)
st.markdown(
    "<h3 style='text-align:center;'>Predictions of the next round's results - TOP 5 European leagues</h3>",
    unsafe_allow_html=True,
)

tabs = st.tabs([LEAGUE_LABELS[i] for i in LEAGUES.keys()])

for league_key in LEAGUES.keys():
    st.session_state.setdefault(f"show_table_{league_key}", False)
    st.session_state.setdefault(f"show_pred_{league_key}", False)


# FUNKCJE


def card_start():
    st.markdown(
        """
        <style>
        @media (prefers-color-scheme: dark) {
            .card { background:#020617; color:#E5E7EB; }
        }
        @media (prefers-color-scheme: light) {
            .card { background:white; color:#111827; }
        }
        </style>
        <div class="card" style="
            padding:16px;
            border-radius:14px;
            box-shadow:0 6px 18px rgba(0,0,0,0.06);
            margin-bottom:16px;
        ">
        """,
        unsafe_allow_html=True,
    )

def card_end():
    st.markdown("</div>", unsafe_allow_html=True)

def dataframe_height(df, row_height=35, header_height=40, max_height=900):
    return min(header_height + row_height * len(df), max_height)

def pred_pretty(row):
    if row["Pred"] == "H":
        return row["HomeTeam"]
    if row["Pred"] == "A":
        return row["AwayTeam"]
    return "Draw"

def highlight_league(row, league_key):
    pos = row.name
    zones = LEAGUE_ZONES.get(league_key, {})

    def apply(bg, border, text):
        s = [f"background-color:{bg}; color:{text};" for _ in row]
        s[0] += f"border-left:6px solid {border};"
        return s

    if pos in zones.get("ucl", []):
        return apply(COLORS["ucl_bg"], COLORS["ucl_border"], COLORS["ucl_text"])
    if pos in zones.get("uel", []):
        return apply(COLORS["uel_bg"], COLORS["uel_border"], COLORS["uel_text"])
    if pos in zones.get("uecl", []):
        return apply(COLORS["uecl_bg"], COLORS["uecl_border"], COLORS["uecl_text"])
    if pos in zones.get("playoff", []):
        return apply(COLORS["playoff_bg"], COLORS["playoff_border"], COLORS["playoff_text"])
    if pos in zones.get("relegation", []):
        return apply(COLORS["rel_bg"], COLORS["rel_border"], COLORS["rel_text"])

    return [""] * len(row)

def highlight_goal_diff(val):
    if val > 0:
        return "color:#15803D; font-weight:600;"
    if val < 0:
        return "color:#B91C1C; font-weight:600;"
    return ""

def table_styles():
    return [
        {"selector": "th", "props": [("font-weight", "600")]},
        {"selector": "td", "props": [("padding", "6px 10px"), ("font-size", "14px")]},
    ]

def league_legend(league_key):
    zones = LEAGUE_ZONES.get(league_key, {})
    legend = []
    if "ucl" in zones: legend.append(("#1F7AE0", "Champions League"))
    if "uel" in zones: legend.append(("#F59E0B", "Europa League"))
    if "uecl" in zones: legend.append(("#22C55E", "Europa Conference League"))
    if "playoff" in zones: legend.append(("#9CA3AF", "Relegation play-off"))
    if "relegation" in zones: legend.append(("#DC2626", "Relegation"))

    cols = st.columns(len(legend))
    for col, (color, label) in zip(cols, legend):
        with col:
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="width:14px;height:14px;background:{color};border-radius:3px;"></span>
                    {label}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ZAWARTOŚĆ


for tab, league_key in zip(tabs, LEAGUES.keys()):
    with tab:
        st.header(league_key.replace("_", " ").title())
        col1, col2 = st.columns(2)

        # ---------- TABELA ----------
        with col1:
            card_start()

            if st.button(" League table preview", key=f"table_{league_key}"):
                st.session_state[f"show_table_{league_key}"] = not st.session_state[f"show_table_{league_key}"]

            if st.session_state[f"show_table_{league_key}"]:
                df = pd.read_csv(TABLE_PATHS[league_key])
                df.index += 1

                styled = (
                    df.style
                    .apply(lambda r: highlight_league(r, league_key), axis=1)
                    .applymap(highlight_goal_diff, subset=["GoalDiff"])
                    .set_properties(subset=LEAGUE_CENTER_COLS, **{"text-align": "center"})
                    .set_table_styles(table_styles())
                )

                st.dataframe(styled, use_container_width=True, hide_index=True, height=dataframe_height(df))
                league_legend(league_key)

            card_end()

        # ---------- PREDYKCJE ----------
        with col2:
            card_start()

            if st.button(" Predictions for the next round", key=f"pred_{league_key}"):
                st.session_state[f"show_pred_{league_key}"] = not st.session_state[f"show_pred_{league_key}"]

            if st.session_state[f"show_pred_{league_key}"]:
                df = pd.read_csv(PRED_PATHS[league_key])

                df["Match"] = df["HomeTeam"] + " vs " + df["AwayTeam"]
                df["Prediction"] = df.apply(pred_pretty, axis=1)
                df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%d-%m-%Y %H:%M")

                view = df[["Date", "Match", "Prediction", "Confidence", "P(H)", "P(D)", "P(A)"]]
                view.index += 1

                for c in ["Confidence", "P(H)", "P(D)", "P(A)"]:
                    view[c] = (view[c] * 100).round(1).astype(str) + "%"

                styled = (
                    view.style
                    .set_properties(subset=PRED_CENTER_COLS, **{"text-align": "center"})
                    .set_table_styles(table_styles())
                )

                st.dataframe(styled, use_container_width=True, height=dataframe_height(view))

            card_end()
