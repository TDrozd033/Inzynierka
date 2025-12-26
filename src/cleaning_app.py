import pandas as pd
import numpy as np
import glob
from pathlib import Path

# ============================================================
# KONFIGURACJA KOLUMN (1:1 z notebooka)
# ============================================================

CORE_COLS = [
    'Div', 'Date', 'HomeTeam', 'AwayTeam',
    'FTHG', 'FTAG', 'FTR',
    'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC',
    'HY', 'AY', 'HR', 'AR',
    'B365H', 'B365D', 'B365A',
    'PSH', 'PSD', 'PSA',
    'BbMxH', 'BbMxD', 'BbMxA',
    'BbAvH', 'BbAvD', 'BbAvA',
    'Bb1X2'
]

NUMERIC_COLS = [
    "FTHG","FTAG","HTHG","HTAG",
    "HS","AS","HST","AST","HF","AF","HC","AC","HY","AY","HR","AR",
    "B365H","B365D","B365A",
    "PSH","PSD","PSA",
    "BbMxH","BbMxD","BbMxA",
    "BbAvH","BbAvD","BbAvA",
    "Bb1X2"
]

TEAM_ALIAS = {
    # === PREMIER LEAGUE ===
    "Man United": "Manchester United",
    "Man Utd": "Manchester United",
    "Man City": "Manchester City",
    "Spurs": "Tottenham",
    "Wolves": "Wolverhampton",
    "Newcastle Utd": "Newcastle",
    "Brighton and Hove Albion": "Brighton",
    "Brighton & Hove Albion": "Brighton",

    # === SERIE A ===
    "Inter": "Inter Milan",
    "Internazionale": "Inter Milan",
    "AC Milan": "Milan",
    "AS Roma": "Roma",
    "Hellas Verona": "Verona",

    # === LA LIGA ===
    "Ath Madrid": "Atletico Madrid",
    "Ath Bilbao": "Athletic Bilbao",
    "Betis": "Real Betis",
    "La Coruna": "Deportivo La Coruna",

    # === BUNDESLIGA ===
    "Bayern Munich": "Bayern Munich",
    "RB Leipzig": "RB Leipzig",
    "B M'gladbach": "Borussia Mönchengladbach",
    "B Dortmund": "Borussia Dortmund",

    # === LIGUE 1 ===
    "PSG": "Paris Saint-Germain",
    "Paris SG": "Paris Saint-Germain",
    "St Etienne": "Saint-Etienne",
    "Nimes": "Nîmes",
}

# ============================================================
# FUNKCJE POMOCNICZE
# ============================================================

def parse_date(df, col="Date"):
    df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    return df


def add_season_from_date(df):
    y = pd.to_numeric(df["Date"].dt.year, errors="coerce")
    m = df["Date"].dt.month

    season_start = np.where(m >= 7, y, y - 1)
    season_start = pd.Series(season_start).fillna(0).astype(int)

    df["Season"] = season_start.astype(str) + "/" + (season_start + 1).astype(str)
    return df


def normalize_team_names(df):
    for col in ["HomeTeam", "AwayTeam"]:
        df[col] = df[col].replace(TEAM_ALIAS)
    return df


def standardize_numeric(df):
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ============================================================
# CLEANING JEDNEJ LIGI
# ============================================================

def load_and_clean_league(league_dir: Path, league_name: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(league_dir / "*.csv")))
    dfs = []

    for path in files:
        df = pd.read_csv(path)

        df = parse_date(df)
        df = normalize_team_names(df)
        df = standardize_numeric(df)

        for c in CORE_COLS:
            if c not in df.columns:
                df[c] = np.nan

        df = df[CORE_COLS]
        df["SourceFile"] = Path(path).name
        df = add_season_from_date(df)
        df["League"] = league_name

        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    df_all = (
        df_all
        .dropna(subset=["Date", "HomeTeam", "AwayTeam"])
        .sort_values("Date")
        .drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"])
        .reset_index(drop=True)
    )

    if "FTR" in df_all.columns:
        df_all["FTR"] = df_all["FTR"].astype("category")

    return df_all


# ============================================================
# CLEANING WSZYSTKICH LIG (POD APLIKACJĘ)
# ============================================================

def load_all_leagues(raw_base_dir: Path) -> pd.DataFrame:
    league_dirs = {
        "premier_league": raw_base_dir / "premier_league",
        "serie_a": raw_base_dir / "serie_a",
        "la_liga": raw_base_dir / "la_liga",
        "bundesliga": raw_base_dir / "bundesliga",
        "ligue_1": raw_base_dir / "ligue_1",
    }

    all_dfs = []

    for league, path in league_dirs.items():
        print(f" Ładowanie ligi: {league}")
        df_league = load_and_clean_league(path, league)
        all_dfs.append(df_league)

    return pd.concat(all_dfs, ignore_index=True)
