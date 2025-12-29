from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from src.football_data_client import get_future_fixtures, LEAGUES



PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_HISTORY_PATH = PROJECT_ROOT / "data_app" / "processed" / "all_leagues" / "all_leagues_features.csv"
OUT_DIR = PROJECT_ROOT / "data_app" / "prepared"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MANUAL_MAP = {
    "Alaves": "Deportivo Alavés",
    "Angers": "Angers SCO",
    "Arsenal": "Arsenal FC",
    "Aston Villa": "Aston Villa FC",
    "Atalanta": "Atalanta BC",
    "Athletic Bilbao": "Athletic Club",
    "Atletico Madrid": "Club Atlético de Madrid",
    "Augsburg": "FC Augsburg",
    "Auxerre": "AJ Auxerre",
    "Barcelona": "FC Barcelona",
    "Bayern Munich": "FC Bayern München",
    "Bologna": "Bologna FC 1909",
    "Bournemouth": "AFC Bournemouth",
    "Brentford":"Brentford FC",
    "Brest": "Stade Brestois 29",
    "Brighton": "Brighton & Hove Albion FC",
    "Burnley": "Burnley FC",
    "Cagliari": "Cagliari Calcio",
    "Celta": "RC Celta de Vigo",
    "Chelsea": "Chelsea FC",
    "Como": "Como 1907",
    "Cremonese": "US Cremonese",
    "Crystal Palace": "Crystal Palace FC",
    "Dortmund": "Borussia Dortmund",
    "Ein Frankfurt": "Eintracht Frankfurt",
    "Elche": "Elche CF",
    "Espanol": "RCD Espanyol de Barcelona",
    "Everton": "Everton FC",
    "FC Koln": "1. FC Köln",
    "Fiorentina": "ACF Fiorentina",
    "Freiburg": "SC Freiburg",
    "Fulham": "Fulham FC",
    "Genoa": "Genoa CFC",
    "Getafe": "Getafe CF",
    "Girona": "Girona FC",
    "Hamburg": "Hamburger SV",
    "Heidenheim":"1. FC Heidenheim 1846",
    "Hoffenheim": "TSG 1899 Hoffenheim",
    "Inter Milan": "FC Internazionale Milano",
    "Juventus": "Juventus FC",
    "Lazio": "SS Lazio",
    "Le Havre": "Le Havre AC",
    "Lecce": "US Lecce",
    "Leeds": "Leeds United FC",
    "Lens": "Racing Club de Lens",
    "Levante": "Levante UD",
    "Leverkusen": "Bayer 04 Leverkusen",
    "Lille": "Lille OSC",
    "Liverpool": "Liverpool FC",
    "Lorient": "FC Lorient",
    "Lyon": "Olympique Lyonnais",
    "M'gladbach": "Borussia Mönchengladbach",
    "Mainz": "1. FSV Mainz 05",
    "Mallorca": "RCD Mallorca",
    "Manchester City": "Manchester City FC",
    "Manchester United": "Manchester United FC",
    "Marseille": "Olympique de Marseille",
    "Metz": "FC Metz",
    "Milan": "AC Milan",
    "Monaco": "AS Monaco FC",
    "Nantes": "FC Nantes",
    "Napoli": "SSC Napoli",
    "Newcastle": "Newcastle United FC",
    "Nice": "OGC Nice",
    "Nott'm Forest": "Nottingham Forest FC",
    "Osasuna": "CA Osasuna",
    "Oviedo": "Real Oviedo",
    "Paris FC": "Paris FC",
    "Paris Saint-Germain": "Paris Saint-Germain FC",
    "Parma": "Parma Calcio 1913",
    "Pisa": "AC Pisa 1909",
    "RB Leipzig": "RB Leipzig",
    "Real Betis": "Real Betis Balompié",
    "Real Madrid": "Real Madrid CF",
    "Rennes": "Stade Rennais FC 1901",
    "Roma": "AS Roma",
    "Sassuolo": "US Sassuolo Calcio",
    "Sevilla": "Sevilla FC",
    "Sociedad": "Real Sociedad de Fútbol",
    "St Pauli": "FC St. Pauli 1910",
    "Strasbourg": "RC Strasbourg Alsace",
    "Stuttgart": "VfB Stuttgart",
    "Sunderland": "Sunderland AFC",
    "Torino": "Torino FC",
    "Tottenham": "Tottenham Hotspur FC",
    "Toulouse": "Toulouse FC",
    "Udinese": "Udinese Calcio",
    "Union Berlin": "1. FC Union Berlin",
    "Valencia": "Valencia CF",
    "Vallecano": "Rayo Vallecano de Madrid",
    "Verona": "Hellas Verona FC",
    "Villarreal": "Villarreal CF",
    "Werder Bremen": "SV Werder Bremen",
    "West Ham": "West Ham United FC",
    "Wolfsburg": "VfL Wolfsburg",
    "Wolverhampton": "Wolverhampton Wanderers FC"
}



@dataclass(frozen=True)
class LeagueConfig:
    key: str           
    code: str          


def _ensure_datetime_utc(series: pd.Series) -> pd.Series:
    """Zamienia tekst ISO z API (utcDate) na datetime (UTC)."""
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    return dt





def _add_calendar_cols(fixtures: pd.DataFrame) -> pd.DataFrame:
    """
    Dodaje Month/Weekday/IsWeekend tak jak w FE.
    Zwraca kopię.
    """
    out = fixtures.copy()
    out["Month"] = out["Date"].dt.month.astype("int16")
    out["Weekday"] = out["Date"].dt.weekday.astype("int16")
    out["IsWeekend"] = out["Weekday"].isin([5, 6]).astype("int8")
    return out


def _pick_latest_team_snapshot(history, league, team_col, team_name, asof_date):
    # FIX: tz-aware vs tz-naive
    if isinstance(asof_date, pd.Timestamp) and asof_date.tzinfo is not None:
        asof_date = asof_date.tz_localize(None)

    history = history.copy()
    history["Date"] = pd.to_datetime(history["Date"]).dt.tz_localize(None)

    mask = (
        (history["League"] == league)
        & (history[team_col] == team_name)
        & (history["Date"] < asof_date)
    )

    subset = history.loc[mask].sort_values("Date")

    if subset.empty:
        return None

    return subset.iloc[-1]



def _build_features_for_fixture_row(
    history: pd.DataFrame,
    row: pd.Series,
    feature_cols: list[str],
) -> dict:
    """
    Dla pojedynczego meczu z API buduje wektor cech:
    - bierze ostatni snapshot Home_* dla gospodarza
    - bierze ostatni snapshot Away_* dla gościa
    - uzupełnia Month/Weekday/IsWeekend
    - pilnuje, żeby zwrócić tylko feature_cols (to potem pójdzie do modelu)
    """
    league = row["League"]
    match_date = row["Date"]

    home = row["HomeTeam"]
    away = row["AwayTeam"]

    home_snap = _pick_latest_team_snapshot(history, league, "HomeTeam", home, match_date)
    away_snap = _pick_latest_team_snapshot(history, league, "AwayTeam", away, match_date)

    features = {}

    features["Month"] = int(row["Month"])
    features["Weekday"] = int(row["Weekday"])
    features["IsWeekend"] = int(row["IsWeekend"])
    features["IsHomeAlways1"] = 1

    if home_snap is not None:
        for c in home_snap.index:
            if c.startswith("Home_") or c.startswith("FormDiff_") or c.startswith("SeasonStrength_"):
                features[c] = home_snap[c]

    if away_snap is not None:
        for c in away_snap.index:
            if c.startswith("Away_") or c.startswith("FormDiff_") or c.startswith("SeasonStrength_"):
                features[c] = away_snap[c]

    final = {c: features.get(c, pd.NA) for c in feature_cols}
    return final


def load_history_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    if "League" not in df.columns:
        raise ValueError("Brak kolumny 'League' w all_leagues_features.csv. Upewnij się, że FE było robione na all_leagues.")
    return df


def get_next_matchday_fixtures_for_league(league_key: str, limit: int = 40) -> pd.DataFrame:
    """
    Pobiera przyszłe mecze i wybiera NAJBLIŻSZĄ KOLEJKĘ (matchday) dla danej ligi.
    """
    code = LEAGUES[league_key]
    fx = get_future_fixtures(code, limit=limit)

    if fx.empty:
        return fx

    fx["Date"] = _ensure_datetime_utc(fx["Date"])

    API_TO_HIST_MAP = {v: k for k, v in MANUAL_MAP.items()}

    fx["HomeTeam"] = fx["HomeTeam"].map(API_TO_HIST_MAP)
    fx["AwayTeam"] = fx["AwayTeam"].map(API_TO_HIST_MAP)
    next_md = int(fx["Matchday"].min())
    fx = fx[fx["Matchday"] == next_md].copy()

    fx["League"] = league_key
    fx = fx.sort_values("Date").reset_index(drop=True)
    fx = _add_calendar_cols(fx)
    return fx


def build_future_features(
    league_key: str,
    feature_list_path: Path,
    history_df: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Zwraca:
      - fixtures_df (terminarz najbliższej kolejki)
      - X_future (cechy pod model, kolumny 1:1 jak w feature_list.json)
    """
    if history_df is None:
        history_df = load_history_features(FEATURES_HISTORY_PATH)

    feature_cols = pd.read_json(feature_list_path, typ="series").tolist()

    fixtures = get_next_matchday_fixtures_for_league(league_key)

    if fixtures.empty:
        X_future = pd.DataFrame(columns=feature_cols)
        return fixtures, X_future

    rows = []
    for _, r in fixtures.iterrows():
        rows.append(_build_features_for_fixture_row(history_df, r, feature_cols))

    X_future = pd.DataFrame(rows, columns=feature_cols)
    
    if "SeasonStrength_PtsDiff" in X_future.columns:
        X_future["SeasonStrength_PtsDiff"] *= 0.6
    return fixtures, X_future


def main():
    """
    Szybki test: buduje cechy dla PL (albo zmień league_key) i zapisuje CSV do data_app/prepared/
    """
    models_dir = PROJECT_ROOT / "models"
    feature_list_path = models_dir / "feature_list.json"

    league_key = "premier_league"

    fixtures, X_future = build_future_features(league_key=league_key, feature_list_path=feature_list_path)

    out_fixtures = OUT_DIR / f"{league_key}_next_matchday_fixtures.csv"
    out_features = OUT_DIR / f"{league_key}_next_matchday_features.csv"

    fixtures.to_csv(out_fixtures, index=False)
    X_future.to_csv(out_features, index=False)

    print("=== OK ===")
    print("Fixtures:", fixtures.shape, "->", out_fixtures)
    print("Features:", X_future.shape, "->", out_features)
    if not fixtures.empty:
        print(fixtures.head())
    if not X_future.empty:
        print(X_future.head())


if __name__ == "__main__":
    main()
