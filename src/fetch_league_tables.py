# src/fetch_league_tables.py
from __future__ import annotations

from pathlib import Path
import os
import requests
import pandas as pd
from dotenv import load_dotenv

from src.football_data_client import LEAGUES



load_dotenv()

API_KEY = os.getenv("FOOTBALL_DATA_KEY")
if API_KEY is None:
    raise RuntimeError("Brak FOOTBALL_DATA_KEY w .env")

HEADERS = {
    "X-Auth-Token": API_KEY
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = PROJECT_ROOT / "data_app" / "standings"
OUT_DIR.mkdir(parents=True, exist_ok=True)



def fetch_standings_for_league(league_key: str, league_code: str) -> pd.DataFrame:
    """
    Pobiera aktualną tabelę ligową (standings) dla danej ligi.
    """
    url = f"https://api.football-data.org/v4/competitions/{league_code}/standings"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()

    data = r.json()


    standings = next(
        s for s in data["standings"] if s["type"] == "TOTAL"
    )

    rows = []
    for row in standings["table"]:
        team = row["team"]

        rows.append({
            "Position": row["position"],
            "Team": team["name"],
            "Played": row["playedGames"],
            "Won": row["won"],
            "Draw": row["draw"],
            "Lost": row["lost"],
            "GoalsFor": row["goalsFor"],
            "GoalsAgainst": row["goalsAgainst"],
            "GoalDiff": row["goalDifference"],
            "Points": row["points"],
        })

    df = pd.DataFrame(rows).sort_values("Position").reset_index(drop=True)
    return df


def main():
    print("=== FETCH LEAGUE TABLES ===\n")

    for league_key, league_code in LEAGUES.items():
        print(f"Pobieram tabelę: {league_key}")

        df = fetch_standings_for_league(league_key, league_code)

        out_path = OUT_DIR / f"{league_key}_table.csv"
        df.to_csv(out_path, index=False)

        print(f"Zapisano: {out_path}")
        print(df.head(5))
        print()

    print("=== DONE ===")


if __name__ == "__main__":
    main()
