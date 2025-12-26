import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.getenv("FOOTBALL_DATA_KEY")

BASE_URL = "https://api.football-data.org/v4"

HEADERS = {
    "X-Auth-Token": API_TOKEN
}

# Mapowanie lig
LEAGUES = {
    "premier_league": "PL",
    "serie_a": "SA",
    "la_liga": "PD",
    "bundesliga": "BL1",
    "ligue_1": "FL1"
}


def get_future_fixtures(league_code, limit=10):
    """
    Zwraca przyszłe mecze (TIMED / SCHEDULED)
    """
    url = f"{BASE_URL}/competitions/{league_code}/matches"
    params = {
        "status": "SCHEDULED"
    }

    response = requests.get(url, headers=HEADERS, params=params)
    response.raise_for_status()

    matches = response.json()["matches"][:limit]

    rows = []
    for m in matches:
        rows.append({
            "Date": m["utcDate"],
            "HomeTeam": m["homeTeam"]["name"],
            "AwayTeam": m["awayTeam"]["name"],
            "Matchday": m["matchday"]
        })

    return pd.DataFrame(rows)


def get_league_table(league_code):
    """
    Zwraca aktualną tabelę ligi
    """
    url = f"{BASE_URL}/competitions/{league_code}/standings"

    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()

    standings = response.json()["standings"][0]["table"]

    rows = []
    for s in standings:
        rows.append({
            "Position": s["position"],
            "Team": s["team"]["name"],
            "Played": s["playedGames"],
            "Points": s["points"],
            "GoalDiff": s["goalDifference"],
            "GoalsFor": s["goalsFor"],
            "GoalsAgainst": s["goalsAgainst"]
        })

    return pd.DataFrame(rows)
