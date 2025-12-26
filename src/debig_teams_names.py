import requests
import pandas as pd
import os
from dotenv import load_dotenv

# === konfiguracja ===
load_dotenv()

API_KEY = os.getenv("FOOTBALL_DATA_KEY")  
assert API_KEY is not None, "❌ Brak API key w .env"

HEADERS = {
    "X-Auth-Token": API_KEY
}

LEAGUES = {
    "premier_league": "PL",
    "la_liga": "PD",
    "serie_a": "SA",
    "bundesliga": "BL1",
    "ligue_1": "FL1"
}

rows = []

for league_name, code in LEAGUES.items():
    print(f"⬇️ Pobieram terminarz: {league_name}")

    url = f"https://api.football-data.org/v4/competitions/{code}/matches"
    params = {
        "status": "SCHEDULED"   # tylko przyszłe mecze
    }

    r = requests.get(url, headers=HEADERS, params=params)
    r.raise_for_status()

    data = r.json()["matches"]

    for m in data:
        rows.append({
            "league": league_name,
            "home": m["homeTeam"]["name"],
            "away": m["awayTeam"]["name"],
            "date": m["utcDate"]
        })

api_fixtures = pd.DataFrame(rows)

print("\n=== PRZYKŁADOWE MECZE ===")    
print(api_fixtures.head(10))

api_teams = (
    pd.Series(pd.concat([api_fixtures["home"], api_fixtures["away"]]))
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)

print("\n=== NAZWY DRUŻYN Z API (TERMINARZ) ===")
print(f"Liczba drużyn: {len(api_teams)}")
print(api_teams)


from pathlib import Path

# === zapis nazw drużyn z API ===
out_dir = Path("data_app/debug")
out_dir.mkdir(parents=True, exist_ok=True)

api_teams_df = api_teams.to_frame(name="team_name_api")

out_path = out_dir / "teams_from_api.csv"
api_teams_df.to_csv(out_path, index=False, encoding="utf-8")

print(f" Zapisano nazwy drużyn z API: {out_path}")