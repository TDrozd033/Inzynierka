import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("FOOTBALL_DATA_KEY")

headers = {
    "X-Auth-Token": API_KEY
}

url = "https://api.football-data.org/v4/competitions/PL/matches"

params = {
    "status": "SCHEDULED",
    "limit": 10
}

r = requests.get(url, headers=headers, params=params)

print("Status:", r.status_code)
data = r.json()

print("Liczba meczów:", len(data.get("matches", [])))
if data.get("matches"):
    m = data["matches"][0]
    print("PRZYKŁAD:")
    print(
        m["utcDate"],
        m["homeTeam"]["name"],
        "-",
        m["awayTeam"]["name"],
        "| status:",
        m["status"]
    )