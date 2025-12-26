from src.football_data_client import (
    get_future_fixtures,
    get_league_table,
    LEAGUES
)

df_fixtures = get_future_fixtures(LEAGUES["premier_league"], limit=5)
df_table = get_league_table(LEAGUES["premier_league"])

print("=== TERMINARZ ===")
print(df_fixtures)

print("\n=== TABELA ===")
print(df_table.head())
