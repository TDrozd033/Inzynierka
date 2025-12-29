

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.precision", 4)



LEAGUE = "all_leagues"
EXPERIMENT = "feature_engineering"

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data_app" / "processed" / "all_leagues"
RESULTS_DIR = BASE_DIR / "results" / "all_leagues" / EXPERIMENT

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IN_PATH = DATA_DIR / "all_leagues_clean.csv"
OUT_PATH = DATA_DIR / "all_leagues_features.csv"

print("IN_PATH :", IN_PATH)
print("OUT_PATH:", OUT_PATH)



df = pd.read_csv(IN_PATH, parse_dates=["Date"])
df = df.sort_values(["League", "Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

print("Shape after cleaning:", df.shape)



df["ShotsDiff"] = df["HS"] - df["AS"]
df["ShotsOnTargetDiff"] = df["HST"] - df["AST"]
df["CornersDiff"] = df["HC"] - df["AC"]
df["FoulsDiff"] = df["HF"] - df["AF"]
df["CardsDiff"] = (df["HY"] + df["HR"]) - (df["AY"] + df["AR"])

df["GoalDiff"] = df["FTHG"] - df["FTAG"]
df["TotalGoals"] = df["FTHG"] + df["FTAG"]


inv_sum_b365 = (1/df["B365H"]) + (1/df["B365D"]) + (1/df["B365A"])
df["Prob_H_b365"] = (1/df["B365H"]) / inv_sum_b365
df["Prob_D_b365"] = (1/df["B365D"]) / inv_sum_b365
df["Prob_A_b365"] = (1/df["B365A"]) / inv_sum_b365
df["Margin_B365"] = inv_sum_b365 - 1
df["OddsDiff_B365"] = df["B365H"] - df["B365A"]

mask_ps = df[["PSH", "PSD", "PSA"]].notna().all(axis=1)
inv_sum_ps = (1/df["PSH"]) + (1/df["PSD"]) + (1/df["PSA"])

df["Prob_H_ps"] = np.where(mask_ps, (1/df["PSH"]) / inv_sum_ps, np.nan)
df["Prob_D_ps"] = np.where(mask_ps, (1/df["PSD"]) / inv_sum_ps, np.nan)
df["Prob_A_ps"] = np.where(mask_ps, (1/df["PSA"]) / inv_sum_ps, np.nan)
df["Margin_PS"] = np.where(mask_ps, inv_sum_ps - 1, np.nan)
df["OddsDiff_PS"] = df["PSH"] - df["PSA"]

df["OddsSpread_H"] = df["BbMxH"] - df["BbAvH"]
df["OddsSpread_D"] = df["BbMxD"] - df["BbAvD"]
df["OddsSpread_A"] = df["BbMxA"] - df["BbAvA"]


base_cols = ["League", "Date", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

home = df[base_cols].copy()
home["Team"] = home["HomeTeam"]
home["Opponent"] = home["AwayTeam"]
home["GF"] = home["FTHG"]
home["GA"] = home["FTAG"]

away = df[base_cols].copy()
away["Team"] = away["AwayTeam"]
away["Opponent"] = away["HomeTeam"]
away["GF"] = away["FTAG"]
away["GA"] = away["FTHG"]

def points_for_row(gf, ga):
    if gf > ga: return 3
    if gf == ga: return 1
    return 0

for part in (home, away):
    part["Points"] = [
        points_for_row(gf, ga) for gf, ga in zip(part["GF"], part["GA"])
    ]

tmatches = pd.concat(
    [
        home[["League","Date","Season","Team","Opponent","GF","GA","Points"]],
        away[["League","Date","Season","Team","Opponent","GF","GA","Points"]],
    ],
    ignore_index=True
).sort_values(["League","Team","Season","Date"]).reset_index(drop=True)

GROUP_TEAM = ["League", "Team"]
GROUP_TEAM_SEASON = ["League", "Team", "Season"]



WINDOW = 5
grp = tmatches.groupby(GROUP_TEAM, group_keys=False)

tmatches["GF_last5"] = grp["GF"].apply(lambda s: s.shift(1).rolling(WINDOW, 1).mean())
tmatches["GA_last5"] = grp["GA"].apply(lambda s: s.shift(1).rolling(WINDOW, 1).mean())
tmatches["GD_last5"] = grp.apply(
    lambda g: (g["GF"] - g["GA"]).shift(1).rolling(WINDOW, 1).mean()
).reset_index(level=0, drop=True)
tmatches["Pts_last5"] = grp["Points"].apply(lambda s: s.shift(1).rolling(WINDOW, 1).mean())

for W in (3, 10):
    tmatches[f"Pts_last{W}"] = grp["Points"].apply(lambda s: s.shift(1).rolling(W, 1).mean())
    tmatches[f"GD_last{W}"] = grp.apply(
        lambda g: (g["GF"] - g["GA"]).shift(1).rolling(W, 1).mean()
    ).reset_index(level=0, drop=True)

tmatches["Pts_trend_3v10"] = tmatches["Pts_last3"] - tmatches["Pts_last10"]


team_form_cols = [
    "GF_last5","GA_last5","GD_last5","Pts_last5",
    "Pts_last3","GD_last3","Pts_last10","GD_last10","Pts_trend_3v10"
]

home_form = tmatches[["League","Date","Team"] + team_form_cols] \
    .rename(columns={c: f"Home_{c}" for c in team_form_cols}) \
    .rename(columns={"Team":"HomeTeam"})

away_form = tmatches[["League","Date","Team"] + team_form_cols] \
    .rename(columns={c: f"Away_{c}" for c in team_form_cols}) \
    .rename(columns={"Team":"AwayTeam"})

df = df.merge(home_form, on=["League","Date","HomeTeam"], how="left")
df = df.merge(away_form, on=["League","Date","AwayTeam"], how="left")

for c in team_form_cols:
    df[f"FormDiff_{c}"] = df[f"Home_{c}"] - df[f"Away_{c}"]



df["IsHomeAlways1"] = 1
df["Month"] = df["Date"].dt.month.astype("int16")
df["Weekday"] = df["Date"].dt.weekday.astype("int16")
df["IsWeekend"] = df["Weekday"].isin([5,6]).astype("int8")



tm_season = tmatches.copy()
g = tm_season.groupby(GROUP_TEAM_SEASON)

tm_season["Games_played_season"] = g.cumcount()
tm_season["CumPoints_season"] = g["Points"].cumsum() - tm_season["Points"]
tm_season["CumGD_season"] = (g["GF"].cumsum() - g["GA"].cumsum()) - (tm_season["GF"] - tm_season["GA"])

tm_season["AvgPts_season_sofar"] = np.where(
    tm_season["Games_played_season"] > 0,
    tm_season["CumPoints_season"] / tm_season["Games_played_season"],
    np.nan
)
tm_season["AvgGD_season_sofar"] = np.where(
    tm_season["Games_played_season"] > 0,
    tm_season["CumGD_season"] / tm_season["Games_played_season"],
    np.nan
)

home_season = tm_season[["League","Date","Season","Team","AvgPts_season_sofar","AvgGD_season_sofar"]] \
    .rename(columns={
        "Team":"HomeTeam",
        "AvgPts_season_sofar":"Home_AvgPts_season",
        "AvgGD_season_sofar":"Home_AvgGD_season"
    })

away_season = tm_season[["League","Date","Season","Team","AvgPts_season_sofar","AvgGD_season_sofar"]] \
    .rename(columns={
        "Team":"AwayTeam",
        "AvgPts_season_sofar":"Away_AvgPts_season",
        "AvgGD_season_sofar":"Away_AvgGD_season"
    })

df = df.merge(home_season, on=["League","Date","Season","HomeTeam"], how="left")
df = df.merge(away_season, on=["League","Date","Season","AwayTeam"], how="left")

df["SeasonStrength_PtsDiff"] = df["Home_AvgPts_season"] - df["Away_AvgPts_season"]
df["SeasonStrength_GDDiff"] = df["Home_AvgGD_season"] - df["Away_AvgGD_season"]



eps = 1e-3

for side in ["Home", "Away"]:
    gf5 = df[f"{side}_GF_last5"]
    ga5 = df[f"{side}_GA_last5"]

    df[f"{side}_AttDefRatio_last5"] = gf5 / (ga5 + 1)
    df[f"{side}_GoalsShare_last5"] = gf5 / (gf5 + ga5 + eps)

df["FormDiff_AttDefRatio_last5"] = (
    df["Home_AttDefRatio_last5"] - df["Away_AttDefRatio_last5"]
)
df["FormDiff_GoalsShare_last5"] = (
    df["Home_GoalsShare_last5"] - df["Away_GoalsShare_last5"]
)

df["Home_RecentFormIdx"] = (
    0.6 * df["Home_Pts_last3"] + 0.4 * df["Home_Pts_last5"]
)
df["Away_RecentFormIdx"] = (
    0.6 * df["Away_Pts_last3"] + 0.4 * df["Away_Pts_last5"]
)

df["FormDiff_RecentFormIdx"] = (
    df["Home_RecentFormIdx"] - df["Away_RecentFormIdx"]
)

if {"Prob_H_b365", "Prob_A_b365"}.issubset(df.columns):
    df["Market_ImpliedDiff_b365"] = (
        df["Prob_H_b365"] - df["Prob_A_b365"]
    )
else:
    df["Market_ImpliedDiff_b365"] = np.nan

if {"Prob_H_ps", "Prob_A_ps"}.issubset(df.columns):
    df["Market_ImpliedDiff_ps"] = (
        df["Prob_H_ps"] - df["Prob_A_ps"]
    )
else:
    df["Market_ImpliedDiff_ps"] = np.nan

if {"Prob_H_b365", "Prob_A_b365"}.issubset(df.columns):
    df["Market_BigHomeFav"] = (df["Prob_H_b365"] >= 0.60).astype("int8")
    df["Market_BigAwayFav"] = (df["Prob_A_b365"] >= 0.60).astype("int8")
    df["Market_BalancedMatch"] = (
        (df["Prob_H_b365"] < 0.45) &
        (df["Prob_A_b365"] < 0.35)
    ).astype("int8")
else:
    df["Market_BigHomeFav"] = 0
    df["Market_BigAwayFav"] = 0
    df["Market_BalancedMatch"] = 0



df.to_csv(OUT_PATH, index=False)
print("FINAL SHAPE:", df.shape)
print("Saved to:", OUT_PATH)
