import json
import joblib
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, classification_report



PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data_app/processed/all_leagues/all_leagues_features.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "random_forest_model.pkl"
ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
FEATURES_PATH = MODELS_DIR / "feature_list.json"

print("PROJECT_ROOT:", PROJECT_ROOT)
print("DATA_PATH:", DATA_PATH)



df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

print("Dane:", df.shape)



target = "FTR"

cols_to_exclude = {
    # meta
    "Div", "Date", "Season", "SourceFile", "HomeTeam", "AwayTeam", "Referee",

    # wynik meczu
    "FTHG", "FTAG", "HTHG", "HTAG", "FTR",
    "GoalDiff", "TotalGoals",
    "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC",
    "HY", "AY", "HR", "AR",

    # bukmacher – nadmiarowe
    "Prob_D_b365", "Prob_A_b365",
    "Prob_H_ps", "Prob_D_ps", "Prob_A_ps",
    "Margin_B365", "Margin_PS",
    "OddsDiff_B365", "OddsDiff_PS",
    "OddsSpread_H", "OddsSpread_D", "OddsSpread_A",

    # forma – duplikaty
    "Home_Wins_last5", "Home_CleanSheets_last5", "Home_Conceded0plus_last5",
    "Home_GD_last3", "Home_Pts_last3", "Home_GD_last10", "Home_AvgGD_season",

    "Away_Wins_last5", "Away_CleanSheets_last5", "Away_Conceded0plus_last5",
    "Away_GD_last3", "Away_Pts_last3", "Away_GD_last10", "Away_AvgGD_season",

    "FormDiff_Wins_last5", "FormDiff_CleanSheets_last5",
    "FormDiff_Conceded0plus_last5", "FormDiff_GD_last3",
    "FormDiff_Pts_last3", "FormDiff_GD_last10", "FormDiff_Pts_last10",

    "IsWeekend",
    "SeasonStrength_GDDiff",
}

candidate_cols = [c for c in df.columns if c not in cols_to_exclude]

keep_patterns = (
    "Prob_", "Margin_", "OddsDiff", "OddsSpread",
    "Home_", "Away_", "FormDiff_", "Pts_trend_3v10",
    "SeasonStrength_", "Month", "Weekday"
)

X_features = [c for c in candidate_cols if any(p in c for p in keep_patterns)]
X_features = [c for c in X_features if c != "Prob_H_b365"]  
print("Liczba cech:", len(X_features))



X = df[X_features].copy()

le = LabelEncoder()
y = le.fit_transform(df[target])

print("Klasy:", list(le.classes_))


rf_model = RandomForestClassifier(
    n_estimators=600,
    max_depth=8,
    min_samples_leaf=10,
    min_samples_split=20,
    max_features=0.3,
    max_samples=0.8,
    class_weight="balanced",
    random_state=0,
    n_jobs=-1
)

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", rf_model)
])




calibrated_model = CalibratedClassifierCV(
    estimator=pipe,
    method="isotonic",
    cv=5
)

print("Trenowanie modelu (Random Forest + kalibracja)")
calibrated_model.fit(X, y)

print("Model wytrenowany.")



joblib.dump(calibrated_model, MODEL_PATH)
joblib.dump(le, ENCODER_PATH)

with open(FEATURES_PATH, "w", encoding="utf-8") as f:
    json.dump(X_features, f, indent=2, ensure_ascii=False)

print("Zapisano:")
print(" - model:", MODEL_PATH)
print(" - encoder:", ENCODER_PATH)
print(" - feature list:", FEATURES_PATH)

print("=== DONE ===")
