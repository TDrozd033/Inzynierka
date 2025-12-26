# src/predict_matches.py
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import joblib

from src.prepare_features import build_future_features
from src.football_data_client import LEAGUES


# =====================
# KONFIG / ŚCIEŻKI
# =====================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "decision_tree_model.pkl"
ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
FEATURE_LIST_PATH = MODELS_DIR / "feature_list.json"

OUT_DIR = PROJECT_ROOT / "data_app" / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ligi aplikacji = 5 lig
APP_LEAGUE_KEYS = list(LEAGUES.keys())  # ["premier_league","la_liga","serie_a","bundesliga","ligue_1"]


# =====================
# HELPERY
# =====================

def _load_feature_list(path: Path) -> list[str]:
    # u Ciebie zapisane jako JSON listy
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("feature_list.json musi być listą kolumn (list[str]).")
    return data


def _safe_predict_proba(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Zwraca DataFrame z kolumnami proba_0, proba_1, proba_2
    (kolejność klas = taka jak model.predict_proba)
    """
    proba = model.predict_proba(X)
    return pd.DataFrame(proba, columns=[f"proba_{i}" for i in range(proba.shape[1])])


def _make_output_df(fixtures: pd.DataFrame, y_pred, y_proba: pd.DataFrame, le) -> pd.DataFrame:
    """
    Składa czytelny wynik:
    Date | HomeTeam | AwayTeam | Matchday | Pred | P(A) | P(D) | P(H)
    UWAGA: kolejność labeli zależy od label_encoder.classes_
    """
    out = fixtures[["Date", "HomeTeam", "AwayTeam", "Matchday", "League"]].copy()

    # predykcja klasy
    pred_labels = le.inverse_transform(y_pred)
    out["Pred"] = pred_labels

    # mapowanie prawdopodobieństw do A/D/H wg encoder.classes_
    # classes_ np. ['A','D','H'] -> wtedy proba_0=A, proba_1=D, proba_2=H
    class_to_col = {cls: f"proba_{i}" for i, cls in enumerate(le.classes_)}

    for cls in ["A", "D", "H"]:
        if cls in class_to_col:
            out[f"P({cls})"] = y_proba[class_to_col[cls]]
        else:
            out[f"P({cls})"] = pd.NA

    # wskazanie “pewności”
    out["Confidence"] = out[[c for c in ["P(A)", "P(D)", "P(H)"] if c in out.columns]].max(axis=1)

    # sort i format daty
    out = out.sort_values("Date").reset_index(drop=True)
    return out


def _debug_nan_report(X_future: pd.DataFrame) -> pd.Series:
    """Ile NaN w kolumnach – do logu."""
    return X_future.isna().sum().sort_values(ascending=False)


# =====================
# GŁÓWNA FUNKCJA
# =====================

def predict_for_league(league_key: str, model, le, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Zwraca:
      - fixtures (najbliższa kolejka)
      - preds_df (z predykcjami)
    """
    fixtures, X_future = build_future_features(
        league_key=league_key,
        feature_list_path=FEATURE_LIST_PATH
    )

    if fixtures.empty or X_future.empty:
        return fixtures, pd.DataFrame()

    # upewnij się, że kolumny są 1:1 jak model chce
    X_future = X_future.reindex(columns=feature_cols)

    # log NaN (to normalne dla Prob_H_b365, bo API nie daje kursów)
    nan_report = _debug_nan_report(X_future)
    top_nan = nan_report[nan_report > 0].head(5)
    if len(top_nan) > 0:
        print(f"[{league_key}] TOP NaN kolumny:\n{top_nan}\n")

    # predykcja
    X_future = X_future.replace({pd.NA: float("nan")})
    y_pred = model.predict(X_future)
    
    ##### wersja przed zmiana pod decisioon tree - byc moze przy zmianie modeli wrocic do tej wersji ale nie pewne 
    '''
    y_proba = _safe_predict_proba(model, X_future)

    preds_df = _make_output_df(fixtures, y_pred, y_proba, le)
    return fixtures, preds_df
    '''
    #### tu koniec 
    
    # predykcja prawdopodobieństw
    y_proba = _safe_predict_proba(model, X_future)

    # klasy w kolejności encodera, np. ['A','D','H']
    classes = le.classes_

    preds = []
    confidences = []

    for _, row in y_proba.iterrows():
        p_dict = {cls: row[f"proba_{i}"] for i, cls in enumerate(classes)}

        P_H = p_dict.get("H", 0.0)
        P_D = p_dict.get("D", 0.0)
        P_A = p_dict.get("A", 0.0)

        # REGUŁA REMISU
        if abs(P_H - P_A) < 0.05 and P_D > 0.25:
            pred = "D"
            conf = P_D
        else:
            pred = max(p_dict, key=p_dict.get)
            conf = p_dict[pred]

        preds.append(pred)
        confidences.append(conf)

    # składanie outputu
    preds_df = fixtures[["Date", "HomeTeam", "AwayTeam", "Matchday", "League"]].copy()
    preds_df["Pred"] = preds
    preds_df["Confidence"] = confidences

    # kolumny P(A), P(D), P(H)
    for cls in ["A", "D", "H"]:
        col = f"P({cls})"
        if cls in classes:
            preds_df[col] = y_proba[f"proba_{list(classes).index(cls)}"]
        else:
            preds_df[col] = pd.NA

    preds_df = preds_df.sort_values("Date").reset_index(drop=True)
    return fixtures, preds_df

###### koniec wstawionych zmian przy edycji 





def main():
    # walidacje
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Brak modelu: {MODEL_PATH}")
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"Brak encodera: {ENCODER_PATH}")
    if not FEATURE_LIST_PATH.exists():
        raise FileNotFoundError(f"Brak listy cech: {FEATURE_LIST_PATH}")

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("MODEL:", MODEL_PATH)
    print("ENCODER:", ENCODER_PATH)
    print("FEATURES:", FEATURE_LIST_PATH)

    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    feature_cols = _load_feature_list(FEATURE_LIST_PATH)

    print("Klasy modelu:", getattr(le, "classes_", None))
    print("Liczba cech:", len(feature_cols))
    print("Ligi:", APP_LEAGUE_KEYS)
    print()

    all_outputs = []

    for league_key in APP_LEAGUE_KEYS:
        print(f"=== PREDYKCJE: {league_key} ===")

        fixtures, preds_df = predict_for_league(
            league_key=league_key,
            model=model,
            le=le,
            feature_cols=feature_cols
        )

        if fixtures.empty or preds_df.empty:
            print(f"[{league_key}] Brak meczów / brak danych do predykcji.\n")
            continue

        out_path = OUT_DIR / f"{league_key}_next_matchday_predictions.csv"
        preds_df.to_csv(out_path, index=False)

        print(f"[{league_key}] Zapisano: {out_path}")
        print(preds_df.head(10))
        print()

        all_outputs.append(preds_df)

    # opcjonalnie: jeden wspólny plik
    if all_outputs:
        merged = pd.concat(all_outputs, ignore_index=True)
        merged_path = OUT_DIR / "all_leagues_next_matchday_predictions.csv"
        merged.to_csv(merged_path, index=False)
        print("=== DONE ===")
        print("Zapisano zbiorczy plik:", merged_path)
        print("Wiersze:", merged.shape[0])
    else:
        print("=== DONE ===")
        print("Nie zapisano nic – brak danych z API / brak przyszłych meczów.")


if __name__ == "__main__":
    main()
