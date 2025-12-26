# Football Match Predictor

Aplikacja do predykcji wyników najbliższej kolejki piłkarskiej dla **5 topowych lig europejskich**:
- Premier League
- La Liga
- Serie A
- Bundesliga
- Ligue 1

Projekt powstał jako część **pracy inżynierskiej** i prezentuje kompletny pipeline:
pobieranie danych → inżynieria cech → predykcja → aplikacja GUI.

---

## Funkcjonalności

-  **Aktualne tabele ligowe** (pobierane z API)
-  **Predykcje wyników najbliższej kolejki**
-  Prawdopodobieństwa: Home / Draw / Away
-  Graficzny interfejs użytkownika (Streamlit)
-  Automatyczny pipeline odświeżania danych

---

## Model predykcyjny

- Algorytm: **Decision Tree Classifier**
- Zbalansowane klasy (`class_weight='balanced'`)
- Kalibracja prawdopodobieństw (`CalibratedClassifierCV`)
- Zestaw 32 cech opisujących formę, siłę drużyn i kontekst meczu

Model został wytrenowany offline, a aplikacja wykorzystuje **gotowy model** do predykcji przyszłych meczów.

---

## Interfejs użytkownika:

Dla każdej ligi dostępne są:
- Podgląd aktualnej tabeli ligowej
- Predykcje najbliższej kolejki (wynik + prawdopodobieństwa)

Tabele zawierają:
- wyróżnienia stref pucharowych i spadkowych
- czytelne formatowanie
- predykcje w formie: zwycięzca / draw

## Automatyzacja

Projekt zawiera skrypt run_pipeline.py, który umożliwia:
- szybkie odświeżenie danych w dowolnym momencie
- brak ręcznej ingerencji w pliki CSV
- łatwą prezentację aktualnych wyników w aplikacji


