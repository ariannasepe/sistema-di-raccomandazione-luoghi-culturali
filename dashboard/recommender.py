"""
recommender.py
--------------
Motore di raccomandazione content-based per luoghi culturali italiani.
Integrato con user_profile.py.

Logica di scoring:
    score_finale = (score_tema * 0.5) + (score_tipologia * 0.3) + (score_distanza * 0.2)

    - score_tema:      peso utente sul tema_principale del luogo (0.0 – 1.0)
    - score_tipologia: peso utente sulla tipologia del luogo (0.0 – 1.0)
    - score_distanza:  1.0 se il luogo è a 0 km, 0.0 al limite del raggio

    I luoghi oltre il raggio vengono esclusi prima del calcolo.
    Se solo_accessibile=True, vengono esclusi i luoghi con accessibile_disabili=0.

Utilizzo con UserProfile:
    from user_profile import UserProfile, get_area_centroid
    from recommender import raccomanda

    profilo = UserProfile(
        temi_pesi={"Archeologia": 0.9, "Storia e Memoria": 0.5},
        tipologie_pesi={"Museo": 1.0, "Sito archeologico": 0.8},
        area={"tipo": "regione", "valore": "Toscana"},
        raggio_km=60,
        solo_accessibile=False,
    )

    df = pd.read_csv("dataset_sistema.csv")
    risultati = raccomanda(profilo, df, top_n=10)
    print(risultati[["nome", "tipologia", "tema_principale", "comune", "distanza_km", "score"]])
"""

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

from user_profile import UserProfile, get_area_centroid, TEMI, TIPOLOGIE


# ------------------------------------------------------------------ #
# Distanza Haversine                                                  #
# ------------------------------------------------------------------ #

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Restituisce la distanza in km tra due punti geografici."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ------------------------------------------------------------------ #
# Feature matrix dei luoghi                                           #
# ------------------------------------------------------------------ #

def build_luoghi_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Costruisce la feature matrix dei luoghi: shape (n_luoghi, 15).

    Ogni riga è un vettore one-hot:
        [tema_1, ..., tema_9, tipologia_1, ..., tipologia_6]

    Non viene normalizzata qui — la normalizzazione avviene nel calcolo
    della cosine similarity.
    """
    tema_matrix = pd.get_dummies(df["tema_principale"]).reindex(columns=TEMI, fill_value=0)
    tipo_matrix = pd.get_dummies(df["tipologia"]).reindex(columns=TIPOLOGIE, fill_value=0)
    matrix = np.hstack([tema_matrix.values, tipo_matrix.values]).astype(float)
    return matrix


# ------------------------------------------------------------------ #
# Motore di raccomandazione                                           #
# ------------------------------------------------------------------ #

def raccomanda(
    profilo: UserProfile,
    df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Genera una lista ordinata di luoghi culturali raccomandati.

    Parametri
    ----------
    profilo : UserProfile
        Profilo utente costruito con user_profile.py.
    df : pd.DataFrame
        Dataset dei luoghi culturali.
    top_n : int
        Numero massimo di risultati da restituire.

    Restituisce
    -----------
    pd.DataFrame con colonne:
        nome, tipologia, tema_principale, comune, regione,
        distanza_km, score_tema, score_tipologia, score_distanza, score
    """

    df = df.copy()

    # ── 1. Centroide area utente ──────────────────────────────────────
    centroid = get_area_centroid(profilo, df)
    if centroid is None:
        return pd.DataFrame(columns=["nome", "tipologia", "tema_principale",
                                     "comune", "regione", "distanza_km", "score"])
    lat_u, lon_u = centroid

    # ── 2. Calcolo distanza ───────────────────────────────────────────
    df["distanza_km"] = df.apply(
        lambda r: haversine(lat_u, lon_u, r["lat"], r["lon"]), axis=1
    )

    # ── 3. Filtro raggio ──────────────────────────────────────────────
    df = df[df["distanza_km"] <= profilo.raggio_km].copy()
    if df.empty:
        return pd.DataFrame()

    # ── 4. Filtro accessibilità ───────────────────────────────────────
    if profilo.solo_accessibile:
        df = df[df["accessibile_disabili"] == 1].copy()
    if df.empty:
        return pd.DataFrame()

    # ── 5. Score tema (dal profilo utente) ────────────────────────────
    df["score_tema"] = df["tema_principale"].map(
        lambda t: profilo.temi_pesi.get(t, 0.0)
    )

    # ── 6. Score tipologia (dal profilo utente) ───────────────────────
    df["score_tipologia"] = df["tipologia"].map(
        lambda t: profilo.tipologie_pesi.get(t, 0.0)
    )

    # ── 7. Score distanza (decadimento lineare) ───────────────────────
    df["score_distanza"] = (1.0 - df["distanza_km"] / profilo.raggio_km).clip(0.0, 1.0)

    # ── 8. Score finale ponderato ─────────────────────────────────────
    df["score"] = (
        df["score_tema"]      * 0.5 +
        df["score_tipologia"] * 0.3 +
        df["score_distanza"]  * 0.2
    )

    # ── 9. Escludi luoghi con score = 0 ──────────────────────────────
    df = df[df["score"] > 0].copy()
    if df.empty:
        return pd.DataFrame()

    # ── 10. Ordinamento e selezione colonne ───────────────────────────
    df = df.sort_values("score", ascending=False).head(top_n)

    colonne = [
        "nome", "tipologia", "tema_principale", "comune", "regione",
        "lat", "lon", "distanza_km", "score_tema", "score_tipologia", "score_distanza", "score", "accessibile_disabili"
    ]
    return (
        df[colonne]
        .round({"distanza_km": 1, "score": 3,
                "score_tema": 3, "score_tipologia": 3, "score_distanza": 3})
        .reset_index(drop=True)
    )


# ------------------------------------------------------------------ #
# Explainability: motivazione della raccomandazione                   #
# ------------------------------------------------------------------ #

def spiega_raccomandazione(row: pd.Series, profilo: UserProfile) -> str:
    """
    Genera una stringa leggibile che motiva la raccomandazione
    per un singolo luogo (da usare nelle schede Streamlit).

    Esempio output:
        "Consigliato perché corrisponde al tuo interesse per l'Archeologia
         (tema: 0.9), è un Sito archeologico che hai indicato come preferito
         (tipologia: 0.8) ed è a soli 12.3 km da te."
    """
    parti = []

    if row["score_tema"] > 0:
        parti.append(
            f"corrisponde al tuo interesse per **{row['tema_principale']}** "
            f"(peso: {profilo.temi_pesi.get(row['tema_principale'], 0):.1f})"
        )

    if row["score_tipologia"] > 0:
        parti.append(
            f"è un **{row['tipologia']}** che hai indicato come preferito "
            f"(peso: {profilo.tipologie_pesi.get(row['tipologia'], 0):.1f})"
        )

    parti.append(f"si trova a **{row['distanza_km']:.1f} km** dall'area selezionata")

    return "Consigliato perché " + ", ".join(parti) + "."


# ------------------------------------------------------------------ #
# Test standalone                                                      #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    from user_profile import UserProfile

    profilo_test = UserProfile(
        nome="Anna",
        temi_pesi={
            "Archeologia": 0.9,
            "Arte antica e medievale": 0.7,
            "Storia e Memoria": 0.5,
            "Scienze e natura": 0.1,
        },
        tipologie_pesi={
            "Museo": 1.0,
            "Sito archeologico": 1.0,
            "Galleria": 0.5,
            "Monumento": 0.3,
        },
        area={"tipo": "regione", "valore": "Toscana"},
        raggio_km=60,
        solo_accessibile=False,
    )

    df = pd.read_csv("dataset_sistema.csv")
    risultati = raccomanda(profilo_test, df, top_n=10)

    if risultati.empty:
        print("Nessun risultato trovato.")
    else:
        print(f"Top {len(risultati)} luoghi per {profilo_test.nome} in {profilo_test.area['valore']}:\n")
        print(risultati.to_string(index=False))
        print("\n--- Spiegazioni ---")
        for _, row in risultati.iterrows():
            print(f"\n{row['nome']}")
            print(spiega_raccomandazione(row, profilo_test))