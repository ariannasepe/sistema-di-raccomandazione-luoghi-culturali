"""
user_profile.py
---------------
Modulo per la gestione del profilo utente nel sistema di raccomandazione
dei luoghi culturali italiani.

Uso standalone (test):
    python user_profile.py

Uso in Streamlit:
    from user_profile import render_profile_form, build_user_vector
"""

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np

# ── Valori ammessi (derivati dal dataset) ─────────────────────────────────────

TEMI = [
    "Archeologia",
    "Arte antica e medievale",
    "Arte moderna e contemporanea",
    "Cultura e sapere",
    "Cultura generale",
    "Etnografia e territorio",
    "Scienze e natura",
    "Spettacolo e musica",
    "Storia e Memoria",
]

TIPOLOGIE = [
    "Biblioteca",
    "Galleria",
    "Monumento",
    "Museo",
    "Sito archeologico",
    "Teatro",
]

# ── Profili predefiniti ───────────────────────────────────────────────────────

PROFILI_PREDEFINITI = {
    "Appassionato di storia e archeologia": {
        "descrizione": "Ami musei storici, siti archeologici e monumenti. Cerchi luoghi che raccontino il passato.",
        "temi_pesi": {
            "Storia e Memoria":          1.0,
            "Archeologia":               0.9,
            "Arte antica e medievale":   0.6,
            "Etnografia e territorio":   0.3,
            "Cultura generale":          0.2,
            "Cultura e sapere":          0.2,
            "Arte moderna e contemporanea": 0.0,
            "Scienze e natura":          0.0,
            "Spettacolo e musica":       0.0,
        },
        "tipologie_pesi": {
            "Museo":             1.0,
            "Sito archeologico": 1.0,
            "Monumento":         0.7,
            "Galleria":          0.2,
            "Biblioteca":        0.2,
            "Teatro":            0.0,
        },
    },
    "Amante dell'arte": {
        "descrizione": "Gallerie, musei d'arte, collezioni. Ti interessano sia i grandi maestri che l'arte contemporanea.",
        "temi_pesi": {
            "Arte moderna e contemporanea": 1.0,
            "Arte antica e medievale":      0.9,
            "Cultura generale":             0.4,
            "Cultura e sapere":             0.3,
            "Storia e Memoria":             0.2,
            "Etnografia e territorio":      0.1,
            "Archeologia":                  0.1,
            "Scienze e natura":             0.0,
            "Spettacolo e musica":          0.0,
        },
        "tipologie_pesi": {
            "Galleria":          1.0,
            "Museo":             0.8,
            "Monumento":         0.3,
            "Biblioteca":        0.2,
            "Sito archeologico": 0.1,
            "Teatro":            0.0,
        },
    },
    "Famiglia con bambini": {
        "descrizione": "Cerchi luoghi coinvolgenti e didattici, adatti a bambini e ragazzi.",
        "temi_pesi": {
            "Scienze e natura":             1.0,
            "Cultura generale":             0.8,
            "Etnografia e territorio":      0.6,
            "Storia e Memoria":             0.4,
            "Archeologia":                  0.4,
            "Arte antica e medievale":      0.2,
            "Arte moderna e contemporanea": 0.2,
            "Cultura e sapere":             0.3,
            "Spettacolo e musica":          0.3,
        },
        "tipologie_pesi": {
            "Museo":             1.0,
            "Sito archeologico": 0.5,
            "Galleria":          0.3,
            "Teatro":            0.3,
            "Monumento":         0.2,
            "Biblioteca":        0.2,
        },
    },
    "Turista culturale": {
        "descrizione": "Vuoi scoprire il meglio di un'area senza preferenze precise. Aperto a tutto.",
        "temi_pesi": {
            "Storia e Memoria":             0.7,
            "Arte antica e medievale":      0.7,
            "Arte moderna e contemporanea": 0.6,
            "Archeologia":                  0.6,
            "Etnografia e territorio":      0.6,
            "Cultura generale":             0.6,
            "Cultura e sapere":             0.6,
            "Scienze e natura":             0.6,
            "Spettacolo e musica":          0.6,
        },
        "tipologie_pesi": {
            "Museo":             0.7,
            "Galleria":          0.7,
            "Monumento":         0.7,
            "Sito archeologico": 0.7,
            "Teatro":            0.7,
            "Biblioteca":        0.7,
        },
    },
    "Appassionato di spettacolo e cultura locale": {
        "descrizione": "Teatri, biblioteche, tradizioni locali. Ti interessa la cultura viva del territorio.",
        "temi_pesi": {
            "Spettacolo e musica":          1.0,
            "Etnografia e territorio":      0.8,
            "Cultura e sapere":             0.7,
            "Cultura generale":             0.5,
            "Storia e Memoria":             0.3,
            "Arte moderna e contemporanea": 0.2,
            "Arte antica e medievale":      0.1,
            "Archeologia":                  0.1,
            "Scienze e natura":             0.1,
        },
        "tipologie_pesi": {
            "Teatro":            1.0,
            "Biblioteca":        0.6,
            "Museo":             0.5,
            "Galleria":          0.3,
            "Monumento":         0.2,
            "Sito archeologico": 0.1,
        },
    },
}

REGIONI = [
    "Abruzzo", "Basilicata", "Calabria", "Campania", "Emilia-Romagna",
    "Friuli-Venezia Giulia", "Lazio", "Liguria", "Lombardia", "Marche",
    "Molise", "Piemonte", "Puglia", "Sardegna", "Sicilia", "Toscana",
    "Trentino-Alto Adige", "Umbria", "Valle d'Aosta", "Veneto",
]

# ── Dataclass profilo ─────────────────────────────────────────────────────────

@dataclass
class UserProfile:
    """
    Rappresenta le preferenze di un utente.

    Attributi
    ---------
    temi_pesi : dict[str, float]
        Peso da 0.0 a 1.0 per ciascun tema. 0 = nessun interesse, 1 = massimo.
    tipologie_pesi : dict[str, float]
        Peso da 0.0 a 1.0 per ciascuna tipologia.
    area : dict con chiavi 'tipo' ('regione' | 'comune') e 'valore' (str)
        Area geografica di interesse dell'utente.
    raggio_km : float
        Raggio massimo di ricerca in km attorno al centroide dell'area.
    solo_accessibile : bool
        Se True, filtra solo luoghi accessibili ai disabili.
    nome : str
        Nome opzionale dell'utente (per la UI).
    """
    temi_pesi: dict = field(default_factory=dict)
    tipologie_pesi: dict = field(default_factory=dict)
    area: dict = field(default_factory=lambda: {"tipo": "regione", "valore": "Lazio"})
    raggio_km: float = 50.0
    solo_accessibile: bool = False
    nome: str = ""

    def is_valid(self) -> tuple[bool, str]:
        """Verifica che il profilo sia compilato correttamente."""
        if not any(v > 0 for v in self.temi_pesi.values()):
            return False, "Seleziona almeno un tema di interesse."
        if not any(v > 0 for v in self.tipologie_pesi.values()):
            return False, "Seleziona almeno una tipologia di luogo."
        if not self.area.get("valore"):
            return False, "Indica un'area geografica."
        return True, ""

    def summary(self) -> str:
        """Restituisce una stringa leggibile del profilo (per le schede)."""
        temi_attivi = [t for t, v in self.temi_pesi.items() if v > 0]
        tipologie_attive = [t for t, v in self.tipologie_pesi.items() if v > 0]
        lines = [
            f"Utente: {self.nome or 'Anonimo'}",
            f"Area: {self.area['valore']} (entro {self.raggio_km:.0f} km)",
            f"Temi: {', '.join(temi_attivi) or 'nessuno'}",
            f"Tipologie: {', '.join(tipologie_attive) or 'nessuna'}",
            f"Solo accessibile: {'sì' if self.solo_accessibile else 'no'}",
        ]
        return "\n".join(lines)


# ── Costruzione del vettore feature ──────────────────────────────────────────

def build_user_vector(profile: UserProfile) -> np.ndarray:
    """
    Converte il profilo utente in un vettore numerico normalizzato.

    Il vettore è la concatenazione di:
      [pesi temi (9,)] + [pesi tipologie (6,)]
    → shape: (15,)

    I pesi sono normalizzati in [0, 1] e la norma L2 del vettore è 1
    (necessario per il calcolo della cosine similarity).
    """
    tema_vec = np.array([profile.temi_pesi.get(t, 0.0) for t in TEMI], dtype=float)
    tipo_vec = np.array([profile.tipologie_pesi.get(t, 0.0) for t in TIPOLOGIE], dtype=float)
    vec = np.concatenate([tema_vec, tipo_vec])

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


# ── Helper: centroide dell'area selezionata ───────────────────────────────────

def get_area_centroid(profile: UserProfile, df: pd.DataFrame) -> Optional[tuple[float, float]]:
    """
    Calcola il centroide (lat, lon) dell'area selezionata dall'utente
    come media delle coordinate dei luoghi presenti nel dataset in quell'area.

    Restituisce None se l'area non ha luoghi nel dataset.
    """
    tipo = profile.area.get("tipo")
    valore = profile.area.get("valore", "").lower()

    if tipo == "regione":
        subset = df[df["regione"].str.lower() == valore.lower()]
    elif tipo == "comune":
        subset = df[df["comune"].str.lower() == valore.lower()]
    else:
        return None

    if subset.empty:
        return None

    return float(subset["lat"].mean()), float(subset["lon"].mean())


# ── Rendering Streamlit ───────────────────────────────────────────────────────

def render_profile_form(df: pd.DataFrame) -> UserProfile:
    """
    Renderizza il form di onboarding in Streamlit e restituisce
    un oggetto UserProfile compilato.

    Parametri
    ---------
    df : pd.DataFrame
        Il dataset dei luoghi culturali (serve per popolare i comuni).
    """
    try:
        import streamlit as st
    except ImportError:
        raise ImportError("Streamlit non è installato. Esegui: pip install streamlit")

    st.subheader("🎯 Le tue preferenze")

    nome = st.text_input("Il tuo nome (opzionale)", placeholder="Es. Mario")

    st.markdown("#### Temi di interesse")
    st.caption("Sposta il cursore su 0 per escludere un tema, su 1 per massimizzare l'interesse.")
    temi_pesi = {}
    cols = st.columns(3)
    for i, tema in enumerate(TEMI):
        with cols[i % 3]:
            temi_pesi[tema] = st.slider(tema, 0.0, 1.0, 0.0, 0.1, key=f"tema_{tema}")

    st.markdown("#### Tipologie di luogo")
    st.caption("Seleziona le tipologie che ti interessano.")
    tipologie_pesi = {}
    cols2 = st.columns(3)
    for i, tipologia in enumerate(TIPOLOGIE):
        with cols2[i % 3]:
            checked = st.checkbox(tipologia, value=True, key=f"tipo_{tipologia}")
            tipologie_pesi[tipologia] = 1.0 if checked else 0.0

    st.markdown("#### Area geografica")
    area_tipo = st.radio(
        "Cerca per:", ["Regione", "Comune"], horizontal=True, key="area_tipo"
    )

    if area_tipo == "Regione":
        area_valore = st.selectbox("Regione", REGIONI, key="area_regione")
        area = {"tipo": "regione", "valore": area_valore}
    else:
        comuni_disponibili = sorted(df["comune"].dropna().str.title().unique().tolist())
        area_valore = st.selectbox("Comune", comuni_disponibili, key="area_comune")
        area = {"tipo": "comune", "valore": area_valore.lower()}

    raggio_km = st.slider("Raggio di ricerca (km)", 10, 300, 50, 10, key="raggio")

    solo_accessibile = st.checkbox(
        "Mostra solo luoghi accessibili ai disabili", key="accessibile"
    )

    return UserProfile(
        nome=nome,
        temi_pesi=temi_pesi,
        tipologie_pesi=tipologie_pesi,
        area=area,
        raggio_km=raggio_km,
        solo_accessibile=solo_accessibile,
    )


# ── Test standalone ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Profilo di esempio per test
    profilo = UserProfile(
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
        },
        area={"tipo": "regione", "valore": "Lazio"},
        raggio_km=80,
        solo_accessibile=False,
    )

    print("=== Profilo utente ===")
    print(profilo.summary())

    valid, msg = profilo.is_valid()
    print(f"\nProfilo valido: {valid}" + (f" — {msg}" if not valid else ""))

    vec = build_user_vector(profilo)
    print(f"\nVettore feature (shape={vec.shape}):")
    labels = TEMI + TIPOLOGIE
    for label, val in zip(labels, vec):
        if val > 0:
            print(f"  {label}: {val:.4f}")

    df = pd.read_csv("dataset_sistema.csv")
    centroid = get_area_centroid(profilo, df)
    print(f"\nCentroide area '{profilo.area['valore']}': {centroid}")