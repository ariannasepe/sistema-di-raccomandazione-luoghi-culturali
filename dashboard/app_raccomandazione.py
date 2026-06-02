"""
app_raccomandazione.py
----------------------
Interfaccia Streamlit per il sistema di raccomandazione
dei luoghi culturali italiani.

Avvio:
    streamlit run app_raccomandazione.py

Struttura:
    - Sidebar sinistra: form profilo utente
    - Colonna sinistra: mappa interattiva Folium
    - Colonna destra: schede con spiegazione raccomandazione
"""

import streamlit as st
import pandas as pd
import folium
import os
from streamlit_folium import st_folium

from user_profile import (
    UserProfile, get_area_centroid,
    TEMI, TIPOLOGIE, REGIONI, PROFILI_PREDEFINITI
)
from recommender import raccomanda, spiega_raccomandazione

LIKERT_SCALE = {
    "Per nulla": 0.0,
    "Poco": 0.25,
    "Abbastanza": 0.5,
    "Molto": 0.75,
    "Moltissimo": 1.0
}

LIKERT_LABELS = list(LIKERT_SCALE.keys())

# ── Configurazione pagina ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sistema di raccomandazione personalizzata per luoghi culturali",
    page_icon="[cultura]",
    layout="wide",
)

# ── Stile custom ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .scheda {
        background-color: #f8f9fa;
        border-left: 4px solid #4a90d9;
        border-radius: 6px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .scheda h4 {
        margin: 0 0 4px 0;
        font-size: 1rem;
        color: #1a1a2e;
    }
    .scheda .meta {
        font-size: 0.82rem;
        color: #555;
        margin-bottom: 6px;
    }
    .scheda .spiegazione {
        font-size: 0.85rem;
        color: #333;
        border-top: 1px solid #ddd;
        padding-top: 6px;
        margin-top: 6px;
    }
    .score-badge {
        display: inline-block;
        background: #4a90d9;
        color: white;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.78rem;
        font-weight: bold;
        float: right;
    }
    .tipologia-tag {
        display: inline-block;
        background: #e8f0fe;
        color: #1a73e8;
        border-radius: 4px;
        padding: 1px 8px;
        font-size: 0.78rem;
        margin-right: 4px;
    }
    .tema-tag {
        display: inline-block;
        background: #fce8e6;
        color: #c5221f;
        border-radius: 4px;
        padding: 1px 8px;
        font-size: 0.78rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Caricamento dataset ───────────────────────────────────────────────────────

import os  # ← Aggiungi questo import in cima al file (se non c'è già)

# ── Caricamento dataset ───────────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    # Ottieni il percorso assoluto della cartella dove si trova app_raccomandazione.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "dataset_sistema.csv")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Dataset caricato correttamente: {len(df)} righe da {csv_path}")
        return df
        
    except FileNotFoundError:
        st.error(f"File non trovato: {csv_path}")
        st.info("Assicurati che `dataset_sistema.csv` sia nella stessa cartella di app_raccomandazione.py")
        st.stop()
    except Exception as e:
        st.error(f"Errore nel caricamento del dataset: {e}")
        st.stop()
# ── Colori marker per tipologia ───────────────────────────────────────────────

COLORI_FOLIUM = {
    "Museo": "blue",
    "Galleria": "purple",
    "Biblioteca": "green",
    "Teatro": "orange",
    "Monumento": "gray",
    "Sito archeologico": "red",
}

# ── Sidebar: form profilo ─────────────────────────────────────────────────────

with st.sidebar:
    st.title("Sistema di raccomandazione personalizzata per luoghi culturali")
    st.divider()

    nome = st.text_input("Il tuo nome (opzionale)", placeholder="Es. Mario")

        # ── Selezione profilo predefinito ─────────────────────────────────
    st.markdown("#### Chi sei?")
    nomi_profili = list(PROFILI_PREDEFINITI.keys())
    
    # Chiave che cambia quando si seleziona un nuovo profilo
    profilo_scelto = st.selectbox(
        "Scegli il tuo profilo", 
        nomi_profili,
        key="profilo_selezionato"
    )
    
    dati_profilo = PROFILI_PREDEFINITI[profilo_scelto]
    st.caption(dati_profilo["descrizione"])

    # ── Personalizzazione avanzata (opzionale) ────────────────────────
    with st.expander("Personalizza i pesi (opzionale)"):
        st.caption("Quanto ti interessano i seguenti aspetti?")
        st.markdown("**Temi**")
        
        temi_pesi = {}
        for tema in TEMI:
            # Creiamo una chiave che include il profilo corrente → forza il reset
            widget_key = f"tema_{tema}_{profilo_scelto}"
            
            default_value = dati_profilo["temi_pesi"].get(tema, 0.5)
            
            default_label = next(
                (label for label, val in LIKERT_SCALE.items() if abs(val - default_value) < 0.01),
                "Abbastanza"
            )
            
            selezione = st.select_slider(
                label=tema,
                options=LIKERT_LABELS,
                value=default_label,
                key=widget_key
            )
            
            temi_pesi[tema] = LIKERT_SCALE[selezione]
        
        # Tipologie
        st.markdown("**Tipologie**")
        tipologie_pesi = {}
        cols = st.columns(2)
        for i, tipologia in enumerate(TIPOLOGIE):
            with cols[i % 2]:
                default = dati_profilo["tipologie_pesi"].get(tipologia, 0.0) > 0
                checked = st.checkbox(
                    tipologia, 
                    value=default, 
                    key=f"tipo_{tipologia}_{profilo_scelto}"
                )
                tipologie_pesi[tipologia] = dati_profilo["tipologie_pesi"].get(tipologia, 0.0) if checked else 0.0
    

    st.markdown("#### Area geografica")
    area_tipo = st.radio("Cerca per:", ["Regione", "Comune"], horizontal=True)

    if area_tipo == "Regione":
        area_valore = st.selectbox("Regione", REGIONI)
        area = {"tipo": "regione", "valore": area_valore}
    else:
        comuni_lista = sorted(df["comune"].dropna().str.title().unique().tolist())
        area_valore = st.selectbox("Comune", comuni_lista)
        area = {"tipo": "comune", "valore": area_valore.lower()}

    raggio_km = st.slider("Raggio di ricerca (km)", 10, 300, 50, 10)
    solo_accessibile = st.checkbox("Solo luoghi accessibili ai disabili")
    top_n = st.slider("Numero di risultati", 5, 30, 10, 5)

    st.divider()
    cerca = st.button("Cerca luoghi", use_container_width=True, type="primary")

# ── Costruzione profilo ───────────────────────────────────────────────────────

profilo = UserProfile(
    nome=nome,
    temi_pesi=temi_pesi,
    tipologie_pesi=tipologie_pesi,
    area=area,
    raggio_km=raggio_km,
    solo_accessibile=solo_accessibile,
)

# ── Header principale ─────────────────────────────────────────────────────────

st.title("Sistema di raccomandazione personalizzata per luoghi culturali")

if not cerca:
    st.info("Imposta le tue preferenze nella sidebar e clicca Cerca luoghi.")
    st.stop()

# ── Validazione profilo ───────────────────────────────────────────────────────

valid, msg = profilo.is_valid()
if not valid:
    st.warning(msg)
    st.stop()

# ── Calcolo raccomandazioni ───────────────────────────────────────────────────

with st.spinner("Calcolo raccomandazioni in corso..."):
    risultati = raccomanda(profilo, df, top_n=top_n)

if risultati.empty:
    st.error("Nessun luogo trovato con i criteri selezionati. Prova ad aumentare il raggio o a selezionare piu tipologie.")
    st.stop()

centroid = get_area_centroid(profilo, df)

# ── Intestazione risultati ────────────────────────────────────────────────────

nome_display = f"**{nome}**" if nome else "te"
st.success(
    f"Trovati **{len(risultati)}** luoghi per {nome_display} "
    f"in **{area['valore'].title()}** entro **{raggio_km} km**."
)

# ── Layout: mappa | schede ────────────────────────────────────────────────────

col_mappa, col_schede = st.columns([1.2, 1], gap="medium")

# ── Mappa Folium ──────────────────────────────────────────────────────────────

with col_mappa:
    st.markdown("### Mappa")

    mappa = folium.Map(
        location=list(centroid),
        zoom_start=9,
        tiles="CartoDB positron",
    )

    # Marker centroide area
    folium.Marker(
        location=list(centroid),
        tooltip=f"Centro ricerca: {area['valore'].title()}",
        icon=folium.Icon(color="darkblue", icon="home", prefix="fa"),
    ).add_to(mappa)

    # Cerchio raggio
    folium.Circle(
        location=list(centroid),
        radius=raggio_km * 1000,
        color="#4a90d9",
        fill=True,
        fill_opacity=0.05,
        weight=1.5,
    ).add_to(mappa)

    # Marker luoghi
    for rank, (_, row) in enumerate(risultati.iterrows(), start=1):
        colore = COLORI_FOLIUM.get(row["tipologia"], "blue")
        popup_html = f"""
        <div style='min-width:180px'>
            <b>#{rank} {row['nome']}</b><br>
            <span style='color:#555'>{row['tipologia']} · {row['comune'].title()}</span><br>
            <span style='color:#888'>{row['tema_principale']}</span><br>
            <b>Score: {row['score']:.3f}</b> · {row['distanza_km']:.1f} km
        </div>
        """
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"#{rank} {row['nome']}",
            icon=folium.Icon(color=colore, icon="info-sign"),
        ).add_to(mappa)

    st_folium(mappa, width="100%", height=520, returned_objects=[])

# ── Schede risultati ──────────────────────────────────────────────────────────

with col_schede:
    st.markdown("### Luoghi consigliati")

    for rank, (_, row) in enumerate(risultati.iterrows(), start=1):
        spiegazione = spiega_raccomandazione(row, profilo)
        accessibile = row.get("accessibile_disabili", 0) == 1

        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**#{rank} {row['nome']}**")
            with c2:
                st.markdown(
                    f"<div style='text-align:right;color:#4a90d9;font-weight:bold'>"
                    f"Score {row['score']:.3f}</div>",
                    unsafe_allow_html=True,
                )
            meta = f"{row['comune'].title()}, {row['regione']} · {row['distanza_km']:.1f} km"
            if accessibile:
                meta += " · Accessibile"
            st.caption(meta)
            st.markdown(f"`{row['tipologia']}` &nbsp; `{row['tema_principale']}`", unsafe_allow_html=True)
            st.info(spiegazione, icon=None)

# ── Tabella dati grezzi (espandibile) ────────────────────────────────────────

with st.expander("Dati completi risultati"):
    st.dataframe(
        risultati.drop(columns=["lat", "lon"], errors="ignore"),
        use_container_width=True,
        hide_index=True,
    )
