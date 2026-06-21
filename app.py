"""
app.py
-------
VibroStats Enterprise — Surveillance industrielle temps réel
Capteurs : MPU6050 (accéléromètre 3 axes) + DHT11 (température / humidité)
Connexion : Arduino via port série (USB)

Détection d'anomalies :
    - Z-score glissant (domaine temporel)   -> chocs / décrochages ponctuels
    - FFT (domaine fréquentiel)             -> signatures de défauts mécaniques
"""

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis import (
    detect_temporal_anomalies,
    compute_fft,
    fft_health_metrics,
    classify_spectral_signature,
    build_system_diagnosis,
)
from serial_reader import SerialReader, list_available_ports

# ====================================================================== #
# 1. CONFIGURATION DE LA PAGE
# ====================================================================== #
st.set_page_config(
    page_title="VibroStats Enterprise",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

VIB_AXES = ["accX", "accY", "accZ"]

# ====================================================================== #
# 2. ÉTAT PERSISTANT (session_state)
# ====================================================================== #
if "reader" not in st.session_state:
    st.session_state.reader = None
if "mode" not in st.session_state:
    st.session_state.mode = "Temps réel (Arduino)"
if "csv_history" not in st.session_state:
    st.session_state.csv_history = None
if "baseline_energy" not in st.session_state:
    st.session_state.baseline_energy = None  # énergie spectrale "saine" de référence

# ====================================================================== #
# 3. BARRE LATÉRALE — PANNEAU DE CONTRÔLE
# ====================================================================== #
st.sidebar.markdown("<h2 style='color:#00ffcc; text-align:center;'>🎛️ CONTROL PANEL</h2>", unsafe_allow_html=True)
st.sidebar.write("---")

st.sidebar.subheader("📡 Source des données")
mode = st.sidebar.radio(
    "Mode d'acquisition",
    ["Temps réel (Arduino)", "Fichier CSV (log)"],
    index=0 if st.session_state.mode == "Temps réel (Arduino)" else 1,
)
st.session_state.mode = mode

if mode == "Temps réel (Arduino)":
    available_ports = list_available_ports()
    port_options = available_ports if available_ports else ["Aucun port détecté"]
    selected_port = st.sidebar.selectbox("Port série", port_options)
    baudrate = st.sidebar.selectbox("Baudrate", [9600, 19200, 38400, 57600, 115200], index=0)
    max_buffer = st.sidebar.slider("Taille du buffer (échantillons)", 200, 10000, 2000, 100)

    col_a, col_b = st.sidebar.columns(2)
    start_clicked = col_a.button("▶️ Connecter", use_container_width=True)
    stop_clicked = col_b.button("⏹️ Arrêter", use_container_width=True)

    if start_clicked and available_ports:
        if st.session_state.reader:
            st.session_state.reader.stop()
        st.session_state.reader = SerialReader(selected_port, baudrate, max_samples=max_buffer)
        st.session_state.reader.start()

    if stop_clicked and st.session_state.reader:
        st.session_state.reader.stop()
        st.session_state.reader = None

    refresh_rate = st.sidebar.slider("Rafraîchissement (secondes)", 1, 10, 2)
    auto_refresh = st.sidebar.checkbox("Auto-refresh activé", value=True)

else:
    uploaded_file = st.sidebar.file_uploader("📂 Charger un log CSV", type="csv")
    if uploaded_file:
        st.session_state.csv_history = pd.read_csv(uploaded_file)

st.sidebar.write("---")
st.sidebar.subheader("🎯 Réglages de détection")
sensitivity = st.sidebar.slider("Seuil Z-score (anomalie ponctuelle)", 1.0, 6.0, 3.0, 0.1)
window_size = st.sidebar.slider("Fenêtre glissante (échantillons)", 5, 200, 20, 5)
sample_rate = st.sidebar.number_input(
    "Fréquence d'échantillonnage (Hz)", min_value=1.0, max_value=1000.0, value=20.0, step=1.0,
    help="Nombre d'échantillons MPU6050 envoyés par seconde par l'Arduino. Nécessaire pour calibrer l'axe fréquentiel de la FFT."
)
energy_ratio_threshold = st.sidebar.slider("Seuil ratio d'énergie FFT (vs baseline)", 1.2, 5.0, 2.0, 0.1)

if st.sidebar.button("📌 Définir la fenêtre actuelle comme BASELINE saine"):
    st.session_state.baseline_energy = "PENDING"  # sera calculé après le prochain calcul FFT

st.sidebar.write("---")
st.sidebar.subheader("🏭 Informations sur l'actif")
asset_name = st.sidebar.text_input("Machine ID / Tag", "MOTOR_COMP_042")
location_tag = st.sidebar.text_input("Localisation usine", "Zone A - Ligne principale")
temp_high_limit = st.sidebar.number_input("Seuil température critique (°C)", value=45.0)
humidity_high_limit = st.sidebar.number_input("Seuil humidité critique (%)", value=80.0)

# ====================================================================== #
# 4. EN-TÊTE
# ====================================================================== #
st.markdown("<h1>⚡ VIBROSTATS ENTERPRISE <span style='font-size:16px; color:#888;'>v4.0 — Real-Time Edition</span></h1>", unsafe_allow_html=True)
st.markdown(f"**Actif :** `{asset_name}` | **Localisation :** `{location_tag}` | **Capteurs :** MPU6050 + DHT11")
st.markdown("---")

# ====================================================================== #
# 5. RÉCUPÉRATION DES DONNÉES (temps réel ou CSV)
# ====================================================================== #
df = None
is_live = False

if mode == "Temps réel (Arduino)":
    reader = st.session_state.reader
    if reader is None:
        st.info("📡 Sélectionnez un port et cliquez sur **Connecter** pour démarrer l'acquisition Arduino.")
    elif reader.last_error and not reader.connected:
        st.error(f"❌ {reader.last_error}")
    else:
        samples = reader.get_dataframe()
        if len(samples) < 5:
            st.warning(f"⏳ En attente de données... ({len(samples)} échantillons reçus)")
        else:
            df = pd.DataFrame(samples)
            is_live = True
            status_cols = st.columns(4)
            status_cols[0].metric("Statut connexion", "🟢 EN LIGNE" if reader.connected else "🔴 HORS LIGNE")
            status_cols[1].metric("Échantillons (buffer)", reader.sample_count())
            status_cols[2].metric("Lignes mal formées", reader.malformed_lines)
            status_cols[3].metric("Dernier échantillon", reader.last_sample_time.strftime("%H:%M:%S") if reader.last_sample_time else "—")
else:
    if st.session_state.csv_history is not None:
        df = st.session_state.csv_history.copy()
    else:
        st.markdown(
            "<div style='text-align: center; padding: 50px; border: 2px dashed #334; border-radius: 10px; background: #111625;'>"
            "<h3 style='color: #00ffcc;'>Chargez un fichier CSV pour rejouer un historique de télémétrie</h3></div>",
            unsafe_allow_html=True,
        )

# ====================================================================== #
# 6. PIPELINE D'ANALYSE
# ====================================================================== #
if df is not None and len(df) >= 5:

    missing_axes = [a for a in VIB_AXES if a not in df.columns]
    if missing_axes:
        st.error(f"❌ Colonnes vibratoires manquantes dans les données : {missing_axes}. "
                  f"Vérifiez le format envoyé par l'Arduino (attendu : accX,accY,accZ,temp,humidity).")
        st.stop()

    present_axes = [a for a in VIB_AXES if a in df.columns]

    # --- 6.1 Détection temporelle (Z-score) ---
    df = detect_temporal_anomalies(df, present_axes, window_size, sensitivity)

    max_z = float(df["Z_Score"].max())
    anomaly_points = int(df["Anomaly_Temporal"].sum())
    anomaly_percentage = (anomaly_points / len(df)) * 100

    has_temp = "temp" in df.columns
    has_humidity = "humidity" in df.columns
    avg_temp = float(df["temp"].mean()) if has_temp else None
    avg_humidity = float(df["humidity"].mean()) if has_humidity else None

    # --- 6.2 Détection fréquentielle (FFT) sur l'axe de plus forte énergie ---
    primary_axis = df["vib_magnitude"] if "vib_magnitude" in df.columns else df[present_axes[0]]
    freqs, spectrum = compute_fft(primary_axis.to_numpy(), sample_rate)

    if st.session_state.baseline_energy == "PENDING":
        _, _spec = compute_fft(primary_axis.to_numpy(), sample_rate)
        st.session_state.baseline_energy = float(np.sum(_spec ** 2)) if len(_spec) else None
        st.sidebar.success("✅ Baseline spectrale enregistrée.")

    fft_metrics = fft_health_metrics(freqs, spectrum, st.session_state.baseline_energy)
    fft_diag, fft_severity = classify_spectral_signature(fft_metrics, energy_ratio_threshold)

    # --- 6.3 Diagnostic système global ---
    status, findings, recommendations = build_system_diagnosis(
        df, fft_diag, fft_severity, anomaly_percentage, max_z,
        temp_series=df["temp"] if has_temp else None,
        humidity_series=df["humidity"] if has_humidity else None,
        temp_high_limit=temp_high_limit,
        humidity_high_limit=humidity_high_limit,
    )

    # ================================================================== #
    # 7. KPI CARDS
    # ================================================================== #
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("Pic d'anomalie (Z)", f"{max_z:.2f} σ",
                   delta="CRITIQUE" if max_z > sensitivity else "NOMINAL", delta_color="inverse")
    with kpi2:
        st.metric("Taux de déviation", f"{anomaly_percentage:.1f}%",
                   delta="Action requise" if anomaly_percentage > 5 else "Acceptable")
    with kpi3:
        st.metric("Température moy.", f"{avg_temp:.1f} °C" if has_temp else "N/A",
                   delta=f"Max: {df['temp'].max():.1f}°C" if has_temp else "Capteur absent")
    with kpi4:
        st.metric("Humidité moy.", f"{avg_humidity:.1f} %" if has_humidity else "N/A",
                   delta=f"Max: {df['humidity'].max():.1f}%" if has_humidity else "Capteur absent")
    with kpi5:
        emoji = {"NOMINAL": "✅", "SURVEILLANCE RENFORCÉE": "🟡", "CRITIQUE": "🚨"}[status]
        css_class = {"NOMINAL": "status-ok", "SURVEILLANCE RENFORCÉE": "status-warning", "CRITIQUE": "status-alert"}[status]
        st.markdown(f"<div class='status-box {css_class}'>{emoji} {status}</div>", unsafe_allow_html=True)

    st.write("##")

    # ================================================================== #
    # 8. ONGLETS
    # ================================================================== #
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Forme d'onde temps réel",
        "🔬 Analyse spectrale (FFT)",
        "🌡️ Conditions ambiantes",
        "📋 Diagnostic & Rapport",
    ])

    # --- TAB 1 : Waveform temporel ---
    with tab1:
        st.subheader("Visualisation vibratoire haute fréquence")
        fig = go.Figure()
        colors = {"accX": "#00ffcc", "accY": "#ff007f", "accZ": "#ffcc00"}
        x_axis = df["timestamp"] if "timestamp" in df.columns else df.index

        for col in present_axes:
            fig.add_trace(go.Scatter(x=x_axis, y=df[col], mode="lines", name=f"Vibration {col}",
                                      line=dict(color=colors.get(col, "#fff"), width=1.3)))

        anomalies = df[df["Anomaly_Temporal"]]
        if not anomalies.empty:
            anomaly_x = anomalies["timestamp"] if "timestamp" in anomalies.columns else anomalies.index
            fig.add_trace(go.Scatter(x=anomaly_x, y=anomalies[present_axes[0]], mode="markers",
                                      name="Anomalie détectée", marker=dict(color="#ff4b4b", size=8, symbol="x")))

        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           margin=dict(l=20, r=20, t=20, b=20),
                           xaxis=dict(showgrid=True, gridcolor="#223"), yaxis=dict(showgrid=True, gridcolor="#223"))
        st.plotly_chart(fig, use_container_width=True)

    # --- TAB 2 : FFT ---
    with tab2:
        col_l, col_r = st.columns([2, 1])
        with col_l:
            st.subheader("Spectre fréquentiel (magnitude vibratoire)")
            if len(freqs) > 0:
                fig_fft = go.Figure()
                fig_fft.add_trace(go.Scatter(x=freqs, y=spectrum, mode="lines", fill="tozeroy",
                                              line=dict(color="#00ffcc", width=1.5)))
                fig_fft.add_vline(x=fft_metrics["dominant_freq"], line_dash="dash", line_color="#ff4b4b",
                                   annotation_text=f"Pic dominant : {fft_metrics['dominant_freq']:.1f} Hz")
                fig_fft.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                       margin=dict(l=20, r=20, t=20, b=20),
                                       xaxis_title="Fréquence (Hz)", yaxis_title="Amplitude")
                st.plotly_chart(fig_fft, use_container_width=True)
            else:
                st.info("Pas assez d'échantillons pour calculer une FFT fiable (minimum ~8 points).")

        with col_r:
            st.subheader("Indicateurs spectraux")
            st.metric("Fréquence dominante", f"{fft_metrics['dominant_freq']:.2f} Hz")
            st.metric("Amplitude dominante", f"{fft_metrics['dominant_amp']:.4f}")
            st.metric("Énergie spectrale totale", f"{fft_metrics['total_energy']:.4f}")
            if st.session_state.baseline_energy and st.session_state.baseline_energy != "PENDING":
                st.metric("Ratio vs baseline", f"{fft_metrics['energy_ratio']:.2f}x")
            else:
                st.caption("ℹ️ Définissez une baseline (panneau latéral) pour activer le ratio comparatif.")

            severity_color = {"ok": "🟢", "warning": "🟡", "critical": "🔴"}[fft_severity]
            st.markdown(f"**Diagnostic spectral :** {severity_color} {fft_diag}")

    # --- TAB 3 : Conditions ambiantes ---
    with tab3:
        col_t, col_h = st.columns(2)
        x_axis = df["timestamp"] if "timestamp" in df.columns else df.index

        with col_t:
            st.subheader("Évolution de la température (DHT11)")
            if has_temp:
                fig_t = go.Figure()
                fig_t.add_trace(go.Scatter(x=x_axis, y=df["temp"], mode="lines", line=dict(color="#ff9900", width=1.5)))
                fig_t.add_hline(y=temp_high_limit, line_dash="dash", line_color="#ff4b4b", annotation_text="Seuil critique")
                fig_t.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                     margin=dict(l=20, r=20, t=20, b=20), yaxis_title="°C")
                st.plotly_chart(fig_t, use_container_width=True)
            else:
                st.info("Capteur de température non détecté dans les données.")

        with col_h:
            st.subheader("Évolution de l'humidité (DHT11)")
            if has_humidity:
                fig_h = go.Figure()
                fig_h.add_trace(go.Scatter(x=x_axis, y=df["humidity"], mode="lines", line=dict(color="#3399ff", width=1.5)))
                fig_h.add_hline(y=humidity_high_limit, line_dash="dash", line_color="#ff4b4b", annotation_text="Seuil critique")
                fig_h.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                     margin=dict(l=20, r=20, t=20, b=20), yaxis_title="%")
                st.plotly_chart(fig_h, use_container_width=True)
            else:
                st.info("Capteur d'humidité non détecté dans les données.")

        st.subheader("Corrélation thermique / contrainte mécanique")
        if has_temp:
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(x=df["temp"], y=primary_axis, mode="markers",
                                              marker=dict(color=df["Z_Score"], colorscale="Viridis", showscale=True, size=6)))
            fig_scatter.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                       margin=dict(l=20, r=20, t=20, b=20),
                                       xaxis_title="Température (°C)", yaxis_title="Magnitude vibratoire")
            st.plotly_chart(fig_scatter, use_container_width=True)

    # --- TAB 4 : Diagnostic & rapport ---
    with tab4:
        st.subheader(f"Description du système — Statut global : {status}")

        st.markdown("**Constats :**")
        for f in findings:
            st.markdown(f"- {f}")

        st.markdown("**Recommandations de maintenance industrielle :**")
        for r in recommendations:
            st.markdown(f"- {r}")

        st.write("---")
        st.markdown("**Journal des anomalies vibratoires détectées (Z-score)**")
        anomaly_log_cols = present_axes + ["Z_Score"]
        if has_temp:
            anomaly_log_cols.append("temp")
        if has_humidity:
            anomaly_log_cols.append("humidity")

        anomaly_log = df[df["Anomaly_Temporal"]][anomaly_log_cols].tail(100)
        if not anomaly_log.empty:
            st.dataframe(anomaly_log.style.format(precision=3), use_container_width=True)
        else:
            st.success("🎉 Aucune anomalie temporelle détectée sur la fenêtre actuelle.")

        st.download_button(
            "📥 Télécharger le journal complet (CSV)",
            data=df.to_csv(index=False),
            file_name=f"Diagnostic_{asset_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

# ====================================================================== #
# 9. AUTO-REFRESH EN MODE TEMPS RÉEL
# ====================================================================== #
if mode == "Temps réel (Arduino)" and is_live and auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
