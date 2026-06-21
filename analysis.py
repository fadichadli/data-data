"""
analysis.py
------------
Logique de détection d'anomalies pour la maintenance industrielle prédictive.

Deux familles de détection, complémentaires :

1. Z-SCORE GLISSANT (domaine temporel)
   - Détecte les écarts ponctuels / pics brusques par rapport à la moyenne
     mobile locale. Sensible aux chocs, impacts, décrochages soudains.

2. FFT / ANALYSE FRÉQUENTIELLE (domaine fréquentiel)
   - Détecte les signatures de défauts mécaniques classiques :
       * Déséquilibre rotor      -> pic à 1x la fréquence de rotation
       * Désalignement           -> pics à 1x et 2x
       * Défaut de roulement     -> pics à hautes fréquences, large bande
       * Résonance structurelle  -> pic isolé très fin et très haut
   - Calcule l'énergie spectrale totale, comparée à une baseline "saine".
"""

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq


# ---------------------------------------------------------------------- #
# Z-SCORE (domaine temporel)
# ---------------------------------------------------------------------- #
def compute_zscore(series: pd.Series, window_size: int) -> pd.Series:
    """Z-score glissant : (valeur - moyenne locale) / écart-type local."""
    rolling_mean = series.rolling(window=window_size, min_periods=1).mean()
    rolling_std = series.rolling(window=window_size, min_periods=1).std().fillna(1e-4)
    rolling_std = rolling_std.replace(0, 1e-4)
    return (series - rolling_mean) / rolling_std


def detect_temporal_anomalies(df: pd.DataFrame, axes: list, window_size: int, threshold: float) -> pd.DataFrame:
    """
    Calcule un Z-score par axe vibratoire et un Z-score combiné (norme),
    puis marque les échantillons hors-seuil.
    """
    for axis in axes:
        df[f"Z_{axis}"] = compute_zscore(df[axis], window_size)

    # Magnitude vibratoire combinée (norme euclidienne des 3 axes si dispo)
    if all(a in df.columns for a in ["accX", "accY", "accZ"]):
        df["vib_magnitude"] = np.sqrt(df["accX"] ** 2 + df["accY"] ** 2 + df["accZ"] ** 2)
        df["Z_magnitude"] = compute_zscore(df["vib_magnitude"], window_size)
        z_cols = [f"Z_{a}" for a in axes] + ["Z_magnitude"]
    else:
        z_cols = [f"Z_{a}" for a in axes]

    df["Z_Score"] = df[z_cols].abs().max(axis=1)
    df["Anomaly_Temporal"] = df["Z_Score"] > threshold
    return df


# ---------------------------------------------------------------------- #
# FFT (domaine fréquentiel)
# ---------------------------------------------------------------------- #
def compute_fft(signal: np.ndarray, sample_rate: float):
    """
    Calcule le spectre d'amplitude (FFT réelle, normalisée) d'un signal.
    Retourne (fréquences, amplitudes) en ignorant la composante DC (0 Hz).
    """
    n = len(signal)
    if n < 8:
        return np.array([]), np.array([])

    signal = signal - np.mean(signal)  # retrait de la composante continue
    window = np.hanning(n)             # fenêtrage pour réduire les fuites spectrales
    spectrum = np.abs(rfft(signal * window)) * (2.0 / n)
    freqs = rfftfreq(n, d=1.0 / sample_rate)

    # On ignore le tout premier bin (DC / très basse fréquence résiduelle)
    return freqs[1:], spectrum[1:]


def fft_health_metrics(freqs: np.ndarray, spectrum: np.ndarray, baseline_energy=None):
    """
    Extrait des indicateurs de santé mécanique depuis le spectre :
    - fréquence dominante et son amplitude
    - énergie spectrale totale
    - ratio par rapport à une baseline (si fournie) -> indique une dérive
    """
    if len(spectrum) == 0:
        return {
            "dominant_freq": 0.0,
            "dominant_amp": 0.0,
            "total_energy": 0.0,
            "energy_ratio": 1.0,
        }

    dominant_idx = int(np.argmax(spectrum))
    total_energy = float(np.sum(spectrum ** 2))
    ratio = total_energy / baseline_energy if baseline_energy and baseline_energy > 0 else 1.0

    return {
        "dominant_freq": float(freqs[dominant_idx]),
        "dominant_amp": float(spectrum[dominant_idx]),
        "total_energy": total_energy,
        "energy_ratio": ratio,
    }


def classify_spectral_signature(metrics: dict, energy_ratio_threshold: float = 2.0):
    """
    Heuristique simple de diagnostic mécanique basée sur la signature spectrale.
    Sert de première lecture explicative pour l'équipe maintenance — à affiner
    avec les seuils réels de la machine surveillée.
    """
    freq = metrics["dominant_freq"]
    ratio = metrics["energy_ratio"]

    if ratio < energy_ratio_threshold:
        return "Spectre nominal — aucune signature de défaut dominante", "ok"

    if freq < 5:
        return "Énergie basse fréquence élevée — possible désalignement ou jeu mécanique", "warning"
    elif freq < 50:
        return "Pic en moyenne fréquence — possible déséquilibre rotor", "warning"
    else:
        return "Énergie haute fréquence anormale — possible défaut de roulement / engrenage", "critical"


# ---------------------------------------------------------------------- #
# DIAGNOSTIC GLOBAL (synthèse temporel + fréquentiel + thermique)
# ---------------------------------------------------------------------- #
def build_system_diagnosis(df: pd.DataFrame, fft_diag: str, fft_severity: str,
                            anomaly_rate: float, max_z: float,
                            temp_series: pd.Series = None, humidity_series: pd.Series = None,
                            temp_high_limit: float = 45.0, humidity_high_limit: float = 80.0):
    """
    Construit une description textuelle du système à protéger, en croisant :
    - le taux d'anomalies vibratoires temporelles
    - la signature spectrale (FFT)
    - les conditions ambiantes (température / humidité, via DHT11)

    Retourne (statut_global, constats, recommandations).
    """
    findings = []
    severity_score = 0  # 0 = sain, croît avec la gravité

    # --- Volet vibratoire temporel ---
    if anomaly_rate > 10 or max_z > 6:
        findings.append("Taux d'anomalies vibratoires élevé : choc(s) ou décrochage mécanique probable.")
        severity_score += 2
    elif anomaly_rate > 3:
        findings.append("Anomalies vibratoires ponctuelles détectées : à surveiller sur les prochains cycles.")
        severity_score += 1
    else:
        findings.append("Comportement vibratoire temporel stable.")

    # --- Volet spectral ---
    findings.append(f"Analyse spectrale (FFT) : {fft_diag}.")
    if fft_severity == "critical":
        severity_score += 2
    elif fft_severity == "warning":
        severity_score += 1

    # --- Volet thermique (DHT11) ---
    if temp_series is not None and len(temp_series) > 0:
        max_temp = float(temp_series.max())
        if max_temp > temp_high_limit:
            findings.append(f"Température maximale {max_temp:.1f}°C dépasse le seuil de {temp_high_limit:.0f}°C : risque de surchauffe.")
            severity_score += 2
        else:
            findings.append(f"Température dans la plage normale (max {max_temp:.1f}°C).")

    if humidity_series is not None and len(humidity_series) > 0:
        max_hum = float(humidity_series.max())
        if max_hum > humidity_high_limit:
            findings.append(f"Humidité maximale {max_hum:.1f}% dépasse {humidity_high_limit:.0f}% : risque pour l'électronique embarquée.")
            severity_score += 1
        else:
            findings.append(f"Humidité ambiante normale (max {max_hum:.1f}%).")

    # --- Statut global et recommandations ---
    if severity_score >= 4:
        status = "CRITIQUE"
        recommendations = [
            "Arrêt programmé recommandé pour inspection physique immédiate.",
            "Vérifier l'état des roulements, l'alignement de l'arbre et le serrage des fixations.",
            "Contrôler le système de refroidissement / ventilation si surchauffe confirmée.",
        ]
    elif severity_score >= 2:
        status = "SURVEILLANCE RENFORCÉE"
        recommendations = [
            "Planifier une inspection préventive dans les prochains jours.",
            "Augmenter la fréquence d'échantillonnage ou la durée des relevés.",
            "Comparer la signature spectrale actuelle à l'historique de la machine.",
        ]
    else:
        status = "NOMINAL"
        recommendations = [
            "Aucune action immédiate requise.",
            "Poursuivre la surveillance continue selon le plan de maintenance prédictive.",
        ]

    return status, findings, recommendations
