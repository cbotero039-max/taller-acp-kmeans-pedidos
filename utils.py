"""Funciones de procesamiento, ACP y K-Means (sin dependencias de Streamlit)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

DATA_PATH = "data/global_food_delivery_enriched_market_2026.csv"

# Variables cuantitativas "primarias" (medidas directamente, no derivadas por fórmula de otras)
CANDIDATAS = {
    "order_value_usd": "Valor del pedido (USD)",
    "discount_percentage": "Descuento (%)",
    "delivery_distance_km": "Distancia de entrega (km)",
    "estimated_prep_time_min": "Tiempo de preparación (min)",
    "actual_delivery_time_min": "Tiempo de trayecto (min)",
    "delivery_fee_usd": "Tarifa de envío (USD)",
    "customer_rating": "Calificación del cliente (1-5)",
    "logistics_friction_index": "Índice de fricción logística (0-10)",
    "estimated_restaurant_net_margin_usd": "Margen neto del restaurante (USD)",
}
# Variables por defecto: se excluye el margen (r = 0,98 con el valor del pedido -> redundante)
DEFECTO = [c for c in CANDIDATAS if c != "estimated_restaurant_net_margin_usd"]

# Variables excluidas del modelo y razón (para la pantalla de preparación)
EXCLUIDAS = pd.DataFrame(
    [
        ("order_id", "Identificador único: no tiene significado analítico."),
        ("cot_market_dynamics_rationale", "Texto libre generado (en indonesio); no es cuantitativo."),
        ("confidence_score", "Casi constante (0,953–0,990): sin variación útil."),
        ("net_effective_order_price", "Derivada: valor − descuento + tarifa. Duplica información."),
        ("delivery_speed_km_per_hr", "Derivada: distancia / tiempo de trayecto."),
        ("total_fulfillment_time_min", "Derivada: preparación + trayecto."),
        ("delivery_fee_ratio", "Derivada: tarifa / valor del pedido."),
        ("customer_spending_tier, customer_satisfaction_tier, primary_satisfaction_driver",
         "Etiquetas categóricas construidas a partir de las numéricas: se usan solo para interpretar."),
    ],
    columns=["Variable", "Motivo de exclusión"],
)


NOMBRE_CSV = "global_food_delivery_enriched_market_2026.csv"


def cargar_datos(path=None):
    """Busca el CSV en data/, en la raíz o en cualquier subcarpeta del repositorio."""
    base = Path(__file__).resolve().parent
    if path is None:
        candidatos = [base / "data" / NOMBRE_CSV, base / NOMBRE_CSV]
        candidatos += list(base.rglob(NOMBRE_CSV))
        for c in candidatos:
            if c.exists():
                return pd.read_csv(c)
        raise FileNotFoundError(
            f"No se encontró {NOMBRE_CSV} en el repositorio. Súbelo a la carpeta data/.")
    return pd.read_csv(path)


def estandarizar(df, cols):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[cols])
    return X, scaler


def ajustar_acp(X, cols):
    pca = PCA().fit(X)
    var = pca.explained_variance_ratio_
    tabla = pd.DataFrame({
        "Componente": [f"PC{i+1}" for i in range(len(var))],
        "Autovalor": pca.explained_variance_,
        "Varianza explicada (%)": var * 100,
        "Acumulado (%)": var.cumsum() * 100,
    })
    cargas = pd.DataFrame(
        pca.components_.T * np.sqrt(pca.explained_variance_),  # correlación variable-componente
        index=cols, columns=tabla["Componente"],
    )
    return pca, tabla, cargas


def evaluar_k(Z, k_min=2, k_max=10, semilla=42, muestra=4000):
    filas = []
    for k in range(k_min, k_max + 1):
        km = KMeans(k, n_init=10, random_state=semilla).fit(Z)
        sil = silhouette_score(Z, km.labels_, sample_size=min(muestra, len(Z)), random_state=semilla)
        filas.append((k, km.inertia_, sil))
    return pd.DataFrame(filas, columns=["K", "Inercia", "Silueta"])


def ajustar_kmeans(Z, k, semilla=42):
    km = KMeans(k, n_init=10, random_state=semilla).fit(Z)
    sil_i = silhouette_samples(Z, km.labels_)
    return km, sil_i

