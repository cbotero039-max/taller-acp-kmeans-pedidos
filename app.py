import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (CANDIDATAS, DEFECTO, EXCLUIDAS, cargar_datos, estandarizar,
                   ajustar_acp, evaluar_k, ajustar_kmeans)

st.set_page_config(page_title="Pedidos de comida a domicilio · ACP y K-Means",
                   page_icon="🛵", layout="wide")

# ----------------------------------------------------------------- datos
@st.cache_data
def datos():
    return cargar_datos()

@st.cache_data
def calcular_acp(cols):
    df = datos()
    X, _ = estandarizar(df, list(cols))
    pca, tabla, cargas = ajustar_acp(X, list(cols))
    scores = pca.transform(X)
    return tabla, cargas, scores

@st.cache_data
def calcular_k(cols, n_comp):
    _, _, scores = calcular_acp(cols)
    return evaluar_k(scores[:, :n_comp])

@st.cache_data
def calcular_grupos(cols, n_comp, k):
    _, _, scores = calcular_acp(cols)
    km, sil_i = ajustar_kmeans(scores[:, :n_comp], k)
    return km.labels_, sil_i

df = datos()

# ----------------------------------------------------------------- barra lateral
st.sidebar.title("🛵 Taller ACP y K-Means")
pagina = st.sidebar.radio("Navegación", [
    "1. Contexto", "2. Exploración", "3. Preparación", "4. ACP",
    "5. K-Means", "6. Discusión y conclusiones"])

st.sidebar.markdown("---")
st.sidebar.subheader("Variables del modelo")
nombres = {v: k for k, v in CANDIDATAS.items()}
elegidas_nom = st.sidebar.multiselect(
    "Variables cuantitativas", list(CANDIDATAS.values()),
    default=[CANDIDATAS[c] for c in DEFECTO])
cols = tuple(nombres[n] for n in elegidas_nom)
if len(cols) < 4:
    st.sidebar.error("Selecciona al menos 4 variables.")
    st.error("El taller exige al menos cuatro variables cuantitativas. Elige más variables en la barra lateral.")
    st.stop()
if {"order_value_usd", "estimated_restaurant_net_margin_usd"} <= set(cols):
    st.sidebar.warning("Valor del pedido y margen neto tienen r = 0,98: están midiendo casi lo mismo "
                       "y el ACP les dará doble peso.")

tabla, cargas, scores = calcular_acp(cols)
etq = CANDIDATAS  # etiquetas legibles

# ================================================================= 1. CONTEXTO
if pagina == "1. Contexto":
    st.title("¿Qué tipos de pedido existen en el mercado global de comida a domicilio?")
    st.markdown("""
**Pregunta guía.** Un pedido de comida combina decisiones del cliente (cuánto gasta, qué descuento usa),
condiciones logísticas (distancia, tiempos, tarifa) y resultados (calificación). ¿Esas variables se
resumen en pocas dimensiones? ¿Hay *tipos* de pedido con perfiles distintos que justifiquen estrategias
diferentes (zonas de reparto, tarifas, promociones)?

**Unidad de observación.** Cada fila es **un pedido** (8.000 pedidos en 12 ciudades: Nueva York, Tokio,
Yakarta, Londres, Singapur, París, Sídney, Berlín, São Paulo, Seúl, Bombay y Toronto), con tres canales:
app de domicilio, recogida en local y consumo en el restaurante.

**Fuente.** `global_food_delivery_enriched_market_2026.csv` (26 columnas). Incluye variables medidas
(valor, distancia, tiempos, tarifa, calificación), variables derivadas por fórmula y etiquetas
categóricas.
""")
    st.subheader("¿Por qué ACP y K-Means?")
    c1, c2 = st.columns(2)
    c1.info("**ACP.** Las variables logísticas (distancia, tiempo de trayecto, tarifa, fricción) se mueven "
            "juntas. El ACP permite resumirlas en un eje común y verificar si el valor económico del pedido "
            "es un eje separado, además de eliminar la redundancia antes de agrupar.")
    c2.info("**K-Means.** No hay una etiqueta de “tipo de pedido” que predecir. K-Means busca grupos "
            "de pedidos con perfil similar usando distancias; por eso se aplica sobre componentes "
            "estandarizados, donde ninguna escala domina.")
    st.subheader("Limitaciones de la fuente")
    st.warning("""
- Los datos parecen **sintéticos/simulados** (el texto explicativo de cada fila es generado y las columnas
  derivadas siguen fórmulas exactas): los hallazgos describen la estructura de *esta base*, no el mercado real.
- En recogida y consumo en local la distancia, el trayecto y la tarifa valen 0 **por definición**, no por
  falta de dato. Esto crea una división estructural entre “con domicilio” y “sin domicilio”.
- Los montos están en USD sin ajuste por poder adquisitivo entre ciudades.
- Un pedido no es un cliente: no se puede seguir a una persona en el tiempo.
""")

# ================================================================= 2. EXPLORACIÓN
elif pagina == "2. Exploración":
    st.title("Exploración de los datos")
    a, b, c, d = st.columns(4)
    a.metric("Pedidos", f"{len(df):,}")
    b.metric("Columnas", df.shape[1])
    c.metric("Valores faltantes", int(df.isna().sum().sum()))
    d.metric("Filas duplicadas", int(df.duplicated().sum()))

    st.subheader("Vista previa")
    st.dataframe(df.drop(columns=["cot_market_dynamics_rationale"]).head(20), use_container_width=True)

    st.subheader("Estadísticas descriptivas de las variables seleccionadas")
    desc = df[list(cols)].describe().T.rename(index=etq).round(2)
    st.dataframe(desc, use_container_width=True)

    st.subheader("Distribuciones")
    v = st.selectbox("Variable", cols, format_func=lambda x: etq[x])
    cat = st.selectbox("Colorear por", ["(ninguno)", "order_channel", "metro_city", "cuisine_category",
                                         "weather_condition"])
    fig = px.histogram(df, x=v, color=None if cat == "(ninguno)" else cat, nbins=50, barmode="overlay",
                       opacity=0.7, labels={v: etq[v]}, title=f"Distribución de: {etq[v]}")
    fig.update_yaxes(title="Número de pedidos")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Las colas largas (pedidos de alto valor, trayectos muy largos) y el pico en cero de las "
               "variables logísticas (recogida y consumo en local) condicionan todo el análisis posterior.")

    st.subheader("Matriz de correlaciones")
    corr = df[list(cols)].corr().rename(index=etq, columns=etq)
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="Correlación de Pearson entre variables seleccionadas")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Se ven dos bloques: (i) distancia, tiempo de trayecto y tarifa, muy correlacionados; "
               "(ii) valor del pedido y margen, casi idénticos. El descuento y el tiempo de preparación "
               "son casi independientes del resto.")

# ================================================================= 3. PREPARACIÓN
elif pagina == "3. Preparación":
    st.title("Preparación de los datos")
    st.markdown("**Calidad:** la base no tiene valores faltantes ni duplicados (ver pantalla 2). "
                "Las decisiones principales fueron de *selección de variables*:")
    st.dataframe(EXCLUIDAS, use_container_width=True, hide_index=True)
    st.markdown("""
**Valores atípicos.** Se conservaron: pedidos grandes (hasta USD 145) y trayectos de hasta 75 min son
comportamientos reales del negocio y justamente lo que interesa segmentar. Se anota que pueden jalar
los componentes.

**Ceros estructurales.** No se imputan ni eliminan: en recogida y consumo en local un 0 en distancia es
información válida.

**Margen neto.** Se excluye por defecto porque su correlación con el valor del pedido es 0,98. Puedes
incluirlo en la barra lateral y ver cómo cambia el ACP.
""")
    st.subheader("Estandarización (media 0, desviación 1)")
    X, _ = estandarizar(df, list(cols))
    antes = df[list(cols)].agg(["mean", "std"]).T.rename(index=etq)
    despues = pd.DataFrame(X, columns=cols).agg(["mean", "std"]).T.rename(index=etq)
    c1, c2 = st.columns(2)
    c1.write("**Antes** (escalas muy distintas)")
    c1.dataframe(antes.round(2))
    c2.write("**Después**")
    c2.dataframe(despues.round(2))
    st.caption("Sin estandarizar, variables como el valor del pedido (USD) o el tiempo (min) dominarían "
               "la varianza y las distancias solo por tener unidades más grandes.")

# ================================================================= 4. ACP
elif pagina == "4. ACP":
    st.title("Análisis de Componentes Principales")
    st.subheader("Varianza explicada")
    st.dataframe(tabla.round(2), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    fig = go.Figure()
    fig.add_bar(x=tabla["Componente"], y=tabla["Varianza explicada (%)"], name="Individual")
    fig.add_scatter(x=tabla["Componente"], y=tabla["Acumulado (%)"], name="Acumulada", mode="lines+markers")
    fig.add_hline(y=80, line_dash="dot", annotation_text="80 %")
    fig.update_layout(title="Varianza explicada por componente", yaxis_title="% de varianza")
    c1.plotly_chart(fig, use_container_width=True)
    fig = go.Figure(go.Scatter(x=tabla["Componente"], y=tabla["Autovalor"], mode="lines+markers"))
    fig.add_hline(y=1, line_dash="dot", annotation_text="Criterio de Kaiser (autovalor = 1)")
    fig.update_layout(title="Gráfico de sedimentación (scree plot)", yaxis_title="Autovalor")
    c2.plotly_chart(fig, use_container_width=True)

    n80 = int((tabla["Acumulado (%)"] < 80).sum() + 1)
    nk = int((tabla["Autovalor"] > 1).sum())
    st.info(f"Con estas variables: Kaiser sugiere **{nk}** componentes; para acumular ≥ 80 % se necesitan "
            f"**{n80}**. Elige cuántos conservar según el compromiso entre simplicidad y pérdida de información.")
    n_comp = st.slider("Componentes a conservar", 2, len(cols), min(n80, len(cols)), key="ncomp")
    st.success(f"Los {n_comp} primeros componentes conservan "
               f"**{tabla['Acumulado (%)'].iloc[n_comp-1]:.1f} %** de la varianza original.")

    st.subheader("Cargas (correlación entre variable y componente)")
    c = cargas.iloc[:, :n_comp].rename(index=etq)
    fig = px.imshow(c, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                    title="Cargas de las variables en cada componente")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Lectura: una carga cercana a ±1 indica que la variable define ese componente; el signo "
               "indica la dirección. El signo global de un componente es arbitrario.")
    for i in range(min(n_comp, 4)):
        s = cargas.iloc[:, i]
        top = s.reindex(s.abs().sort_values(ascending=False).index).head(3)
        txt = ", ".join(f"{etq[k]} ({v:+.2f})" for k, v in top.items())
        st.markdown(f"- **PC{i+1}** ({tabla['Varianza explicada (%)'].iloc[i]:.1f} %): {txt}")

    st.subheader("Pedidos en el espacio de componentes")
    ex, ey = st.columns(2)
    px_ = ex.selectbox("Eje X", [f"PC{i+1}" for i in range(n_comp)], 0)
    py_ = ey.selectbox("Eje Y", [f"PC{i+1}" for i in range(n_comp)], 1)
    ix, iy = int(px_[2:]) - 1, int(py_[2:]) - 1
    color = st.selectbox("Colorear por", ["order_channel", "metro_city", "cuisine_category",
                                           "weather_condition"], key="col_acp")
    muestra = df.assign(X=scores[:, ix], Y=scores[:, iy]).sample(3000, random_state=1)
    fig = px.scatter(muestra, x="X", y="Y", color=color, opacity=0.6,
                     labels={"X": px_, "Y": py_},
                     title=f"{px_} vs {py_} (muestra de 3.000 pedidos)")
    st.plotly_chart(fig, use_container_width=True)
    pct = tabla["Varianza explicada (%)"].iloc[[ix, iy]].sum()
    st.caption(f"Este plano resume solo el **{pct:.1f} %** de la información original: lo que no se separa "
               "aquí puede estar separado en otros componentes.")

# ================================================================= 5. K-MEANS
elif pagina == "5. K-Means":
    st.title("Segmentación con K-Means")
    n_comp = st.sidebar.slider("Componentes usados en K-Means", 2, len(cols),
                               min(4, len(cols)), key="ncomp_km")
    st.caption(f"K-Means se aplica sobre los {n_comp} primeros componentes "
               f"({tabla['Acumulado (%)'].iloc[n_comp-1]:.1f} % de la varianza).")

    st.subheader("¿Cuántos grupos? Codo y silueta")
    ev = calcular_k(cols, n_comp)
    c1, c2 = st.columns(2)
    fig = px.line(ev, x="K", y="Inercia", markers=True, title="Método del codo")
    c1.plotly_chart(fig, use_container_width=True)
    fig = px.line(ev, x="K", y="Silueta", markers=True, title="Coeficiente de silueta promedio")
    c2.plotly_chart(fig, use_container_width=True)
    mejor = int(ev.loc[ev["Silueta"].idxmax(), "K"])
    st.info(f"La silueta máxima ocurre en **K = {mejor}** ({ev['Silueta'].max():.2f}). Valores por debajo de "
            "≈0,25 indican estructura débil. La elección final debe equilibrar silueta, codo e interpretabilidad.")
    k = st.slider("Número de grupos (K)", 2, 10, 4)

    etiquetas, sil_i = calcular_grupos(cols, n_comp, k)
    d = df.copy()
    d["Grupo"] = pd.Series(etiquetas, index=d.index).map(lambda x: f"Grupo {x+1}")
    d["silueta"] = sil_i
    orden = sorted(d["Grupo"].unique())

    c1, c2, c3 = st.columns(3)
    c1.metric("Silueta promedio", f"{sil_i.mean():.3f}")
    c2.metric("Pedidos con silueta < 0", f"{(sil_i < 0).mean()*100:.1f} %")
    c3.metric("Grupo más pequeño", f"{d['Grupo'].value_counts().min():,} pedidos")

    st.subheader("Visualización de los grupos")
    ex, ey = st.columns(2)
    px_ = ex.selectbox("Eje X", [f"PC{i+1}" for i in range(n_comp)], 0, key="kx")
    py_ = ey.selectbox("Eje Y", [f"PC{i+1}" for i in range(n_comp)], 1, key="ky")
    ix, iy = int(px_[2:]) - 1, int(py_[2:]) - 1
    d["X"], d["Y"] = scores[:, ix], scores[:, iy]
    fig = px.scatter(d.sample(3000, random_state=1), x="X", y="Y", color="Grupo", opacity=0.6,
                     category_orders={"Grupo": orden}, labels={"X": px_, "Y": py_},
                     title=f"Grupos en el plano {px_}–{py_} (muestra de 3.000 pedidos)")
    st.plotly_chart(fig, use_container_width=True)
    pct = tabla["Varianza explicada (%)"].iloc[[ix, iy]].sum()
    st.caption(f"Este plano resume el {pct:.1f} % de la varianza. Los grupos se definieron con todos los "
               f"{n_comp} componentes, por eso pueden verse traslapados en 2D.")

    fig = px.box(d, x="Grupo", y="silueta", category_orders={"Grupo": orden},
                 title="Silueta por pedido y grupo", labels={"silueta": "Coeficiente de silueta"})
    fig.add_hline(y=0, line_dash="dot")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Perfil de cada grupo con las variables originales")
    perfil = d.groupby("Grupo")[list(cols)].mean().rename(columns=etq)
    perfil.insert(0, "Pedidos", d["Grupo"].value_counts())
    st.dataframe(perfil.round(2), use_container_width=True)

    z = (d.groupby("Grupo")[list(cols)].mean() - df[list(cols)].mean()) / df[list(cols)].std()
    fig = px.imshow(z.rename(columns=etq), text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-2, zmax=2, aspect="auto",
                    title="Diferencia del grupo frente al promedio general (en desviaciones estándar)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Composición por variables categóricas (no usadas en el modelo)")
    cat = st.selectbox("Variable categórica", ["order_channel", "metro_city", "cuisine_category",
                                                "weather_condition", "restaurant_price_tier",
                                                "customer_satisfaction_tier"])
    ct = pd.crosstab(d["Grupo"], d[cat], normalize="index") * 100
    fig = px.bar(ct, barmode="stack", labels={"value": "% del grupo", "Grupo": ""},
                 title=f"Composición de cada grupo según {cat}")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Las etiquetas “Grupo 1, 2…” son solo identificadores: no implican orden ni jerarquía.")

# ================================================================= 6. CONCLUSIONES
else:
    st.title("Discusión y conclusiones")
    st.caption("Los resultados escritos corresponden a la configuración por defecto "
               "(8 variables, sin margen neto; 4 componentes; K = 4). Si cambiaste opciones, revisa que "
               "las cifras sigan siendo coherentes.")
    n80 = int((tabla["Acumulado (%)"] < 80).sum() + 1)
    st.success(f"Con tu selección actual, {n80} componentes conservan "
               f"{tabla['Acumulado (%)'].iloc[n80-1]:.1f} % de la varianza; PC1 explica "
               f"{tabla['Varianza explicada (%)'].iloc[0]:.1f} %.")
    st.markdown("""
### ¿Qué variables definen los componentes?
- **PC1 (≈42 %) – “carga logística del domicilio”**: trayecto, tarifa y distancia (cargas ≈ 0,91–0,96) y,
  en menor medida, fricción logística (0,66); la calificación va en sentido contrario (−0,46).
  Es el eje que separa pedidos sin domicilio de pedidos lejanos y caros de enviar.
- **PC2 (≈14 %) – valor económico del pedido**, contrapuesto al descuento: pedidos grandes con poco descuento
  frente a pedidos pequeños muy descontados.
- **PC3 (≈13 %) – tiempo de cocina**: tiempo de preparación (0,71), independiente del trayecto.
- **PC4 (≈11 %) – pedidos grandes con descuento alto**: valor del pedido (+0,67) y descuento (+0,67) cargan en el mismo sentido.

### ¿Cuánta información se conservó?
Cuatro componentes retienen ≈ 80,8 %. No hay una reducción limpia a dos dimensiones (56 %): las dimensiones
económica, de cocina y logística son en buena parte independientes entre sí. Esto es un resultado en sí mismo:
la calificación casi no se explica por el valor del pedido ni por el descuento; se asocia con la logística.

### ¿Qué grupos aparecen? (K = 4)
| Grupo (según perfil) | Perfil | Observación |
|---|---|---|
| Sin trayecto, pedidos de mayor valor | Valor ≈ USD 36, descuento ≈ 3 %, trayecto ≈ 4 min, calificación 4,4 | ≈ 35 % de los pedidos; mezcla de consumo en local, recogida y cercanos |
| Sin trayecto, muy descontados | Valor ≈ USD 24, descuento ≈ 25 % | ≈ 29 %; la promoción es lo que los distingue |
| Domicilio de distancia media | ≈ 4,6 km, ≈ 17 min, tarifa ≈ USD 4,8 | ≈ 27 %; solo app de domicilio |
| Domicilio lejano | ≈ 12 km, ≈ 45 min, tarifa ≈ USD 8,5, calificación 3,5 | ≈ 9 %; el grupo mejor definido y el de peor calificación |

El grupo de domicilio lejano es el hallazgo más accionable: aunque el valor del pedido es parecido al de los
demás, la calificación es casi un punto menor, lo que sugiere que el costo y la espera, y no el gasto, es lo que
erosiona la satisfacción.

### ¿Son grupos claramente distinguibles?
**Solo parcialmente.** La silueta promedio ronda 0,23 (estructura débil), y entre 5 % y 6 % de los pedidos
tienen silueta negativa, especialmente en la frontera entre los dos grupos “sin trayecto” y los de distancia
media. La silueta es mayor con K = 2 (≈ 0,38: domicilio vs. sin domicilio), pero esa partición es casi
trivial y esconde diferencias de descuento y valor. K = 4 sacrifica algo de separación estadística a cambio
de interpretabilidad. Los datos se parecen más a un **continuo con una división estructural** que a
segmentos naturales nítidos.

### Limitaciones
1. Base aparentemente sintética: las relaciones pueden reflejar cómo se generó, no el comportamiento real.
2. Los ceros estructurales (sin domicilio) dominan PC1 y pueden jalar a K-Means; sería válido repetir el
   análisis solo con pedidos de la app.
3. K-Means supone grupos esféricos de tamaños similares y es sensible a atípicos y a la semilla.
4. No se controla por ciudad, poder adquisitivo ni moneda.

### Posible extensión
Repetir el análisis solo con pedidos de la app de domicilio, comparar por ciudad y probar DBSCAN o
mezclas gaussianas.
""")
