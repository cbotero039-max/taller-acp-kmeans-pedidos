# Taller individual · ACP y K-Means
**Machine Learning aplicado a Economía y Finanzas · Pontificia Universidad Javeriana Cali**

## Contexto y pregunta
¿Se pueden resumir en pocas dimensiones las características de un pedido de comida a domicilio (valor,
descuento, distancia, tiempos, tarifa, calificación) y existen *tipos* de pedido con perfiles distintos?
Cada observación es **un pedido**; la base tiene 8.000 pedidos de 12 ciudades y 3 canales
(app de domicilio, recogida, consumo en local).

- **Fuente:** `data/global_food_delivery_enriched_market_2026.csv` (26 columnas).
- **Limitaciones:** datos aparentemente sintéticos; ceros estructurales en distancia/tiempo/tarifa para
  pedidos sin domicilio; montos en USD sin ajuste por poder adquisitivo.
- **Por qué ACP:** las variables logísticas están muy correlacionadas y hay pares redundantes.
- **Por qué K-Means:** no hay etiqueta de “tipo de pedido”; se buscan grupos por similitud en el espacio estandarizado.

## Metodología
1. **Preparación:** sin faltantes ni duplicados. Se excluyen identificador, texto libre, `confidence_score`
   (casi constante), variables derivadas por fórmula y etiquetas categóricas (solo para interpretar).
   `estimated_restaurant_net_margin_usd` se excluye por defecto (r = 0,98 con el valor del pedido).
   Estandarización con `StandardScaler`.
2. **ACP:** varianza explicada, scree plot, cargas, criterio de Kaiser y ≥ 80 % acumulado (4 componentes ≈ 80,8 %).
3. **K-Means:** sobre los componentes; método del codo y silueta para K = 2…10; perfil de grupos en variables originales.

## Resultados principales (configuración por defecto)
- PC1 (≈42 %): carga logística del domicilio (trayecto, tarifa, distancia). PC2: valor vs. descuento. PC3: tiempo de cocina.
- Silueta máxima en K = 2 (≈ 0,38, domicilio vs. no domicilio); se elige K = 4 por interpretabilidad (silueta ≈ 0,23: estructura débil).
- Hallazgo: los pedidos de domicilio lejano (≈ 9 %) tienen la peor calificación (≈ 3,5) con un valor de pedido similar al resto.

## Ejecución
```bash
pip install -r requirements.txt
streamlit run app.py
```
Estructura: `app.py` (interfaz), `utils.py` (procesamiento, ACP, K-Means), `data/` (CSV).

## Despliegue
Streamlit Community Cloud → conectar este repositorio → archivo principal `app.py`.
