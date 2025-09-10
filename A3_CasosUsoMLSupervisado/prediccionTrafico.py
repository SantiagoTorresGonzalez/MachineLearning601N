import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # backend para generar imágenes sin mostrar ventana
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Datos (asegúrate que las listas tengan la misma longitud)
datos = {
    'Distancia_km': [5, 10, 3, 8, 12, 28, 27, 4, 15],
    'Trafico_promedio': [200, 350, 100, 250, 400, 370, 500, 380, 300],
    'Tiempo_entrega': [15, 30, 8, 22, 40, 50, 30, 61, 21]
}

df = pd.DataFrame(datos)

# Variables (X: 2 columnas, y: 1D)
X = df[['Distancia_km', 'Trafico_promedio']].values
y = df['Tiempo_entrega'].values

# Entrenar modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Función de predicción
def CalcularTiempoProm(distancia, trafico):
    pred = modelo.predict([[distancia, trafico]])[0]
    return float(pred)

# Crear figura (2D)
def crear_fig(fix_trafico=None):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(df['Distancia_km'], df['Tiempo_entrega'], label='Datos reales', s=40)

    if fix_trafico is None:
        fix_trafico = df['Trafico_promedio'].mean()

    # Rango de distancias para dibujar la "línea" proyectada
    dist_range = np.linspace(df['Distancia_km'].min(), df['Distancia_km'].max(), 200)
    traf_array = np.full_like(dist_range, fill_value=fix_trafico)
    preds = modelo.predict(np.column_stack((dist_range, traf_array)))

    ax.plot(dist_range, preds, label=f"Predicción (traf={fix_trafico:.0f})", linewidth=2)
    ax.set_xlabel("Distancia (km)")
    ax.set_ylabel("Tiempo de entrega (min)")
    ax.set_title("Relación Distancia vs Tiempo de Entrega (proyección del modelo)")
    ax.legend()
    fig.tight_layout()
    return fig
