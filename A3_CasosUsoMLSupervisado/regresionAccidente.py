import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from pathlib import Path
import joblib

# --- Rutas base ---
BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "dataset" / "datos.xlsx"

# --- Carga de datos ---
data = pd.read_excel(file_path)

# Explorar el conjunto de datos (solo para debug/inspección)
print(data.head())
print(data.info())
print(data.describe())

# Variable dependiente
y = data['Accidente'].map({'Si': 1, 'No': 0})

# Variables independientes
x = data.drop('Accidente', axis=1)

# OneHotEncoding para variables categóricas
x = pd.get_dummies(x, columns=['Clima', 'EstadoVia'], drop_first=True)

# División entrenamiento / prueba
x_entrena, x_prueba, y_entrena, y_prueba = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# Escalador
scaler = StandardScaler()
X_entrena_scaled = scaler.fit_transform(x_entrena)
X_prueba_scaled = scaler.transform(x_prueba)

# Modelo
logistic_Model = LogisticRegression()
logistic_Model.fit(X_entrena_scaled, y_entrena)

# Guardamos columnas del modelo (importante para alinear datos nuevos)
columnas_modelo = x.columns

# --- Funciones auxiliares ---
def evaluate(model, X_test, y_test):
    """Calcula métricas y genera imagen de matriz de confusión"""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf = confusion_matrix(y_test, y_pred)

    # Visualizar matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Matriz de confusión')
    plt.show()

    return acc, report, conf


def predict_label(model, scaler, features, threshold=0.5):
    """Recibe un DataFrame de características y retorna 'Sí'/'No' y probabilidad"""
    X_scaled = scaler.transform(features)
    prob = model.predict_proba(X_scaled)[0][1]
    label = 'Sí' if prob >= threshold else 'No'
    return label, prob


# --- Evaluación ---
accuracy_val, report_val, conf_val = evaluate(logistic_Model, X_prueba_scaled, y_prueba)
print(f'Exactitud del modelo: {accuracy_val*100:.2f}%')

# --- Guardar modelo, scaler y columnas ---
joblib.dump(logistic_Model, BASE_DIR / "modelo.pkl")
joblib.dump(scaler, BASE_DIR / "scaler.pkl")
joblib.dump(columnas_modelo, BASE_DIR / "columnas.pkl")

print(" Modelo, scaler y columnas guardados exitosamente.")

# --- Exportar al importar este archivo ---
# Esto permite que app.py solo tenga que importar, sin reentrenar nada
logistic_Model = joblib.load(BASE_DIR / "modelo.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")
columnas_modelo = joblib.load(BASE_DIR / "columnas.pkl")
