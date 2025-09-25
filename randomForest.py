# Importación de librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# --- Carga de datos ---
BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "dataset" / "dataArbol.xlsx"
datos = pd.read_excel(file_path)

# Limpiar strings
datos["HistorialFamiliar"] = datos["HistorialFamiliar"].str.strip()
datos["Diagnostico"] = datos["Diagnostico"].str.strip()

# Separación de variables
x = datos.drop("Diagnostico", axis=1)
y = datos["Diagnostico"]

# Variables categóricas (Codificación)
le_fam = LabelEncoder()
x["HistorialFamiliar"] = le_fam.fit_transform(x["HistorialFamiliar"])

le_diag = LabelEncoder()
y = le_diag.fit_transform(y)

# Creación del bosque aleatorio con la función RandomForestClassifier y entrenamiento
bosque = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_features="sqrt",
    bootstrap=True,
    max_samples=2/3,
    oob_score=True
)
bosque.fit(x, y)

# --- Funciones para gráficas ---
def plot_confusion_matrix():
    """Genera la figura de la matriz de confusión."""
    y_pred = bosque.predict(x)
    cm = confusion_matrix(y, y_pred, labels=np.unique(y))
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=le_diag.classes_,
        yticklabels=le_diag.classes_,
        ax=ax
    )
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusión - Random Forest")
    return fig


def plot_tree_example():
    """Genera la figura de un árbol del Random Forest."""
    estimator = bosque.estimators_[0]  # Tomamos un árbol
    fig, ax = plt.subplots(figsize=(15, 10))
    tree.plot_tree(
        estimator,
        feature_names=x.columns,
        class_names=le_diag.classes_,
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax
    )
    ax.set_title("Ejemplo de Árbol del Random Forest")
    return fig


# --- Prueba rápida ---
if __name__ == "__main__":
    # Predicción de ejemplo
    entry = np.array([[50, 24.7, 108.53, 101, le_fam.transform(["No"])[0]]])
    pred = bosque.predict(entry)
    print("Predicción:", le_diag.inverse_transform(pred))
    print("Precisión: ", bosque.score(x, y))
    print("OOB score: ", bosque.oob_score_)
