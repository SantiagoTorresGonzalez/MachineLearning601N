# randomForest.py
# Entrenamiento del modelo Random Forest para Diabetes Tipo II
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

# --- Rutas base ---
BASE_DIR = Path(__file__).resolve().parent

# --- Carga de datos ---
file_path = BASE_DIR / "dataset" / "dataArbol.xlsx"
datos = pd.read_excel(file_path)

# Limpieza de variables categóricas
datos["HistorialFamiliar"] = datos["HistorialFamiliar"].str.strip()
datos["Diagnostico"] = datos["Diagnostico"].str.strip()

# Separación X / y
X = datos.drop("Diagnostico", axis=1)
y = datos["Diagnostico"]

# Codificación de variables categóricas
le_fam = LabelEncoder()
X["HistorialFamiliar"] = le_fam.fit_transform(X["HistorialFamiliar"])

le_diag = LabelEncoder()
y = le_diag.fit_transform(y)

# División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Entrenamiento del RandomForest
bosque = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_features="sqrt",
    bootstrap=True,
    max_samples=2/3,
    oob_score=True,
    random_state=42
)
bosque.fit(X_train, y_train)

# --- Métricas ---
y_pred = bosque.predict(X_test)
acc = accuracy_score(y_test, y_pred)
accuracy_val = round(acc * 100, 2)

report_dict = classification_report(y_test, y_pred, output_dict=True)
reporte_df = pd.DataFrame(report_dict).transpose()

# Guardar exactitud en static
with open(BASE_DIR / "static" / "accuracy_diabetes.txt", "w") as f:
    f.write(f"{accuracy_val:.2f}")

# Guardar reporte como HTML en templates
with open(BASE_DIR / "templates" / "clasificacion_diabetes.html", "w", encoding="utf-8") as f:
    f.write(
        reporte_df.to_html(
            classes="table table-striped table-bordered text-white", border=0
        )
    )

# Guardar matriz de confusión en static
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le_diag.classes_,
            yticklabels=le_diag.classes_)
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de Confusión - Random Forest")
plt.savefig(BASE_DIR / "static" / "confusion_matrix_diabetes.png", bbox_inches="tight")
plt.close()


# --- Función para dibujar un árbol representativo ---
def plot_tree_example():
    estimator = bosque.estimators_[0]
    fig, ax = plt.subplots(figsize=(15, 10))
    tree.plot_tree(
        estimator,
        feature_names=X.columns,
        class_names=le_diag.classes_,
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax
    )
    ax.set_title("Ejemplo de Árbol del Random Forest")
    return fig