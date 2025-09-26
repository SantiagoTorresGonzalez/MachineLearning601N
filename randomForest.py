import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree

# --- Carga de datos ---
BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "dataset" / "dataArbol.xlsx"
datos = pd.read_excel(file_path)

# Limpieza de variables
datos["HistorialFamiliar"] = datos["HistorialFamiliar"].str.strip()
datos["Diagnostico"] = datos["Diagnostico"].str.strip()

# Variables independientes / dependiente
X = datos.drop("Diagnostico", axis=1)
y = datos["Diagnostico"]

# Codificación
le_fam = LabelEncoder()
X["HistorialFamiliar"] = le_fam.fit_transform(X["HistorialFamiliar"])
le_diag = LabelEncoder()
y = le_diag.fit_transform(y)

# División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Entrenamiento
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

# --- Evaluación ---
y_pred = bosque.predict(X_test)
acc = accuracy_score(y_test, y_pred)
accuracy_val = round(acc * 100, 2)  # Variable global exportable

# Guardar archivo de accuracy
with open(BASE_DIR / "static" / "accuracy_diabetes.txt", "w") as f:
    f.write(f"{accuracy_val:.2f}")

# Guardar reporte como HTML
report_dict = classification_report(y_test, y_pred, output_dict=True)
reporte_df = pd.DataFrame(report_dict).transpose()
with open(BASE_DIR / "templates" / "clasificacion_diabetes.html", "w", encoding="utf-8") as f:
    f.write(reporte_df.to_html(classes="table table-striped table-bordered text-white", border=0))

# Guardar matriz de confusión en static
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le_diag.classes_,
            yticklabels=le_diag.classes_)
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de Confusión - Diabetes")
plt.savefig(BASE_DIR / "static" / "confusion_matrix_diabetes.png", bbox_inches="tight")
plt.close()

# Guardar ROC en static
y_prob = bosque.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("1 - Especificidad (FPR)")
plt.ylabel("Sensibilidad (TPR)")
plt.title("Curva ROC - Random Forest")
plt.legend()
plt.savefig(BASE_DIR / "static" / "roc_diabetes.png", bbox_inches="tight")
plt.close()

# --- Guardar árbol binario del primer árbol ---
arbol = bosque.estimators_[0]  # Tomamos el primer árbol
plt.figure(figsize=(20, 10))
plot_tree(
    arbol,
    feature_names=X.columns,
    class_names=le_diag.classes_,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Árbol Binario del Random Forest")
plt.savefig(BASE_DIR / "static" / "arbol_binario_diabetes.png", bbox_inches="tight")
plt.close()

# --- Función predict_label() para predicciones individuales ---
def predict_label(features, threshold=0.5):
    prob = bosque.predict_proba(features)[0][1]  # prob de clase positiva
    label = "Sí" if prob >= threshold else "No"
    return label, prob
