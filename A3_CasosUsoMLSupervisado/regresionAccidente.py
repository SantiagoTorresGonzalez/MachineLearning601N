import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Carga de datos
data = pd.read_excel('C:/Users/sasv2/Downloads/datos.xlsx')

# Explorar el conjunto de datos
print(data.head())
print(data.info())
print(data.describe())

# Transformar la variable dependiente a 0 y 1
y = data['Accidente'].map({'Sí': 1, 'No': 0})

# Separación de variables independientes
x = data.drop('Accidente', axis=1)

# Convertir variables categóricas a dummies (OneHotEncoding)
x = pd.get_dummies(x, columns=['Clima', 'EstadoVia'], drop_first=True)

# División del dataset en conjunto de entrenamiento y prueba (80% Entrenamiento, 20% prueba) con estratificación
x_entrena, x_prueba, y_entrena, y_prueba = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# Estandarización de datos (solo columnas numéricas)
scaler = StandardScaler()
X_entrena_scaled = scaler.fit_transform(x_entrena)
X_prueba_scaled = scaler.transform(x_prueba)


# Crear modelo y entrenarlo

logistic_Model = LogisticRegression()
logistic_Model.fit(X_entrena_scaled, y_entrena)


# Función evaluate()
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


# Función predict_label()

def predict_label(model, scaler, features, threshold=0.5):
    """Recibe un DataFrame de características y retorna 'Sí'/'No' y probabilidad"""
    X_scaled = scaler.transform(features)
    prob = model.predict_proba(X_scaled)[0][1]
    label = 'Sí' if prob >= threshold else 'No'
    return label, prob

# Evaluar modelo
accuracy_val, report_val, conf_val = evaluate(logistic_Model, X_prueba_scaled, y_prueba)
print(f'Exactitud del modelo: {accuracy_val*100:.2f}%')


# Predicción de ejemplo

# Crear un DataFrame con un registro de ejemplo
example = pd.DataFrame({
    'Velocidad': [80],
    'EdadConductor': [30],
    'Clima_Nublado': [0],
    'Clima_Lluvioso': [1],
    'EstadoVia_Mojada': [0],
    'EstadoVia_Resbaladiza': [1]
})

label, prob = predict_label(logistic_Model, scaler, example)
print(f'Predicción: {label}, Probabilidad: {prob:.2f}')