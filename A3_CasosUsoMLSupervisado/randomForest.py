#Importación de librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

#Importación del dataframe (Excel en este caso)
datos = pd.read_excel('datos/dataArbol.xlsx')

#Limpiar strings
datos["HistorialFamiliar"] = datos["HistorialFamiliar"].str.strip()
datos["Diagnostico"] = datos["Diagnostico"].str.strip()

#Separación de variables
x= datos.drop("Diagnostico", axis=1)
y= datos["Diagnostico"]

#Variables categóricas (Codificación)
le_fam = LabelEncoder()
x["HistorialFamiliar"] = le_fam.fit_transform(x["HistorialFamiliar"])

le_diag = LabelEncoder()
y = le_diag.fit_transform(y)



#Creación del bosque aleatorio con la función RandomForestClassifier y entrenamiento
bosque= RandomForestClassifier(n_estimators=100,
                               criterion="gini",
                               max_features="sqrt",
                               bootstrap=True,
                               max_samples=2/3,
                               oob_score=True)
bosque.fit(x, y)

#Predicción de ejemplo
entry= np.array([[50, 24.7, 108.53, 101, le_fam.transform(["No"])[0]]])
pred= bosque.predict(entry)
print("Predicción:", le_diag.inverse_transform(pred))

#Predicción y OOB
print("Precisión: ", bosque.score(x,y))
print("OOB score: ", bosque.oob_score_)

#Para matriz confusa
y_pred= bosque.predict(x)
cm= confusion_matrix(y, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le_diag.classes_,
            yticklabels=le_diag.classes_)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión - Random Forest")
plt.show()

#Mostrar árbol de ejemplo
plt.figure(figsize=(15,10))
tree.plot_tree(bosque.estimators_[0], feature_names=x.columns, filled=True)
plt.title("Ejemplo de Árbol del Random Forest")
plt.show()
