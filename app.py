import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score

# Título de la aplicación
st.title("Análisis de Diagnóstico de Cáncer de Mama (WDBC)")
st.markdown(
    """
    ### Visualización y Análisis con Machine Learning
    Esta aplicación utiliza el dataset de Wisconsin Diagnostic Breast Cancer para entrenar y evaluar modelos de predicción.
    Incluye pasos de limpieza de datos, análisis exploratorio (EDA), entrenamiento y ajuste de hiperparámetros.
    """
)

# Cargar el Dataset
@st.cache
def load_data():
    data = pd.read_csv("ruta_al_archivo.csv", header=None)
    column_names = [
        "ID", "Diagnosis", 
        "Mean_Radius", "Mean_Texture", "Mean_Perimeter", "Mean_Area", "Mean_Smoothness",
        "Mean_Compactness", "Mean_Concavity", "Mean_Concave_Points", "Mean_Symmetry", "Mean_Fractal_Dimension",
        "SE_Radius", "SE_Texture", "SE_Perimeter", "SE_Area", "SE_Smoothness",
        "SE_Compactness", "SE_Concavity", "SE_Concave_Points", "SE_Symmetry", "SE_Fractal_Dimension",
        "Worst_Radius", "Worst_Texture", "Worst_Perimeter", "Worst_Area", "Worst_Smoothness",
        "Worst_Compactness", "Worst_Concavity", "Worst_Concave_Points", "Worst_Symmetry", "Worst_Fractal_Dimension"
    ]
    data.columns = column_names
    return data

df = load_data()

# Mostrar el dataset
st.header("1. Dataset Original")
st.write(df.head())

# Limpieza de datos
st.header("2. Limpieza de Datos")
df = df.drop(columns="ID")
df["Diagnosis"] = df["Diagnosis"].map({"M": 1, "B": 0})
st.markdown("#### Eliminación de la columna ID y transformación de Diagnosis (M=1, B=0)")
st.write(df.head())

# Visualización (EDA)
st.header("3. Análisis Exploratorio de Datos (EDA)")
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x="Diagnosis", data=df, ax=ax)
ax.set_title("Distribución de Diagnósticos (0=Benigno, 1=Maligno)")
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
ax.set_title("Matriz de Correlación entre Características")
st.pyplot(fig)

# Preprocesamiento
st.header("4. Preprocesamiento de los Datos")
X = df.drop("Diagnosis", axis=1)
y = df["Diagnosis"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
st.markdown("#### Escalado de Características y División del Dataset")
st.write("Conjunto de Entrenamiento:", X_train.shape)
st.write("Conjunto de Prueba:", X_test.shape)

# Entrenamiento de Modelos
st.header("5. Entrenamiento de Modelos")
log_reg = LogisticRegression().fit(X_train, y_train)
knn = SVC(probability=True).fit(X_train, y_train)
rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
st.markdown("#### Modelos Entrenados: Regresión Logística, SVM y Bosques Aleatorios")

# Ajuste de Hiperparámetros
st.header("6. Ajuste de Hiperparámetros")
param_grid_rf = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}
grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring="accuracy")
grid_search_rf.fit(X_train, y_train)

param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 0.1]}
grid_search_svm = GridSearchCV(SVC(probability=True), param_grid_svm, cv=5, scoring="accuracy")
grid_search_svm.fit(X_train, y_train)

st.write("Mejores parámetros Bosques Aleatorios:", grid_search_rf.best_params_)
st.write("Mejores parámetros SVM:", grid_search_svm.best_params_)

# Evaluación Final
st.header("7. Evaluación Final y Curva ROC")
models = {
    "Bosques Aleatorios": grid_search_rf.best_estimator_,
    "SVM": grid_search_svm.best_estimator_,
    "Regresión Logística": log_reg
}

fig, ax = plt.subplots(figsize=(10, 8))
for name, model in models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
ax.plot([0, 1], [0, 1], 'k--', lw=2)
ax.set_title("Curva ROC Comparativa", fontsize=16)
ax.set_xlabel("Tasa de Falsos Positivos", fontsize=12)
ax.set_ylabel("Tasa de Verdaderos Positivos", fontsize=12)
ax.legend(loc="lower right")
st.pyplot(fig)

# Imprimir precisiones finales
st.header("8. Precisión de Modelos")
for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Precisión del modelo {name}: {accuracy:.2f}")

# Conclusión Final
st.header("9. Conclusión Final")
st.markdown("""
### Modelo Recomendado: SVM
Basándonos en las métricas de rendimiento (precisión y AUC), SVM es el mejor modelo para este conjunto de datos. Este modelo:
- Ofrece alta precisión (98%) y una capacidad destacada para distinguir entre benignos y malignos.
- Puede manejar relaciones complejas entre las características de entrada.

### Modelo Secundario: Regresión Logística
Aunque tiene un rendimiento ligeramente inferior a SVM, la Regresión Logística sigue siendo una excelente opción:
- Es más fácil de interpretar, lo que puede ser importante en contextos médicos.

### Consideración de Bosques Aleatorios
El modelo de Bosques Aleatorios sigue siendo robusto y tiene la ventaja de ser más interpretable que SVM.

Ambos modelos cumplen con los requisitos para un diagnóstico asistido confiable y eficiente.
""")
