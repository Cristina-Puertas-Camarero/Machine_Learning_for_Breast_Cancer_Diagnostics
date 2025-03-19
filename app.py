import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Título de la App
st.title("Diagnóstico de Tumores: Análisis Predictivo")

# Subtítulo
st.write("""
Esta aplicación utiliza Machine Learning para clasificar tumores como benignos o malignos. 
Incluye gráficos de ROC, AUC, y analiza la importancia de las características.
""")

# Cargar el dataset
df = pd.read_csv("4_dataset_sin_correlaciones_altas.csv")
X = df.drop(columns="Diagnosis")
y = df["Diagnosis"]

# Dividir en entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de Bosques Aleatorios
rf = RandomForestClassifier(max_depth=None, n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Generar Curva ROC y AUC
st.subheader("Curva ROC y AUC")
y_proba = rf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"Bosques Aleatorios (AUC = {roc_auc:.2f})", color="blue")
ax.plot([0, 1], [0, 1], 'k--', lw=2, label="Referencia")
ax.set_title("Curva ROC")
ax.set_xlabel("Tasa de Falsos Positivos")
ax.set_ylabel("Tasa de Verdaderos Positivos")
ax.legend(loc="lower right")
st.pyplot(fig)

# Importancia de las características
st.subheader("Importancia de las Características (Bosques Aleatorios)")
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=feature_importances.head(10), x="Importance", y="Feature", palette="coolwarm", ax=ax)
ax.set_title("Top 10 Características Más Importantes")
ax.set_xlabel("Importancia")
ax.set_ylabel("Características")
st.pyplot(fig)

# Mostrar tabla completa de importancias
st.write("Tabla Completa de Importancia de Características:")
st.dataframe(feature_importances)

# Predicción interactiva
st.subheader("Hacer una Predicción")
input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(f"Ingrese el valor para {col}", value=0.0)

# Convertir los datos de entrada a DataFrame
input_df = pd.DataFrame([input_data])

# Predicción
if st.button("Predecir"):
    prediction = rf.predict(input_df)[0]
    prediction_proba = rf.predict_proba(input_df)[0][1]
    resultado = "Maligno" if prediction == 1 else "Benigno"
    st.write(f"**Resultado de la Predicción:** {resultado}")
    st.write(f"**Probabilidad de Maligno:** {prediction_proba:.2f}")
