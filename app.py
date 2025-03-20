# Importar las librerías necesarias
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 📝 Contexto del Proyecto
st.header("📝 Contexto del Proyecto")
st.markdown("""
Este proyecto utiliza datos del **[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)**, 
un recurso público ampliamente utilizado en investigaciones de ciencia de datos.  
El dataset fue recopilado originalmente por el Dr. William H. Wolberg, en el Hospital de la Universidad de Wisconsin, y contiene mediciones relevantes obtenidas de imágenes digitalizadas de tumores mamarios. Estas mediciones incluyen características como el tamaño, la forma y la textura de los tumores.

### Objetivo:
- Desarrollar modelos predictivos usando técnicas de **Machine Learning** que clasifiquen tumores en **benignos** o **malignos**.
- Identificar características clave que contribuyan significativamente al diagnóstico.

**Nota Importante:** Este proyecto se realiza con fines **educativos** y forma parte de mi portafolio como estudiante de ciencia de datos. No está diseñado para reemplazar el juicio médico profesional.
""")

# 🎗️ Introducción del Proyecto
st.title("🎗️ Diagnóstico de Cáncer de Mama con Machine Learning 🎗️")
st.markdown("""
Este proyecto demuestra cómo las técnicas de **Machine Learning** pueden asistir en la detección y diagnóstico temprano de tumores mamarios, clasificándolos en **benignos** o **malignos**.  
**Nota Importante:** Este trabajo ha sido realizado con fines académicos y forma parte de mi portafolio.  
**Autora:** Cristina Puertas Camarero  
""")

# Logo (centrado y tamaño ajustado)
st.markdown(
    """
    <div style="text-align: center;">
        <img src="logo_cancer_mama.png" alt="Logo Cáncer de Mama" style="width: 120px;"/>
    </div>
    """,
    unsafe_allow_html=True
)

# 📂 Carga del Dataset
st.header("📂 Carga del Dataset")
try:
    # Ruta al dataset
    dataset_path = r"C:\Users\Propietario\Documents\IronHack 01\Machine_Learning_for_Breast_Cancer_Diagnostics\Machine_Learning_for_Breast_Cancer_Diagnostics\datos\4_dataset_sin_correlaciones_altas.csv"

    # Cargar dataset
    df = pd.read_csv(dataset_path)
    st.success("Dataset cargado exitosamente. Aquí están las primeras filas:")
    st.dataframe(df.head())  # Mostrar las primeras filas del dataset

except Exception as e:
    st.error("Hubo un error al cargar el dataset. Por favor verifica la ruta y el archivo.")
    st.text(f"Detalles del error: {e}")

# 📊 Análisis Exploratorio de los Datos
st.header("📊 Análisis Exploratorio de los Datos")

# Distribución de tumores
st.subheader("🎗️ Distribución de Tumores")
st.markdown("""
Analizar cuántos tumores son **benignos** y cuántos **malignos** nos permite entender si el dataset está equilibrado, 
lo cual es crucial para el entrenamiento de los modelos.
""")
fig1, ax1 = plt.subplots()
sns.countplot(data=df, x="Diagnosis", palette="Purples", ax=ax1)
ax1.set_title("Distribución de Tumores", fontsize=16)
ax1.set_xlabel("Tipo de Tumor", fontsize=12)
ax1.set_ylabel("Cantidad", fontsize=12)
st.pyplot(fig1)

# Relación entre características clave
st.subheader("📈 Relación entre Características Clave")
st.markdown("""
Exploramos la relación entre características importantes como `Mean_Radius` y `Mean_Concavity`, 
las cuales son esenciales para distinguir entre tumores benignos y malignos.
""")
fig2, ax2 = plt.subplots()
sns.scatterplot(data=df, x="Mean_Radius", y="Mean_Concavity", hue="Diagnosis", palette="coolwarm", ax=ax2)
ax2.set_title("Relación entre `Mean_Radius` y `Mean_Concavity`", fontsize=16)
ax2.set_xlabel("Mean Radius", fontsize=12)
ax2.set_ylabel("Mean Concavity", fontsize=12)
st.pyplot(fig2)

# Histograma de una característica clave
st.subheader("📊 Distribución de `Mean_Radius`")
st.markdown("""
El histograma muestra cómo el `Mean_Radius` (tamaño promedio del tumor) varía entre tumores benignos y malignos.
""")
fig3, ax3 = plt.subplots()
sns.histplot(data=df, x="Mean_Radius", hue="Diagnosis", multiple="stack", palette="Purples", kde=True, ax=ax3)
ax3.set_title("Distribución de `Mean_Radius` por Tipo de Tumor", fontsize=16)
ax3.set_xlabel("Mean Radius", fontsize=12)
ax3.set_ylabel("Frecuencia", fontsize=12)
st.pyplot(fig3)

# 🔍 Exploración Interactiva
st.subheader("🔍 Exploración Interactiva de las Características")
selected_feature_hist = st.selectbox(
    "Selecciona una característica para ver su distribución (Histograma):", 
    df.columns.drop("Diagnosis")
)
selected_feature_x = st.selectbox(
    "Selecciona la característica X (Gráfico de Dispersión):", 
    df.columns.drop("Diagnosis")
)
selected_feature_y = st.selectbox(
    "Selecciona la característica Y (Gráfico de Dispersión):", 
    df.columns.drop("Diagnosis")
)

# Histograma interactivo
st.subheader(f"📊 Distribución de `{selected_feature_hist}`")
fig_hist, ax_hist = plt.subplots()
sns.histplot(data=df, x=selected_feature_hist, hue="Diagnosis", multiple="stack", palette="Purples", kde=True, ax=ax_hist)
ax_hist.set_title(f"Distribución de `{selected_feature_hist}` por Tipo de Tumor", fontsize=16)
ax_hist.set_xlabel(selected_feature_hist, fontsize=12)
ax_hist.set_ylabel("Frecuencia", fontsize=12)
st.pyplot(fig_hist)

# Gráfico de dispersión interactivo
st.subheader(f"📈 Relación entre `{selected_feature_x}` y `{selected_feature_y}`")
fig_scatter, ax_scatter = plt.subplots()
sns.scatterplot(data=df, x=selected_feature_x, y=selected_feature_y, hue="Diagnosis", palette="coolwarm", ax=ax_scatter)
ax_scatter.set_title(f"Relación entre `{selected_feature_x}` y `{selected_feature_y}`", fontsize=16)
ax_scatter.set_xlabel(selected_feature_x, fontsize=12)
ax_scatter.set_ylabel(selected_feature_y, fontsize=12)
st.pyplot(fig_scatter)

# 📊 Métricas del Modelo
st.header("📊 Resultados del Modelo")
metrics_data = {
    "Modelo": ["Random Forest", "Logistic Regression"],
    "Precisión (%)": [95.38, 92.52],
    "AUC": [0.99, 0.99],
    "Recall Maligno (%)": [95.0, 92.0]
}
metrics_df = pd.DataFrame(metrics_data)
st.dataframe(metrics_df)
st.markdown("""
- **Accuracy (Precisión):** Indica qué porcentaje de las predicciones fueron correctas.
- **AUC:** Mide la capacidad del modelo para distinguir entre clases.
- **Recall Maligno:** Prioriza identificar correctamente tumores malignos, reduciendo falsos negativos.
""")

# 🌟 Importancia de las Características
st.subheader("🌟 Importancia de las Características")
feature_importances = {
    "Características": [
        "Mean_Radius", "Worst_Concavity", "Mean_Concavity", 
        "SE_Radius", "Worst_Compactness", "Mean_Texture", 
        "Mean_Compactness", "Worst_Symmetry", "SE_Concavity", 
        "Worst_Smoothness"
    ],
    "Importancia": [0.202525, 0.170028, 0.134734, 0.087134, 0.069220, 0.049379, 0.049048, 0.031817, 0.028763, 0.024632]
}
importances_df = pd.DataFrame(feature_importances)
fig_features, ax_features = plt.subplots(figsize=(10, 6))
sns.barplot(data=importances_df, x="Importancia", y="Características", palette="Purples", ax=ax_features)
ax_features.set_title("Top 10 Características Más Importantes", fontsize=16)
ax_features.set_xlabel("Importancia Relativa", fontsize=12)
ax_features.set_ylabel("Características", fontsize=12)
st.pyplot(fig_features)

# 🛠️ Pasos Realizados en el Proyecto
st.header("🛠️ Pasos Realizados en el Proyecto")
st.markdown("""
A continuación, se describen los pasos que seguimos para desarrollar este proyecto y llegar a las conclusiones finales:

### 1️⃣ Exploración y Visualización de los Datos:
- Cargamos el dataset y examinamos su estructura para comprender las variables disponibles.
- Visualizamos la distribución de tumores benignos y malignos, así como relaciones clave entre variables, utilizando gráficos como histogramas y gráficos de dispersión.
- Identificamos correlaciones entre características, destacando variables que podían ser redundantes o no relevantes.

### 2️⃣ Limpieza y Preprocesamiento:
- Eliminamos columnas irrelevantes (como identificadores) y aquellas con alta correlación para evitar multicolinealidad.
- Transformamos variables categóricas en formato numérico (por ejemplo, la columna de diagnóstico).
- Normalizamos las características para facilitar el entrenamiento de los modelos.

### 3️⃣ Análisis Exploratorio de Datos (EDA):
- Detectamos qué características presentaban las mayores diferencias entre tumores benignos y malignos.
- Algunas características clave identificadas fueron:
  - **`Mean_Radius` (Radio Promedio):** Tamaño promedio del tumor.
  - **`Worst_Concavity` (Mayor Concavidad):** Indicador de irregularidades en los contornos de los tumores.

### 4️⃣ Modelado Predictivo:
- Entrenamos y evaluamos diferentes modelos de **Machine Learning**, incluyendo:
  - **Random Forest:** Un modelo basado en múltiples árboles de decisión, ideal para identificar patrones complejos.
  - **Logistic Regression:** Un modelo más sencillo pero interpretable, útil en contextos médicos.
  - **Support Vector Machines (SVM):** Captura relaciones no lineales entre las características.
- Usamos **cross-validation** para evaluar el rendimiento de los modelos en datos desconocidos.

### 5️⃣ Optimización de Hiperparámetros:
- Ajustamos hiperparámetros clave de los modelos usando **GridSearchCV**, probando múltiples combinaciones para encontrar la configuración óptima.
- Por ejemplo, en **Random Forest**:
  - **Número de árboles:** 100
  - **Máxima profundidad:** Sin restricciones (para permitir una mayor complejidad).
  - **Criterio de división:** Entropía.

### 6️⃣ Comparación de Modelos:
- Comparamos los modelos utilizando métricas clave como:
  - **Accuracy (Precisión):** Qué porcentaje de predicciones fueron correctas.
  - **AUC (Área Bajo la Curva ROC):** Qué tan bien separan los modelos entre tumores benignos y malignos.
  - **Recall:** Qué tan bien identifican los modelos los tumores malignos.
- **Resultados Destacados:**
  - El **Random Forest** fue el modelo más robusto, con un **Accuracy** del **95.38%** y un **AUC** de **0.99**.
  - Aunque la **Logistic Regression** también alcanzó un AUC de **0.99**, tuvo un rendimiento ligeramente inferior en términos de precisión general.

### 7️⃣ Conclusiones:
- **Random Forest** se destacó como el modelo más efectivo y robusto.
- Las características más importantes para el modelo incluyen `Mean_Radius` y `Worst_Concavity`.
- Este análisis demuestra cómo las herramientas de Machine Learning pueden complementar el juicio clínico en el diagnóstico temprano de cáncer de mama.
""")


# ✨ Conclusiones
st.header("✨ Conclusiones")
st.markdown("""
Este proyecto destaca cómo las técnicas de **Machine Learning**, como **Random Forest**, pueden ayudar en el diagnóstico temprano de tumores.  
**Resumen del Proceso:**
- Analizamos los datos para identificar características clave.
- Entrenamos modelos como **Random Forest** y **Logistic Regression** para predecir tumores.
- Evaluamos los modelos con métricas como **Accuracy**, **AUC** y **Recall Maligno**.

**Resultados Clave:**
- **Random Forest** alcanzó una precisión del **95.38%** y un AUC de **0.99**.
- Las características más importantes incluyen `Mean_Radius` y `Worst_Concavity`.

💖 **Gracias por explorar este proyecto educativo, creado con fines académicos y para mi portafolio. El cáncer de mama es un tema de gran sensibilidad. Sigamos innovando para marcar una diferencia positiva.**
""")

