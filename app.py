# Importar las librerías necesarias
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# Configurar el backend de Matplotlib recomendado para Streamlit
matplotlib.use('Agg')

# Configuración inicial de la aplicación
st.set_page_config(page_title="Diagnóstico de Cáncer de Mama", layout="wide")

# Crear una barra lateral para la navegación entre pantallas
page = st.sidebar.selectbox(
    "Navegación",
    ["Portada", "Resultados y Análisis", "Predicciones Interactivas"]
)

# 🌟 Pantalla 1: Portada
if page == "Portada":
    st.title("🎗️ Bienvenidos al Proyecto de Diagnóstico de Cáncer de Mama")
    st.markdown("""
    Este proyecto explora cómo las técnicas de **Machine Learning** pueden asistir en el diagnóstico temprano de tumores mamarios, clasificándolos en **benignos** o **malignos**.
    El cáncer de mama es uno de los desafíos más importantes en el ámbito de la salud pública, siendo una de las principales causas de mortalidad en mujeres. Según la Organización Mundial de la Salud (OMS), el diagnóstico temprano es clave para aumentar las probabilidades de supervivencia.
    """)
    st.image("cancer.jpg", caption="Lucha contra el cáncer de mama", use_container_width=True)
    st.markdown("""
    ### Contexto del Proyecto:
    - **Datos:** Extraídos del [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29).
    - **Propósito:** Crear un modelo predictivo y analizar características clave relacionadas con el diagnóstico.

    Este trabajo es exclusivamente **académico** y forma parte de mi portafolio como estudiante de ciencia de datos.  
    **Autora:** Cristina Puertas Camarero
    """)

# 📊 Pantalla 2: Resultados y Análisis
elif page == "Resultados y Análisis":
    st.title("📊 Resultados y Análisis del Proyecto")
    st.markdown("""
    En esta sección exploraremos las visualizaciones, el análisis de características y las métricas de los modelos utilizados en este estudio. También explicaremos cómo hemos llegado a nuestras conclusiones.  
    """)

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
        st.stop()

    # 🛠️ Pasos del Proyecto
    st.header("🛠️ Pasos del Proyecto")
    st.markdown("""
    ### 1️⃣ Exploración y Visualización de los Datos:
    - Analizamos la estructura del dataset y visualizamos distribuciones clave.
    - Identificamos correlaciones y posibles redundancias entre características.

    ### 2️⃣ Limpieza y Preprocesamiento:
    - Eliminamos columnas irrelevantes y aquellas con alta correlación.
    - Transformamos variables categóricas en formato numérico.
    - Normalizamos características para un mejor rendimiento del modelo.

    ### 3️⃣ Modelado Predictivo:
    - Probamos modelos como **Random Forest**, **Logistic Regression** y **Support Vector Machines (SVM)**.
    - Usamos **GridSearchCV** para ajustar hiperparámetros clave.

    ### 4️⃣ Evaluación y Comparación de Modelos:
    - Comparamos métricas como **Accuracy**, **AUC** y **Recall Maligno**.
    - **Resultados Destacados:**
      - El **Random Forest** fue el modelo más efectivo, con un **Accuracy** del **95.38%** y un **AUC** de **0.99**.
    """)

    # 📊 Métricas del Modelo
    st.subheader("📊 Resultados del Modelo")
    metrics_data = {
        "Modelo": ["Random Forest", "Logistic Regression"],
        "Precisión (%)": [95.38, 92.52],
        "AUC": [0.99, 0.99],
        "Recall Maligno (%)": [95.0, 92.0]
    }
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df)
    st.markdown("""
    - **Accuracy (Precisión):** Porcentaje de predicciones correctas.
    - **AUC:** Capacidad del modelo para distinguir entre tumores benignos y malignos.
    - **Recall Maligno:** Proporción de tumores malignos correctamente identificados.
    """)

    # Distribución de tumores
    st.subheader("🎗️ Distribución de Tumores")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x="Diagnosis", palette="Purples", ax=ax1)
    ax1.set_title("Distribución de Tumores", fontsize=16)
    ax1.set_xlabel("Tipo de Tumor", fontsize=12)
    ax1.set_ylabel("Cantidad", fontsize=12)
    st.pyplot(fig1)

    # Gráfico de correlaciones (Mapa de Calor)
    st.subheader("🌡️ Mapa de Calor de Correlaciones")
    st.markdown("""
    Visualizamos las correlaciones entre características del dataset para identificar posibles relaciones y redundancias.
    """)
    fig_corr, ax_corr = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=False, cmap="Purples", ax=ax_corr)
    ax_corr.set_title("Mapa de Calor de Correlaciones", fontsize=16)
    st.pyplot(fig_corr)

    # Interactividad: Selección de características
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
    st.pyplot(fig_hist)

    # Gráfico de dispersión interactivo
    st.subheader(f"📈 Relación entre `{selected_feature_x}` y `{selected_feature_y}`")
    fig_scatter, ax_scatter = plt.subplots()
    sns.scatterplot(data=df, x=selected_feature_x, y=selected_feature_y, hue="Diagnosis", palette="coolwarm", ax=ax_scatter)
    ax_scatter.set_title(f"Relación entre `{selected_feature_x}` y `{selected_feature_y}`", fontsize=16)
    st.pyplot(fig_scatter)

    # Importancia de las características
    st.subheader("🌟 Importancia de las Características")
    st.markdown("""
    El gráfico muestra las 10 características más importantes identificadas por el modelo de Bosques Aleatorios. Estas variables influyen significativamente en la predicción del diagnóstico.
    """)

    # Datos de importancia de características (ajústalo según tu análisis real)
    feature_importances = {
    "Características": [
        "Mean_Radius", "Worst_Concavity", "Mean_Concavity", 
        "SE_Radius", "Worst_Compactness", "Mean_Texture", 
        "Mean_Compactness", "Worst_Symmetry", "SE_Concavity", 
        "Worst_Smoothness"
    ],
    "Importancia": [0.202525, 0.170028, 0.134734, 0.087134, 0.069220, 0.049379, 0.049048, 0.031817, 0.028763, 0.024632]
    }

    # Convertir a DataFrame
    importances_df = pd.DataFrame(feature_importances)

    # Crear gráfico de barras
    fig_features, ax_features = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importances_df, x="Importancia", y="Características", palette="Purples", ax=ax_features)
    ax_features.set_title("Top 10 Características Más Importantes", fontsize=16)
    ax_features.set_xlabel("Importancia Relativa", fontsize=12)
    ax_features.set_ylabel("Características", fontsize=12)
    st.pyplot(fig_features)

    st.markdown("""
    - **Interpretación:**  
     Las características como `Mean_Radius` y `Worst_Concavity` reflejan propiedades importantes, como tamaño y forma irregular de los tumores.
    """)

    # 🔍 Conclusiones del Proyecto
    st.header("🔍 Conclusiones del Proyecto")
    st.markdown("""
    El objetivo principal de este proyecto fue desarrollar un modelo predictivo eficaz para clasificar tumores de mama como **benignos** o **malignos**, utilizando datos clínicos. A continuación, detallamos nuestras conclusiones basadas en el proceso completo:

    ### 1️⃣ Elección de Random Forest como Mejor Modelo
    Tras probar varios algoritmos de Machine Learning, como:
    - **Random Forest:** Modelo basado en múltiples árboles de decisión, ideal para identificar patrones complejos.
    - **Logistic Regression:** Un modelo interpretable pero con limitaciones al capturar relaciones no lineales.
    - **Support Vector Machines (SVM):** Potente para relaciones no lineales pero más costoso computacionalmente.

    El **Random Forest** destacó como la mejor opción, logrando un:
    - **Accuracy:** 95.38%
    - **AUC:** 0.99 (excelente capacidad para diferenciar entre tumores benignos y malignos).
    - **Recall para la clase maligna:** 95.0% (clave para minimizar falsos negativos en el diagnóstico).

    ---

    ### 2️⃣ Uso de Validación Cruzada (Cross-Validation)
    Para garantizar la robustez y generalización de los modelos, utilizamos **cross-validation** con:
    - **K-Fold:** Dividimos el dataset en 5 pliegues para asegurar que el modelo se evalúe con diferentes subconjuntos del dato.
    - **GridSearchCV:** Exploramos diferentes combinaciones de hiperparámetros en **Random Forest**, incluyendo:
  - Número de árboles (`n_estimators`): 100
  - Máxima profundidad (`max_depth`): Sin restricciones
  - Criterio de división (`criterion`): Entropía

    El uso de cross-validation permitió identificar la configuración óptima del modelo sin sobreajustarlo.

    ---

    ### 3️⃣ Comparación con Otras Métricas
    Para validar que **Random Forest** era la mejor elección, comparamos su desempeño frente a otros modelos usando métricas clave:
    - **Accuracy (Precisión):** Proporción de predicciones correctas.
    - **AUC (Área Bajo la Curva):** Medida de la capacidad del modelo para distinguir entre clases.
    - **Recall Maligno:** Importante para detectar correctamente los tumores malignos y evitar falsos negativos.

    Resultados Comparativos:
    | Modelo               | Accuracy (%) | AUC   | Recall Maligno (%) |
    |----------------------|--------------|-------|--------------------|
    | Random Forest        | 95.38        | 0.99  | 95.0               |
    | Logistic Regression  | 92.52        | 0.99  | 92.0               |
    | Support Vector Machines (SVM) | 93.60 | 0.97  | 93.0               |

    ---

    ### 4️⃣ Interpretación de las Características Clave
    El modelo de **Random Forest** permitió identificar las características más influyentes:
    - **`Mean_Radius` (Radio Promedio):** Indicador del tamaño del tumor.
    - **`Worst_Concavity` (Mayor Concavidad):** Refleja la irregularidad en los bordes del tumor.
    - **`Mean_Concavity` (Concavidad Promedio):** Complementa la detección de formas irregulares.

    Estas características son fundamentales para diferenciar entre tumores benignos y malignos.

    ---

    ### 5️⃣ Importancia del Machine Learning en la Medicina
    Este proyecto demuestra cómo el uso de **Machine Learning** puede complementar el juicio médico en el diagnóstico temprano de enfermedades críticas como el cáncer de mama. Aunque este trabajo es académico y no debe reemplazar evaluaciones médicas, destaca el valor de integrar tecnología avanzada en la salud.

    💖 **Gracias por explorar este proyecto!** Sigamos trabajando en soluciones tecnológicas que marquen la diferencia.
    """)

# 🤖 Pantalla 3: Predicciones Interactivas
elif page == "Predicciones Interactivas":
    st.title("🤖 Predicciones Interactivas")
    st.markdown("""
    En esta página puedes ingresar valores de ciertas características del tumor 
    para obtener una **predicción simulada** de si el tumor es **benigno** o **maligno**.  
    **Nota:** Esta funcionalidad es educativa y no debe usarse con fines médicos.
    """)

    # Entrada de datos por parte del usuario
    input_data = {}
    for feature in ["Mean_Radius", "Mean_Concavity", "Worst_Concavity"]:
        input_data[feature] = st.number_input(
            f"Ingrese el valor para {feature}:",
            min_value=0.0,
            step=0.1
        )

    # Botón de predicción
    if st.button("Predecir"):
        # Simulación de predicción usando el valor de Mean_Radius
        prediction = "Maligno" if input_data["Mean_Radius"] > 15 else "Benigno"
        
        # Mostrar el resultado de la predicción
        st.success(f"El modelo predice que el tumor es: **{prediction}**")
        st.info("Este resultado es una simulación y en ningún momento representa un diagnóstico médico real.")
