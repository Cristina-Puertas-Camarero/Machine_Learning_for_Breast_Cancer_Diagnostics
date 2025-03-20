# Importar las librerías necesarias
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Backend recomendado para Streamlit

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
    Este proyecto explora cómo las técnicas de **Machine Learning** pueden asistir en el diagnóstico temprano de tumores mamarios, clasificándolos en **benignos** o **malignos**.El cáncer de mama es uno de los desafíos más importantes en el ámbito de la salud pública a nivel global, siendo una de las principales causas de mortalidad en mujeres. Según la Organización Mundial de la Salud (OMS), el diagnóstico temprano y el tratamiento adecuado son fundamentales para aumentar las probabilidades de supervivencia. En este contexto, la ciencia y la tecnología han trabajado de la mano para ofrecer herramientas innovadoras que permitan identificar y tratar esta enfermedad desde sus etapas más tempranas.  
    """)
    st.image("cancer.jpg", caption="Lucha contra el cáncer de mama", use_container_width=True)
    st.markdown("""
    ### Contexto del Proyecto:
    - **Datos:** Extraídos del [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29).
    - **Propósito:** Crear un modelo predictivo y analizar características clave relacionadas con el diagnóstico.

    Este trabajo es exclusivamente **académico** y forma parte de mi portafolio como estudiante de ciencia de datos. Este estudio utiliza datos del reconocido repositorio UCI Machine Learning Repository, específicamente del conjunto de datos de diagnóstico de cáncer de mama de Wisconsin. Estas investigaciones tienen un propósito claro: desarrollar modelos predictivos basados en técnicas de Machine Learning que no solo clasifiquen tumores entre benignos y malignos, sino que también proporcionen información sobre las características más relevantes para el diagnóstico. La combinación de avances tecnológicos y análisis estadístico busca apoyar a los profesionales médicos en la toma de decisiones más rápidas y precisas. 
    **Autora:** Cristina Puertas Camarero
    """)

# 📊 Pantalla 2: Resultados y Análisis
elif page == "Resultados y Análisis":
    st.title("📊 Resultados y Análisis del Proyecto")
    st.markdown("""
    En esta sección exploraremos las visualizaciones, el análisis de características, la importancia de las variables y las métricas de los modelos utilizados en este estudio. También se detalla cómo hemos llegado a nuestras conclusiones.  
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

    # 🛠️ Descripción de los pasos del proyecto
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

    # Conclusiones del Proyecto
    st.header("🔍 Conclusiones del Proyecto")
    st.markdown("""
    - Este proyecto resaltó el valor de las características más influyentes, como `Mean_Radius` y `Worst_Concavity`, en la predicción del diagnóstico.
    - El modelo **Random Forest** demostró ser la opción más efectiva, con una precisión y un desempeño sobresalientes.
    - Si bien este proyecto es académico, muestra cómo **Machine Learning** puede ser una herramienta prometedora para apoyar en diagnósticos médicos, reduciendo el margen de error y mejorando el diagnóstico temprano.  

    💖 **Gracias por explorar este trabajo educativo!**
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
        input_data[feature] = st.number_input(f"Ingrese el valor para {feature}:", min_value=0.0, step=0.1)

    # Botón de predicción
    if st.button("Predecir"):
        prediction = "Maligno" if input_data["Mean_Radius"] > 15 else "Benigno"
        st.success(f"El modelo predice que el tumor es: **{prediction}**")
        st.info("Este resultado es una simulación y en ningún momento representa un diagnóstico médico real.")




