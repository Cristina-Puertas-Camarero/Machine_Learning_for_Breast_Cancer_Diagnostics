# 🌸 Machine Learning para Diagnóstico de Cáncer de Mama 🌸

El cáncer de mama es una de las principales causas de muerte en mujeres a nivel mundial. Este proyecto busca contribuir al diagnóstico temprano y preciso mediante la implementación de modelos de **Machine Learning** que permitan clasificar tumores como **benignos** o **malignos**. La detección temprana es crucial para mejorar las tasas de supervivencia y, con ello, salvar vidas. Queremos abordar este tema con el respeto y la importancia que merece, teniendo como objetivo final apoyar al sector médico y a los pacientes.

---

## 🗂️ **Estructura del Proyecto**

El proyecto se ha dividido en múltiples fases, con una organización clara para asegurar un flujo de trabajo eficiente. A continuación, presentamos cómo hemos estructurado las carpetas y notebooks:

# 📂 Machine_Learning_for_Breast_Cancer_Diagnostics

### 📁 data
**Carpeta que contiene los datasets utilizados:**
- `4_dataset_sin_correlaciones_altas.csv`

---

### 📁 notebooks
**Carpeta con todos los Jupyter Notebooks:**
- `1_visualizacion.ipynb` – Paso 1: Visualización inicial de datos
- `2_limpieza.ipynb` – Paso 2: Limpieza y preprocesamiento de datos
- `3_eda.ipynb` – Paso 3: Análisis exploratorio de datos
- `4_modelado_predictivo.ipynb` – Paso 4: Entrenamiento y evaluación de modelos
- `5_hiperparametros.ipynb` – Paso 5: Optimización de hiperparámetros
- `6_streamlit.ipynb` – Desarrollo de la aplicación Streamlit
- `7_conclusiones.ipynb` – Conclusiones y reflexiones finales

---

### `app.py`
**Archivo para ejecutar la aplicación Streamlit.**

---

### `README.md`
**Archivo principal de documentación.**

---

## 🚀 **Objetivo del Proyecto**

Este proyecto se basa en un dataset público disponible en la página de **[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)**, utilizado frecuentemente con fines educativos y de investigación en ciencia de datos. El objetivo principal de este trabajo fue construir un pipeline de Machine Learning para diagnosticar tumores como **benignos** o **malignos**, y practicar habilidades como:
- Limpieza y preprocesamiento de datos.
- Análisis exploratorio de datos (EDA).
- Construcción y evaluación de modelos predictivos.
- Interpretación de resultados.
- Desarrollo de aplicaciones interactivas con Streamlit.

Este proyecto forma parte de mi portafolio como estudiante y profesional en formación, reflejando mi aprendizaje práctico en un tema de gran relevancia.

---

## 📈 **Resultados Destacados**

- **Modelos Finales:**
   - El modelo de **Bosques Aleatorios**, tras su optimización, alcanzó un **AUC de 0.99**, destacándose como el más robusto.
   - La **Regresión Logística** también logró un **AUC de 0.99**, siendo una opción interpretable para contextos médicos.

- **Características Clave:**
   - `Mean_Radius` (radio promedio): Una medida del tamaño del tumor que resulta altamente discriminativa.
   - `Worst_Concavity` (mayor concavidad): Representa las irregularidades en el contorno del tumor.
   - `Mean_Concavity` (concavidad promedio): Complementa la información sobre la forma del tumor.

---

## 🖥️ **Cómo Ejecutar el Proyecto**

### **Paso 1: Instalar Dependencias**
Clona este repositorio y asegúrate de instalar las librerías requeridas
Sigue el orden de los notebooks para obtener todos los resultados
Ejecuta Streamlit para las visualizaciones

## ✨ Conclusiones

### Machine Learning como apoyo médico:
Este proyecto demostró cómo los modelos de clasificación pueden asistir en tareas críticas como el diagnóstico temprano de cáncer, ofreciendo métricas sólidas y modelos interpretables.

---

### Uso educativo:
Este trabajo fue realizado para fines **académicos y prácticos** como parte de mi formación en ciencia de datos. **No pretende ser un sistema clínico ni reemplazar la evaluación médica.**

---

### Características relevantes:
El análisis reveló que variables como el **tamaño** y la **irregularidad en el contorno del tumor** son claves para clasificar su naturaleza.

---

### Siguientes pasos:
- **Validar los modelos con datos externos** para asegurar su capacidad de generalización.
- **Desplegar la aplicación interactiva en una plataforma en la nube** para compartir el trabajo.

---

## 🙏 Consideraciones Éticas
El cáncer de mama es un tema de gran sensibilidad, y por ello queremos dejar claro que:

1. **Este proyecto no es un diagnóstico médico:**  
   Es un trabajo académico basado en un dataset público, diseñado para practicar y mejorar habilidades en ciencia de datos.

2. **Importancia del diagnóstico profesional:**  
   Ningún modelo de Machine Learning debería reemplazar la experiencia y juicio de un médico especializado.

Esperamos que este trabajo inspire más investigaciones en la intersección de la tecnología y la salud, siempre priorizando el bienestar de los pacientes. 🌟

---

## ✍️ Sobre Mí
Soy una apasionada por la ciencia de datos y este proyecto forma parte de mi portafolio. Si deseas saber más sobre mi trabajo, no dudes en contactarme:

- **LinkedIn:** [Cristina Puertas Camarero](https://www.linkedin.com/in/cristina-puertas-camarero-8955a6349/)
- **Correo:** cris.puertascamarero@gmail.com.com

¡Gracias por leer y considerar este análisis!






