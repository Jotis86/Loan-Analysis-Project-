import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Cargar datos
@st.cache_data
def load_data():
    train_df = pd.read_csv("C:\\Users\\juane\\OneDrive\\Escritorio\\Datos\\challenge_lending_club_data_train.csv")
    eval_df = pd.read_csv("C:\\Users\\juane\\OneDrive\\Escritorio\\Datos\\challenge_lending_club_data_evaluation_notarget.csv")
    return train_df, eval_df

train_df, eval_df = load_data()

# Seleccionar solo las columnas relevantes
selected_columns = [
    'grade', 'sub_grade', 'short_emp', 'emp_length_num', 'home_ownership', 
    'dti', 'purpose', 'term', 'last_delinq_none', 'last_major_derog_none', 
    'revol_util', 'total_rec_late_fee', 'bad_loans'
]
train_df = train_df[selected_columns]

# Manejar valores faltantes
train_df.fillna(0, inplace=True)

# Separar características y objetivo en el conjunto de entrenamiento
X = train_df.drop('bad_loans', axis=1)
y = train_df['bad_loans']

# Cargar el modelo desde el archivo .pkl
pipeline = joblib.load('Notebook/loan_model.pkl')

# Hacer predicciones en el conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = pipeline.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Imagen principal
st.image("images/Loan_1.png", use_column_width=True)

# Sidebar para navegación
st.sidebar.image("images/loan_cat.png", use_column_width=True)
st.sidebar.title("Navegación")
menu = st.sidebar.radio("Ir a", ["Objetivos del Proyecto", "Metodología y Herramientas", "Visualizaciones", "Resultados"])

# Objetivos del Proyecto
if menu == "Objetivos del Proyecto":
    st.title("📊 Loan Analysis Project 💸")
    st.header("🎯 Objetivos del Proyecto")
    st.write("""
    - 📈 **Analizar** un conjunto de datos de préstamos para identificar patrones y relaciones clave.
    - 🤖 **Construir** un modelo de machine learning que pueda predecir el riesgo de un préstamo.
    - 📊 **Generar** visualizaciones para interpretar los resultados.
    - 🧠 **Proveer** insights valiosos para la toma de decisiones en la gestión de préstamos.
    """)

# Metodología y Herramientas
elif menu == "Metodología y Herramientas":
    st.title("🛠️ Metodología y Herramientas")
    st.header("📋 Metodología")
    st.write("""
    1. **Carga y limpieza de datos**: Cargar los datos y manejar los valores faltantes.
    2. **Análisis Exploratorio de Datos (EDA)**: Visualizar distribuciones y relaciones entre variables.
    3. **Preprocesamiento de Datos**: Estandarizar datos numéricos y codificar datos categóricos.
    4. **Construcción de Modelos**: Entrenar varios modelos de machine learning.
    5. **Evaluación de Modelos**: Evaluar los modelos utilizando métricas como precisión, recall, F1 score y ROC-AUC.
    """)

    st.header("🔧 Herramientas Usadas")
    st.write("""
    - 🐍 **Python**: Lenguaje de programación principal.
    - 🐼 **Pandas**: Para manipulación y análisis de datos.
    - 📊 **Matplotlib y Seaborn**: Para visualización de datos.
    - 🤖 **Scikit-learn**: Para preprocesamiento de datos y construcción de modelos de machine learning.
    """)

# Visualizaciones
elif menu == "Visualizaciones":
    st.title("📊 Visualizaciones")
    st.header("Distribución de la Tasa de Interés")
    plt.figure(figsize=(10, 6))
    sns.histplot(train_df['revol_util'], kde=True)
    st.pyplot(plt)

    st.header("Estado del Préstamo por Propiedad de Vivienda")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='home_ownership', hue='bad_loans', data=train_df)
    st.pyplot(plt)

    st.header("Distribución de la DTI")
    plt.figure(figsize=(10, 6))
    sns.histplot(train_df['dti'], kde=True)
    st.pyplot(plt)

    st.header("Distribución de la Longitud del Empleo")
    plt.figure(figsize=(10, 6))
    sns.histplot(train_df['emp_length_num'], kde=True)
    st.pyplot(plt)

    st.header("Distribución de la Propiedad de Vivienda")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='home_ownership', data=train_df)
    st.pyplot(plt)

# Resultados
elif menu == "Resultados":
    st.title("📈 Resultados")
    st.header("📊 Métricas del Mejor Modelo")
    st.write("""
    Se entrenaron varios modelos de machine learning, incluyendo:
    - Regresión Logística
    - Bosque Aleatorio
    - Gradient Boosting

    El mejor modelo fue la **Regresión Logística**. A continuación se presentan las métricas de evaluación del modelo:
    """)
    st.write(f"**Precisión**: {accuracy:.2f}")
    st.write(f"**Precisión**: {precision:.2f}")
    st.write(f"**Recall**: {recall:.2f}")
    st.write(f"**F1 Score**: {f1:.2f}")
    st.write(f"**ROC-AUC Score**: {roc_auc:.2f}")

    st.header("📉 Matriz de Confusión")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    st.header("🔮 Próximos Pasos")
    st.write("""
    Para mejorar el modelo, se pueden aplicar las siguientes técnicas:
    - **Ajuste de Hiperparámetros**: Utilizar técnicas como Grid Search o Random Search para encontrar los mejores hiperparámetros.
    - **Ingeniería de Características**: Crear nuevas características a partir de las existentes para mejorar el rendimiento del modelo.
    - **Ensamblado de Modelos**: Combinar varios modelos para mejorar la precisión y robustez de las predicciones.
    - **Validación Cruzada**: Utilizar técnicas de validación cruzada para evaluar el modelo de manera más robusta.
    """)

# Botón para ir al repositorio de GitHub
st.sidebar.title("Repositorio")
if st.sidebar.button("Ir al repositorio de GitHub"):
    st.sidebar.markdown("[GitHub](https://github.com/Jotis86/Loan-Analysis-Project-)")
