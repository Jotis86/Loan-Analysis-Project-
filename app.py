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

# Guardar la columna 'id' del conjunto de evaluación
eval_ids = eval_df['id']

# Seleccionar solo las columnas relevantes
selected_columns = [
    'grade', 'sub_grade', 'short_emp', 'emp_length_num', 'home_ownership', 
    'dti', 'purpose', 'term', 'last_delinq_none', 'last_major_derog_none', 
    'revol_util', 'total_rec_late_fee', 'bad_loans'
]
train_df = train_df[selected_columns]
eval_df = eval_df[[col for col in selected_columns if col != 'bad_loans']]

# Manejar valores faltantes
train_df.fillna(0, inplace=True)
eval_df.fillna(0, inplace=True)

# Asegurarse de que el conjunto de evaluación tenga las mismas columnas que el conjunto de entrenamiento
missing_cols = set(train_df.columns) - set(eval_df.columns)
for col in missing_cols:
    eval_df[col] = 0

# Alinear columnas del conjunto de evaluación con el conjunto de entrenamiento
eval_df = eval_df[train_df.columns.drop('bad_loans')]

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

# Sidebar para navegación
st.sidebar.title("Navegación")
menu = st.sidebar.radio("Ir a", ["Objetivos del Proyecto", "Metodología y Herramientas", "Resultados"])

# Objetivos del Proyecto
if menu == "Objetivos del Proyecto":
    st.title("📊 Loan Analysis Project 💸")
    st.header("🎯 Objetivos del Proyecto")
    st.write("""
    - Analizar un conjunto de datos de préstamos para identificar patrones y relaciones clave.
    - Construir un modelo de machine learning que pueda predecir el riesgo de un préstamo.
    - Generar visualizaciones para interpretar los resultados.
    """)

# Metodología y Herramientas
elif menu == "Metodología y Herramientas":
    st.title("Metodología y Herramientas")
    st.header("Metodología")
    st.write("""
    1. **Carga y limpieza de datos**: Cargar los datos y manejar los valores faltantes.
    2. **Análisis Exploratorio de Datos (EDA)**: Visualizar distribuciones y relaciones entre variables.
    3. **Preprocesamiento de Datos**: Estandarizar datos numéricos y codificar datos categóricos.
    4. **Construcción de Modelos**: Entrenar varios modelos de machine learning.
    5. **Evaluación de Modelos**: Evaluar los modelos utilizando métricas como precisión, recall, F1 score y ROC-AUC.
    """)

    st.header("Herramientas Usadas")
    st.write("""
    - **Python**: Lenguaje de programación principal.
    - **Pandas**: Para manipulación y análisis de datos.
    - **Matplotlib y Seaborn**: Para visualización de datos.
    - **Scikit-learn**: Para preprocesamiento de datos y construcción de modelos de machine learning.
    """)

# Resultados
elif menu == "Resultados":
    st.title("Resultados")
    st.header("Distribución de la Tasa de Interés")
    plt.figure(figsize=(10, 6))
    sns.histplot(train_df['revol_util'], kde=True)
    st.pyplot(plt)

    st.header("Estado del Préstamo por Propiedad de Vivienda")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='home_ownership', hue='bad_loans', data=train_df)
    st.pyplot(plt)

# Botón para ir al repositorio de GitHub
st.sidebar.title("Repositorio")
if st.sidebar.button("Ir al repositorio de GitHub"):
    st.sidebar.markdown("[GitHub](https://github.com/Jotis86/Loan-Analysis-Project-)")


