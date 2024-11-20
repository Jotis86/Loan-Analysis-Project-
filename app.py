import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Load data
@st.cache_data
def load_data():
    train_df = pd.read_csv("C:\\Users\\juane\\OneDrive\\Escritorio\\Datos\\challenge_lending_club_data_train.csv")
    eval_df = pd.read_csv("C:\\Users\\juane\\OneDrive\\Escritorio\\Datos\\challenge_lending_club_data_evaluation_notarget.csv")
    return train_df, eval_df

train_df, eval_df = load_data()

# Select only relevant columns
selected_columns = [
    'grade', 'sub_grade', 'short_emp', 'emp_length_num', 'home_ownership', 
    'dti', 'purpose', 'term', 'last_delinq_none', 'last_major_derog_none', 
    'revol_util', 'total_rec_late_fee', 'bad_loans'
]
train_df = train_df[selected_columns]

# Handle missing values
train_df.fillna(0, inplace=True)

# Separate features and target in the training set
X = train_df.drop('bad_loans', axis=1)
y = train_df['bad_loans']

# Load the model from the .pkl file
pipeline = joblib.load('Notebook/loan_model.pkl')

# Make predictions on the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Main image
st.image("images/Loan_1.png", use_column_width=True)

# Sidebar for navigation
st.sidebar.image("images/loan_cat.png", use_column_width=True)
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Project Objectives", "Methodology and Tools", "Visualizations", "Results"])

# Project Objectives
if menu == "Project Objectives":
    st.title("📊 Loan Analysis Project 💸")
    st.header("🎯 Project Objectives")
    st.write("""
    The main objective of this project is to analyze a loan dataset to identify key patterns and relationships that can help predict the risk of a loan. The specific objectives include:

    - 📈 **Data Analysis**: Conduct a thorough analysis of the dataset to identify patterns and trends that may influence the risk of a loan.
    - 🤖 **Machine Learning Model Construction**: Develop and train several machine learning models to predict the risk of a loan. The models will include logistic regression, random forest, and gradient boosting.
    - 📊 **Generate Visualizations**: Create informative visualizations to help interpret the results of the data analysis and machine learning models.
    - 🧠 **Provide Insights**: Provide valuable insights that can be used by financial institutions to improve decision-making in loan management.
    - 🔍 **Identify Key Factors**: Identify the key factors that influence the risk of a loan and how these factors can be mitigated.
    - 📉 **Risk Reduction**: Propose strategies to reduce the risk of loans based on the results of data analysis and machine learning models.
    """)

# Methodology and Tools
elif menu == "Methodology and Tools":
    st.title("🛠️ Methodology and Tools")
    st.header("📋 Methodology")
    st.write("""
    The project development process was carried out in several stages, each of which is crucial for the success of the analysis and model construction:

    1. **Data Loading and Cleaning**:
        - Load the data from the provided CSV files.
        - Handle missing values by replacing them with zeros or using appropriate imputation techniques.
        - Ensure that the data is in the correct format for analysis and modeling.

    2. **Exploratory Data Analysis (EDA)**:
        - Visualize the distributions of numerical and categorical variables.
        - Identify relationships and patterns between variables.
        - Detect and handle outliers that may affect model performance.

    3. **Data Preprocessing**:
        - Standardize numerical data to ensure that all features are on the same scale.
        - Encode categorical variables using techniques such as one-hot encoding.
        - Split the data into training and testing sets to evaluate model performance.

    4. **Model Construction**:
        - Train several machine learning models, including logistic regression, random forest, and gradient boosting.
        - Use cross-validation techniques to evaluate model performance and select the best model.

    5. **Model Evaluation**:
        - Evaluate the models using metrics such as accuracy, recall, F1 score, and ROC-AUC.
        - Compare the performance of different models and select the best model based on evaluation metrics.

    6. **Generate Visualizations**:
        - Create informative visualizations to help interpret the results of data analysis and machine learning models.
        - Use visualization libraries such as Matplotlib and Seaborn to create graphs and charts.

    7. **Propose Strategies**:
        - Propose strategies to reduce the risk of loans based on the results of data analysis and machine learning models.
        - Identify the key factors that influence the risk of a loan and how these factors can be mitigated.
    """)

    st.header("🔧 Tools Used")
    st.write("""
    - 🐍 **Python**: Main programming language.
    - 🐼 **Pandas**: For data manipulation and analysis.
    - 📊 **Matplotlib and Seaborn**: For data visualization.
    - 🤖 **Scikit-learn**: For data preprocessing and machine learning model construction.
    """)

# Visualizations
elif menu == "Visualizations":
    st.title("📊 Visualizations")
    st.header("Interest Rate Distribution")
    plt.figure(figsize=(10, 6))
    sns.histplot(train_df['revol_util'], kde=True)
    st.pyplot(plt)

    st.header("Loan Status by Home Ownership")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='home_ownership', hue='bad_loans', data=train_df)
    st.pyplot(plt)

    st.header("DTI Distribution")
    plt.figure(figsize=(10, 6))
    sns.histplot(train_df['dti'], kde=True)
    st.pyplot(plt)

    st.header("Employment Length Distribution")
    plt.figure(figsize=(10, 6))
    sns.histplot(train_df['emp_length_num'], kde=True)
    st.pyplot(plt)

    st.header("Home Ownership Distribution")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='home_ownership', data=train_df)
    st.pyplot(plt)

# Results
elif menu == "Results":
    st.title("📈 Results")
    st.header("📊 Best Model Metrics")
    st.write("""
    Several machine learning models were trained, including:
    - Logistic Regression
    - Random Forest
    - Gradient Boosting

    The best model was **Logistic Regression**. The evaluation metrics for the model are as follows:
    """)
    st.write(f"**Accuracy**: {accuracy:.2f}")
    st.write(f"**Precision**: {precision:.2f}")
    st.write(f"**Recall**: {recall:.2f}")
    st.write(f"**F1 Score**: {f1:.2f}")
    st.write(f"**ROC-AUC Score**: {roc_auc:.2f}")

    st.header("📉 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    st.header("🔮 Next Steps")
    st.write("""
    To improve the model, the following techniques can be applied:
    - **Hyperparameter Tuning**: Use techniques such as Grid Search or Random Search to find the best hyperparameters.
    - **Feature Engineering**: Create new features from existing ones to improve model performance.
    - **Model Ensembling**: Combine several models to improve the accuracy and robustness of predictions.
    - **Cross-Validation**: Use cross-validation techniques to evaluate the model more robustly.
    """)

# Button to go to GitHub repository
st.sidebar.title("Repository")
if st.sidebar.button("Go to GitHub Repository"):
    js = "window.open('https://github.com/Jotis86/Loan-Analysis-Project-')"
    st.sidebar.markdown(f'<a href="https://github.com/Jotis86/Loan-Analysis-Project-" target="_blank">GitHub</a>', unsafe_allow_html=True)
