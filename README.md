# 📊 Loan Analysis Project 💸

## 🎯 Objectives
- The goal of this project is to analyze a loan dataset to identify key patterns and relationships that can help predict whether a loan is at risk or not. Additionally, we aim to build a machine learning model that can accurately predict the risk of a loan based on various features.

## ⚙️ Functionality
- 📥 Data loading and cleaning.
- 🔍 Exploratory Data Analysis (EDA).
- 🛠️ Data preprocessing.
- 🧠 Building and evaluating machine learning models.
- 📊 Generating visualizations to interpret results.
- 🔮 Predicting loan status on an evaluation set.

## 🛠️ Tools Used
- Python 🐍: Main programming language.
- Pandas 🐼: For data manipulation and analysis.
- Matplotlib 📉 and Seaborn 🐟: For data visualization.
- Scikit-learn 🤖: For data preprocessing and machine learning model building.

## 🛠️ Development Process
- Library Import 📚
  - Importing necessary libraries for data manipulation, visualization, and machine learning.
- Data Loading 📥
  - Reading the CSV files for training and evaluation datasets.
- Data Exploration 🔍
  - Viewing the first and last rows, data summary, and descriptive statistics to understand the dataset.
- Data Cleaning 🧹
  - Handling missing values and ensuring the evaluation set has the same columns as the training set.
- Exploratory Data Analysis 📊
  - Visualizing distributions and relationships between variables, such as:
   - Distribution of Interest Rate 📉
   - Loan Status by Home Ownership 🏠
   - Employment Length vs. Loan Status 📅
   - Debt-to-Income Ratio vs. Loan Status 💳
- Data Preprocessing 🛠️
  - Standardizing numerical data and one-hot encoding categorical data to prepare for model training.
- Model Building 🧠
  - Defining and training multiple machine learning models, including:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
- Model Evaluation 📈
  - Evaluating the models using metrics such as:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - ROC-AUC Score
- Final Predictions 🔮
  - Applying the best-performing model to the evaluation set and saving the results.

## 📈 Results
- The trained Logistic Regression model achieved an accuracy of 81.6 % on the test set. The generated visualizations helped identify the most important features influencing loan status.

## 📝 Conclusions
 - Interest rate and annual income are key factors in predicting loan status.
 - The Logistic Regression model showed good performance in classifying loans.
 - Further improvement in data preprocessing and feature selection is needed to increase model accuracy.

## 📊 Visualizations
 - Distribution of Interest Rate 📉
 - Loan Status by Home Ownership 🏠
 - Employment Length vs. Loan Status 📅
 - Debt-to-Income Ratio vs. Loan Status 💳

## 🧠 Model and Metrics
 - Model: Logistic Regression
 - Evaluation Metrics: Confusion matrix, ROC curve, precision, recall, F1-score.

## 🗂️ Project Structure
- Data
- NoteBook
- Predictions

## 📬 Contact
- For any inquiries, you can contact me at jotaduranbon@gmail.com.

