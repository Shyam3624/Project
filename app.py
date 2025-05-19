# Basic Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset
df = pd.read_csv(r"C:\Users\junji\OneDrive\Desktop\Data Analytics Major Project\WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Preview Data
df.head(10).style.background_gradient(cmap='Reds')
df.tail(10).style.background_gradient(cmap="Reds")

# Data Info
df.info()
print(df.isna().sum())           # Check for missing values
print(df.duplicated().sum())     # Check for duplicates

# Separate features
cat_features = df.select_dtypes(include='object').columns
num_features = df.select_dtypes(exclude='object').columns

# EDA: Categorical Features (Pie or Countplots)
for col in cat_features:
    unique_vals = df[col].nunique()
    
    if unique_vals <= 5:
        plt.figure(figsize=(7, 5))
        counts = df[col].value_counts()
        colors = ['#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9', '#92A8D1']
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=colors, shadow=True)
        plt.title(f'{col} Distribution (Pie)', fontsize=13)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(data=df, x=col, palette='Set2')
        plt.title(f'Distribution of {col} (Countplot)')
        for container in ax.containers:
            ax.bar_label(container)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# EDA: Categorical Features vs Attrition
for i in cat_features:
    if i == 'Attrition':
        continue
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df, x=i, hue='Attrition', palette='Set2')
    plt.title(f"{i} vs 'Attrition'")
    for container in ax.containers:
        ax.bar_label(container)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# EDA: Numerical Features
for i in num_features:
    if df[i].nunique() <= 20:
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(data=df, x=i, hue='Attrition')
        for container in ax.containers:
            ax.bar_label(container)
        plt.title(i)
        plt.tight_layout()
        plt.show()

# Boxplot: Numerical Features with > 9 unique values
numerical_df = df[num_features]
numerical_df = numerical_df.loc[:, numerical_df.nunique() > 9]
long_df = numerical_df.melt(var_name='Feature', value_name='Value')
plt.figure(figsize=(12, 6))
sns.boxplot(data=long_df, x='Feature', y='Value', palette='Set2')
plt.xticks(rotation=70)
plt.title("Boxplot of Numerical Features")
plt.tight_layout()
plt.show()

# Extra Boxplots: Features with >20 and <=100 unique values
for i in num_features:
    if 20 < df[i].nunique() <= 100:
        plt.figure()
        sns.boxplot(df[i])
        plt.title(i)
        plt.xlabel(i)
        plt.tight_layout()
        plt.show()

# Encode 'Attrition'
df['Attrition_num'] = df['Attrition'].map({'No': 0, 'Yes': 1})

# Correlation Heatmap
correlation_matrix = df.select_dtypes(include=['int64', 'float64']).corr()
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap (Including Attrition_num)", fontsize=14)
plt.xticks(rotation=70)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# -------------------------
# ✅ Model Building & Evaluation
# -------------------------

# Label Encode Categorical Features
df_encoded = df.copy()
label_encoders = {}
for col in cat_features:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

# Prepare Features and Target
drop_cols = ['Attrition', 'Attrition_num', 'EmployeeNumber', 'Over18', 'EmployeeCount', 'StandardHours']
X = df_encoded.drop(columns=drop_cols)
y = df_encoded['Attrition_num']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction & Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n🔍 Confusion Matrix:")
print(cm)

# Classification Report
print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred))

# -------------------------
# ✅ SHAP Feature Importance
# -------------------------

# SHAP Initialization
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# SHAP Summary Plot
shap.summary_plot(shap_values, X_test)
