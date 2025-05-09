# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
dataset = pd.read_csv(r"C:\Users\junji\OneDrive\Desktop\Data Analytics Major Project\WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Display basic info
print(dataset.head())
dataset.info()

# Visualize missing values
plt.figure(figsize=(10, 4))
sns.heatmap(dataset.isnull(), yticklabels=False, cbar=False, cmap='viridis')
sns.set_style('darkgrid')
plt.title("Missing Values Heatmap")
plt.show()

# Countplot for Attrition
sns.countplot(x='Attrition', data=dataset)
plt.title("Attrition Count")
plt.show()

# Scatter plot for Age vs DailyRate
sns.lmplot(x='Age', y='DailyRate', hue='Attrition', data=dataset)
plt.title("Age vs DailyRate by Attrition")
plt.show()

# Boxplot for MonthlyIncome vs Attrition
plt.figure(figsize=(10, 6))
sns.boxplot(y='MonthlyIncome', x='Attrition', data=dataset)
plt.title("Monthly Income vs Attrition")
plt.show()

# Drop unnecessary columns
columns_to_drop = ['EmployeeCount', 'StandardHours', 'EmployeeNumber', 'Over18']
dataset.drop(columns=columns_to_drop, axis=1, inplace=True)

# Target and features
y = dataset['Attrition']
X = dataset.drop('Attrition', axis=1)

# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Yes = 1, No = 0

# One-hot encode categorical features
categorical_columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# Train-Test Split with stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=40)

# Initialize Random Forest with improvements
rf = RandomForestClassifier(
    n_estimators=200, 
    criterion='entropy', 
    class_weight='balanced', 
    max_depth=10,
    random_state=40
)
rf.fit(X_train, y_train)

# Function to print classification results
def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        print("Train Result:")
        print("------------")
        print(f"Classification Report:\n{classification_report(y_train, clf.predict(X_train))}\n")
        print(f"Confusion Matrix:\n{confusion_matrix(y_train, clf.predict(X_train))}\n")
        res = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
        print(f"Average Accuracy (CV): {np.mean(res):.4f}")
        print(f"Accuracy SD: {np.std(res):.4f}")
        print("----------------------------------------------------------")
    else:
        print("Test Result:")
        print("-----------")
        print(f"Classification Report:\n{classification_report(y_test, clf.predict(X_test))}\n")
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, clf.predict(X_test))}\n")
        print(f"Accuracy Score: {accuracy_score(y_test, clf.predict(X_test)):.4f}")
        print("-----------------------------------------------------------")

# Print train and test scores
print_score(rf, X_train, y_train, X_test, y_test, train=True)
print_score(rf, X_train, y_train, X_test, y_test, train=False)

# Feature Importance Plot
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values(ascending=False)
plt.figure(figsize=(14, 6))
importances_sorted.plot(kind='bar')
plt.title("Feature Importances")
plt.tight_layout()
plt.show()

import shap

# Create SHAP explainer
explainer = shap.TreeExplainer(rf)

# Compute SHAP values for the test set
shap_values = explainer.shap_values(X_test)

# Check the shapes of SHAP values and X_test to make sure they match
print(f"Shape of SHAP values for class 1: {shap_values[1].shape}")
print(f"Shape of X_test: {X_test.shape}")

# If the shapes match, plot the SHAP summary plot for class 1 (Attrition = Yes)
if shap_values[1].shape == X_test.shape:
    shap.summary_plot(shap_values[1], X_test, plot_type="bar")
    shap.summary_plot(shap_values[1], X_test)  # Detailed summary plot
else:
    print("Shape mismatch: Check SHAP values and input data.")