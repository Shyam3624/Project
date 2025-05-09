
# 💼 HR Analytics: Predicting Employee Attrition

## 📌 Project Overview
This project aims to analyze employee data and predict attrition using machine learning techniques. The primary goal is to uncover the key factors contributing to employee turnover and assist HR teams in taking proactive retention measures.

## 🛠️ Tools & Technologies
- Python (Pandas, NumPy, Matplotlib, Seaborn, Sklearn)
- Power BI (for interactive dashboards)
- SHAP (for model explainability)

## 📊 Exploratory Data Analysis (EDA)
Performed comprehensive EDA to understand patterns and trends in:
- Department-wise attrition
- Salary bands
- Job satisfaction and promotions
- Overtime and work-life balance
- Correlation of numerical features

### EDA Techniques Used:
- Count plots & pie charts for categorical data
- Boxplots for outlier detection
- Heatmaps for correlation analysis

## 🔍 Feature Engineering
- Converted categorical Attrition to numerical using label encoding
- Separated numerical and categorical features
- Removed duplicate records and confirmed no missing values

## 🤖 Model Building
Built and evaluated classification models to predict employee attrition:
- Logistic Regression
- Decision Tree Classifier

### Metrics Used:
- Accuracy Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)

## 🎯 SHAP Value Analysis
Used SHAP (SHapley Additive exPlanations) to:
- Understand the impact of each feature on model predictions
- Visualize feature importance using summary plots

## 📈 Power BI Dashboard
Created an interactive dashboard to visualize:
- Attrition distribution by department, salary level, overtime
- Impact of promotions and job role on attrition

## 📄 Deliverables
- ✔️ EDA Notebook (`EDA_Attrition_Analysis.ipynb`)
- ✔️ Trained ML Models (`logistic_model.pkl`, `tree_model.pkl`)
- ✔️ SHAP Analysis (`shap_summary_plot.png`)
- ✔️ Power BI Dashboard (`Attrition_Insights.pbix`)
- ✔️ Attrition Prevention Suggestions Report (`Attrition_Suggestions.pdf`)

## 📚 Conclusion
The project successfully identified key attrition drivers such as:
- Overtime
- Lack of promotion
- Job involvement
- Work-life balance

These insights can help HR departments implement targeted strategies to improve employee retention.
