import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv(r"C:\Users\junji\OneDrive\Desktop\Data Analytics Major Project\WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Display first 10 rows with background gradient
df.head(10).style.background_gradient(cmap='Reds')

# Display last 10 rows with background gradient
df.tail(10).style.background_gradient(cmap="Reds")

# Check basic information about the dataset
df.info()

# Check for missing values
df.isna().sum()

# Check for duplicates
df.duplicated().sum()

# Separate categorical and numerical features
cat_features = df.select_dtypes(include='object').columns
num_features = df.select_dtypes(exclude='object').columns

# Information about categorical features
df[cat_features].info()

# Information about numerical features
df[num_features].info()

# Visualizing categorical features (Pie Chart for features with <= 5 unique values)
for col in cat_features:
    unique_vals = df[col].nunique()
    
    if unique_vals <= 5:
        # Pie Chart for categorical variables with <= 5 unique values
        counts = df[col].value_counts()
        colors = ['#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9', '#92A8D1']
        
        plt.figure(figsize=(7, 5))
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=colors, shadow=True)
        plt.title(f'{col} Distribution (Pie)', fontsize=13)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
        
    else:
        # Countplot for categorical features with more than 5 unique values
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(data=df, x=col, palette='Set2')
        plt.title(f'Distribution of {col} (Countplot)')
        for container in ax.containers:
            ax.bar_label(container)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Visualizing categorical features vs. Attrition
for i in cat_features:
    if i == 'Attrition':
        continue
    
    plt.figure(figsize=(10,6))
    ax = sns.countplot(data=df, x=i, hue='Attrition', palette='Set2')
    plt.title(f"{i} vs 'Attrition'")
    for container in ax.containers:
        ax.bar_label(container)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Visualizing numerical features vs. Attrition
for i in num_features:
    if (df[i].nunique() <= 20):
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(data=df, x=i, hue='Attrition')
        for container in ax.containers:
            ax.bar_label(container)
        plt.title(i)
        plt.ylabel('Count')
        plt.xlabel(i)
        plt.tight_layout()
        plt.show()

# Prepare a subset of numerical features with more than 9 unique values
numerical_df = df[num_features]
numerical_df = numerical_df.loc[:, numerical_df.nunique() > 9]

# Reshape the dataframe for visualization
long_df = numerical_df.melt(var_name='Feature', value_name='Value')

# Boxplot for numerical features
plt.figure(figsize=(12, 6))
sns.boxplot(data=long_df, x='Feature', y='Value', palette='Set2')
plt.xticks(rotation=70)
plt.title("Boxplot of Numerical Features")
plt.tight_layout()
plt.show()

# Boxplot for numerical features with more than 20 but <= 100 unique values
for i in num_features:
    if (df[i].nunique() > 20 and df[i].nunique() <= 100):
        plt.figure()
        sns.boxplot(df[i])
        plt.title(i)
        plt.xlabel(i)
        plt.tight_layout()
        plt.show()

# Map 'Attrition' column to numerical values
df['Attrition_num'] = df['Attrition'].map({'No': 0, 'Yes': 1})

# Correlation matrix for numerical features
numerical_df = df.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numerical_df.corr()

# Heatmap of correlations
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap (Including Attrition_num)", fontsize=14)
plt.xticks(rotation=70)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
