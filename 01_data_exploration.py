# %% [markdown]
# # Day 3-4: Data Exploration & Cleaning
# Employee Attrition Prediction Project
# This notebook explores the IBM HR Attrition dataset before we build any model.

# %%
# Import the libraries we need for data exploration
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Load the dataset into a DataFrame
# NOTE: this CSV must sit in the PROJECT ROOT, one level above this notebooks/ folder
df = pd.read_csv("../WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Display the first 5 rows to get a quick look
df.head()

# %%
# Check how many rows and columns we have
print("Shape of dataset:", df.shape)

# Get column names, data types, and non-null counts
df.info()

# %%
# Check for missing values in every column
# .isnull() marks each cell True/False, .sum() adds up the Trues per column
print("Missing values per column:\n")
print(df.isnull().sum())

# %%
# Look at basic statistics for numeric columns
# This shows mean, min, max, std deviation etc. — helps spot outliers or weird values
df.describe()

# %%
# Check which columns are "constant" (same value for every row)
# These give the model ZERO useful information, so they are candidates for removal
for col in df.columns:
    if df[col].nunique() == 1:
        print(f"Constant column found: {col} -> only value is {df[col].unique()}")

# %%
# Check our target variable: Attrition (Yes/No)
print(df['Attrition'].value_counts())
print("\nPercentage:")
print(df['Attrition'].value_counts(normalize=True) * 100)

# %% [markdown]
# ### Observation
# This dataset is "imbalanced" — far more employees stay (No) than leave (Yes).
# This matters later: accuracy alone will look artificially high, so we will also
# check precision/recall/F1-score, not just accuracy, when evaluating our model.

# %%
# Drop the useless constant columns + the pure ID column
# EmployeeCount, StandardHours, Over18 -> constant (no variation, no predictive value)
# EmployeeNumber -> just a unique ID, not a real feature
df_clean = df.drop(columns=['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'])
print("New shape after dropping useless columns:", df_clean.shape)

# %%
# Separate columns into numeric and categorical (text) types
numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()

print("Numeric columns:", numeric_cols)
print("\nCategorical columns:", categorical_cols)

# %%
# Visualize attrition count
plt.figure(figsize=(6,4))
sns.countplot(data=df_clean, x='Attrition', palette='Set2')
plt.title("Employee Attrition Count")
plt.xlabel("Attrition")
plt.ylabel("Number of Employees")
plt.show()

# %%
# Visualize how OverTime relates to Attrition
plt.figure(figsize=(6,4))
sns.countplot(data=df_clean, x='OverTime', hue='Attrition', palette='Set1')
plt.title("Attrition by OverTime Status")
plt.show()

# %%
# Correlation heatmap for numeric columns
# This shows which numeric features move together — helps spot strong relationships
plt.figure(figsize=(14,10))
sns.heatmap(df_clean[numeric_cols].corr(), cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

# %%
# Save the cleaned dataframe so train_model.py can reuse it without repeating these steps
df_clean.to_csv("../cleaned_attrition_data.csv", index=False)
print("Saved cleaned_attrition_data.csv to project root")
