# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 03:38:08 2024

@author: Test
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#  Load the original dataset into DataFrame df
df = pd.read_csv('E:/360 Digi Project/Project 1/Dataset1/Economic_data.csv')

# Print initial info and data preview
print(df.info())
print(df.head())

# Check data types
df.dtypes

# -------------------------------------------------------Type Casting---------------------------------------------------------

# Check and cast columns only if necessary
for col in df.columns:
    if df[col].dtype == 'object':
        # Cast object to category if unique values are significantly less than total records
        if not pd.api.types.is_categorical_dtype(df[col]) and len(df[col].unique()) / len(df) < 0.5:
            df[col] = df[col].astype('category')
            print(f"Column '{col}' casted to 'category'")
        else:
            print(f"Column '{col}' is already optimal or not suitable for casting to 'category'")
    
    # Checking whether casting is required for columns in float datatype
    elif df[col].dtype == 'float64':
        # Cast float to integer only if all values are whole numbers
        if (df[col] % 1).sum() == 0 and not pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype('int64')
            print(f"Column '{col}' casted to 'int64'")
        else:
            print(f"Column '{col}' does not require casting or has fractional values")
    
    elif df[col].dtype == 'int64':
        # Integers already have the correct type
        print(f"Column '{col}' is already of type 'int64', no casting required")

# --------------------------------------------------Type Casting for Specific Date Column------------------------------------------------
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

print(df.dtypes)

# ---------------------------------------------------------------Missing Values---------------------------------------------

# Check for missing values
Missing_values = df.isnull().sum()
print(Missing_values)

# Fill missing numerical values with mean
for col in df.select_dtypes(include='float64').columns:
    df[col].fillna(df[col].mean(), inplace=True)
    
# Fill missing categorical values with mode
for col in df.select_dtypes(include='category').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Re-check missing values
Missing_values = df.isnull().sum() 
print(Missing_values)

# --------------------------------------------------Duplicate Values---------------------------------------------------
# Checking the presence of duplicates and removing if required
Duplicate_Values = df.duplicated().sum()
print(Duplicate_Values)

# -------------------------------------------------Outlier Treatment-------------------------------------------------

# Outlier Detection and Treatment
# Graph before outlier removal
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(df[col], color='skyblue')
    plt.title(f"Boxplot for {col}")
    plt.show()

# Function to detect outliers using the IQR method
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

# Detect outliers for all numeric columns
outlier_counts = {}
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    outliers = detect_outliers_iqr(df, col)
    outlier_counts[col] = len(outliers)
    print(f"Column '{col}' has {len(outliers)} outliers")

# Treat outliers by capping them at the IQR boundaries
def cap_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])

# Apply outlier treatment
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    cap_outliers_iqr(df, col)
    print(f"Outliers in column '{col}' treated")

# Optionally, remove rows with outliers
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# Remove outliers for all numeric columns (if desired)
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    original_len = len(df)
    df = remove_outliers_iqr(df, col)
    new_len = len(df)
    print(f"Outliers removed from column '{col}': {original_len - new_len} rows")

# Recheck for outliers after treatment or removal
outlier_counts_after = {}
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    outliers = detect_outliers_iqr(df, col)
    outlier_counts_after[col] = len(outliers)
    print(f"After treatment, column '{col}' has {len(outliers)} outliers")

# Display summary of outlier detection and treatment
print("Outlier Summary Before Treatment:")
print(outlier_counts)
print("Outlier Summary After Treatment/Removal:")
print(outlier_counts_after)

# Display boxplots after treatment
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(df[col], color='skyblue')
    plt.title(f"Boxplot for {col}")
    plt.show()

# ------------------------------------------------------------Zero and Near Zero Variance----------------------------------------

# Check and Handle Zero and Near-Zero Variance Features
variances = df.select_dtypes(include=['float64', 'int64']).var()

threshold = 0.01
zero_variance_features = variances[variances <= threshold].index.tolist()

if zero_variance_features:
    print(f"Zero/Near-Zero Variance Features: {zero_variance_features}")
    drop_features = input("Do you want to drop these features? (yes/no): ").strip().lower()
    if drop_features == 'yes':
        df.drop(columns=zero_variance_features, inplace=True)
        print(f"Features dropped: {zero_variance_features}")
    else:
        print("No features were dropped.")
else:
    print("No Zero or Near-Zero Variance Features found.")

print(f"Remaining Features: {df.columns.tolist()}")

# ------------------------------------------------Transformation------------------------------------------------------------------

# Log Transformation
df_log_transformed = df.copy()
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    if (df[col] > 0).all():  # Ensure no negative or zero values
        df_log_transformed[col] = np.log(df[col])
        print(f"Log transformation applied to column: {col}")
    else:
        print(f"Skipping column '{col}' as it contains non-positive values")

# Standard Scaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[df.select_dtypes(include=['float64', 'int64']).columns] = scaler.fit_transform(
    df.select_dtypes(include=['float64', 'int64'])
)
print("Standardization applied to numerical columns")


# Normalization (Min-Max Scaling)

# Min-Max Normalization
from sklearn.preprocessing import MinMaxScaler

# Create a copy of the dataframe for normalization
df_normalized = df.copy()

# Apply Min-Max scaling to the numerical columns
scaler = MinMaxScaler()

# Normalizing numerical columns
df_normalized[df.select_dtypes(include=['float64', 'int64']).columns] = scaler.fit_transform(
    df.select_dtypes(include=['float64', 'int64'])
)

# Display message after normalization
print("Normalization (Min-Max Scaling) applied to numerical columns")

#  Previewing the normalized data
print(df_normalized.head())




#  Save the cleaned dataset to a CSV file
df.to_csv('Cleaned_Data.csv', index=False)

df = pd.read_csv('Cleaned_Data.csv')
