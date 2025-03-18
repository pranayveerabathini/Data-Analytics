

'Python EDA (aft pre-pr)'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'Your Path/Cleaned_Data.csv')


print(df.info())
print(df.head())
df.dtypes

# First Moment Business Decision / Measures of Central Tendency

# Mean

# Measures of Central Tendency

# 1. Mean
numeric_cols = df.select_dtypes(exclude=[object]).columns
mean = df[numeric_cols].mean()
print(mean)

#2. Mode
numeric_cols1 = df.select_dtypes(include=[int,float,object]).columns
mode = df[numeric_cols].mode().iloc[0]
print(mode)

# 3. Median

median = df[numeric_cols].median()
print(median)




# Second Moment Business Decision / Measures of Dispersion


# 1. Standard Deviation 
df_stddev = df[numeric_cols].std()
print(df_stddev)


# 2. Range (Max-Min)

Range = df[numeric_cols].max() - df[numeric_cols].min()
print(Range)


# 3. Variance 

Var = df[numeric_cols].var()
print(Var)


# Third Moment Business Decision / Skewness

Skewness = df[numeric_cols].skew()
print(Skewness)

# Fourth Moment Business Decision / Kurtosis

Kurtosis = df[numeric_cols].kurtosis()
print(Kurtosis)



# Graphical Representation
# Plot Histograms for Numeric Columns
for col in numeric_cols:
    plt.figure(figsize=(8, 5))
    plt.hist(df[col], bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


