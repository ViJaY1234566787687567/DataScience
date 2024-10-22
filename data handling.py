import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Titanic-Dataset.csv')

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Basic statistics of the dataset
print("\nBasic statistics:")
print(df.describe(include='all'))  # include='all' to get stats for categorical columns

# Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Handling missing values (example: fill missing 'Age' with mean)
df['Age'] = df['Age'].fillna(df['Age'].mean())  # Updated this line

# Data visualization
sns.set(style="whitegrid")

# Create a histogram of 'Age'
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Create a correlation heatmap
plt.figure(figsize=(12, 8))
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include='number')
correlation_matrix = numeric_df.corr()  # Updated to use numeric_df
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
