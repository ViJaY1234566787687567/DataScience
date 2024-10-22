import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (replace 'your_dataset.csv' with your dataset file)
df = pd.read_csv('Titanic-Dataset.csv')

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Basic statistics of the dataset
print("\nBasic statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Data visualization
# Set the style of seaborn
sns.set(style="whitegrid")

# Create a histogram of a specific column (replace 'column_name' with an actual column)
plt.figure(figsize=(10, 6))
sns.histplot(df['column_name'], bins=30, kde=True)
plt.title('Distribution of Column Name')
plt.xlabel('Column Name')
plt.ylabel('Frequency')
plt.show()

# Create a correlation heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()