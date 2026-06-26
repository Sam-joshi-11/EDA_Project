import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Understanding Dataset")

file_name = 'sales_data.csv'
if not os.path.exists(file_name):
    print(f"Error:{file_name} is not found")
    exit()

# load the dataset

df = pd.read_csv(file_name)
print("Successfully loaded")
print(f"Shape of the dataset:Rows:{df.shape[0]},columns:{df.shape[1]}")

print(f"\nHead of the data set\n {df.head()}")
print(f"\nTail of the data set\n {df.tail()}")
print(f"\nDescription of the data set\n {df.describe()}")

print("\nHandling Missing Values:")

print(df.isnull().sum())

# With using Median
median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)
print(median_age)

median_Spending = df['Spending'].median()
df['Spending'] = df['Spending'].fillna(median_Spending)
print(median_Spending)

# Using Mean

mean_Spending = df['Spending'].mean()
df['Spending'] = df['Spending'].fillna(mean_Spending)
print(mean_Spending)

mean_age = df['Age'].mean()
df['Age'] = df['Age'].fillna(mean_age)
print(mean_age)

# Distribution Analysis

plt.figure(figsize=(7,4))
df['Spending'].hist(bins=10,color='skyblue',edgecolor='black')
plt.title('Distribution of Spending')
plt.xlabel('Speding Amount')
plt.ylabel('Number of Customers')
plt.show()

# Correlation matrix

correlation = df.corr(numeric_only=True)
print(correlation)

print("Plotting Correlation Heatmap")
plt.figure(figsize=(7,4))
sns.heatmap(correlation,annot=True,cmap='coolwarm',fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Outlier Detection

plt.figure(figsize=(7,4))
sns.boxplot(x=df['Age'],color='lightgreen')
plt.title("Boxplot of Customer Age")
plt.xlabel("Age")
plt.show()

print("Find the Outliers of Age:")
outliers = df[df['Age']>100]
print("Found Outliers(s):")
print(outliers)
