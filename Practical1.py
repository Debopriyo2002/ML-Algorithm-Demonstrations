
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv'
data = pd.read_csv(url)


print("First few rows of the dataset:")
print(data.head())


print("\nSummary of the dataset:")
print(data.info())

print("\nSummary statistics of the numerical columns:")
print(data.describe())


print("\nMissing values in the dataset:")
print(data.isnull().sum())


plt.figure(figsize=(8, 6))
sns.histplot(data['total_bill'], kde=True, bins=30, color='skyblue')
plt.title('Distribution of Total Bill')
plt.xlabel('Total Bill Amount')

plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(8, 6))
sns.scatterplot(x='total_bill', y='tip', data=data, color='salmon')
plt.title('Scatter plot of Total Bill vs. Tip')
plt.xlabel('Total Bill Amount')
plt.ylabel('Tip Amount')
plt.show()



# ValueError: could not convert string to float: 'Female'

# correlation_matrix = data.corr()
# plt.figure(figsize=(8, 6))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix')
# plt.show()