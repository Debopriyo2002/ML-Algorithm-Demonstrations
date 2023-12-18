import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv'
tips = pd.read_csv(url)


print("First few rows of the dataset:")
print(tips.head())


plt.figure(figsize=(8, 6))
sns.countplot(x='day', data=tips, palette='viridis')
plt.title('Count of Observations by Day')
plt.xlabel('Day of the Week')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x='day', y='total_bill', data=tips, palette='coolwarm')
plt.title('Total Bill Amount by Day')
plt.xlabel('Day of the Week')
plt.ylabel('Total Bill Amount')
plt.show()
