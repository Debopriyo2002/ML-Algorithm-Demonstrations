
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv'
data = pd.read_csv(url)


X = data['total_bill'].values.reshape(-1, 1)  
y = data['tip']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()


model.fit(X_train, y_train)


predictions = model.predict(X_test)


plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, predictions, color='red', linewidth=2)
plt.title('Simple Linear Regression')
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.show()
