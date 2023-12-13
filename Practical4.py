# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv'
data = pd.read_csv(url)

# Selecting features and target variable
X = data[['total_bill', 'sex', 'smoker', 'day', 'time']]
y = data['tip']

# Label encoding categorical variables
label_encoder = LabelEncoder()
X['sex'] = label_encoder.fit_transform(X['sex'])
X['smoker'] = label_encoder.fit_transform(X['smoker'])
X['day'] = label_encoder.fit_transform(X['day'])
X['time'] = label_encoder.fit_transform(X['time'])

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a KNN Classifier
knn = KNeighborsClassifier(n_neighbors=3)  # You can adjust n_neighbors as needed

# Training the classifier
knn.fit(X_train, y_train)

# Making predictions
predictions = knn.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")
