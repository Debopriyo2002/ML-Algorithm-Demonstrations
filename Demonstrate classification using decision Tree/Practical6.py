import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
X = iris.data[:, :2]  # Considering only the first two features for visualization
y = iris.target

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Preprocessing (Optional step depending on the dataset)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fitting the Decision Tree classifier to the Training set
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Predicting the test set results
y_pred = decision_tree.predict(X_test)

# Calculating accuracy and creating a confusion matrix for test set
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Test Set Accuracy: {accuracy:.2f}")
print("Test Set Confusion Matrix:")
print(conf_matrix)

# Visualizing the test set result
plt.figure(figsize=(12, 8))

# Plotting decision boundaries for Test set
plt.subplot(1, 2, 1)
xx, yy = np.meshgrid(np.arange(X_test[:, 0].min() - 1, X_test[:, 0].max() + 1, 0.01),
                     np.arange(X_test[:, 1].min() - 1, X_test[:, 1].max() + 1, 0.01))
Z = decision_tree.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(('purple', 'green', 'yellow')))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=ListedColormap(('purple', 'green', 'yellow')))
plt.title('Decision Tree - Test Set')
plt.xlabel('Feature 1 (Sepal Length)')
plt.ylabel('Feature 2 (Sepal Width)')
plt.colorbar()

# Predicting and visualizing the training set result
y_pred_train = decision_tree.predict(X_train)

# Calculating accuracy and creating a confusion matrix for training set
accuracy_train = accuracy_score(y_train, y_pred_train)
conf_matrix_train = confusion_matrix(y_train, y_pred_train)

print(f"Training Set Accuracy: {accuracy_train:.2f}")
print("Training Set Confusion Matrix:")
print(conf_matrix_train)

# Plotting decision boundaries for Training set
plt.subplot(1, 2, 2)
Z_train = decision_tree.predict(np.c_[xx.ravel(), yy.ravel()])
Z_train = Z_train.reshape(xx.shape)
plt.contourf(xx, yy, Z_train, alpha=0.3, cmap=ListedColormap(('purple', 'green', 'yellow')))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=ListedColormap(('purple', 'green', 'yellow')))
plt.title('Decision Tree - Training Set')
plt.xlabel('Feature 1 (Sepal Length)')
plt.ylabel('Feature 2 (Sepal Width)')
plt.colorbar()

plt.tight_layout()
plt.show()
