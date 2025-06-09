from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd

# Load the Breast Cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluate on test data
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=data.target_names))

# Visualize the tree (truncated for simplicity)
plt.figure(figsize=(16, 8))
plot_tree(clf, filled=True, feature_names=data.feature_names, class_names=data.target_names, max_depth=3)
plt.title("Decision Tree (Truncated at Depth 3)")
plt.show()

# Classify a new sample (using first row of test set as example)
new_sample = X_test.iloc[0].values.reshape(1, -1)
prediction = clf.predict(new_sample)
print("\nNew Sample Prediction:")
print("Features:\n", X_test.iloc[0])
print("Predicted Class:", data.target_names[prediction[0]])