### Decision Tree Algorithm

The decision tree algorithm is a supervised learning algorithm used for classification and regression tasks. It works by recursively splitting the dataset into subsets based on feature values to maximize homogeneity within each subset.

---

### How It Works

1. **Root Node**: The tree starts at the root node, which represents the entire dataset. A feature is selected to split the dataset into two or more child nodes.
2. **Splitting**: At each node, the algorithm evaluates all possible splits for all features. It chooses the split that best separates the data based on a criterion.
3. **Stopping Condition**: The splitting stops when:
   - The maximum depth of the tree is reached.
   - A node contains fewer samples than a specified minimum.
   - All samples in a node belong to the same class (for classification).

4. **Leaf Nodes**: These are the terminal nodes of the tree and represent the decision or output (class or regression value).

---

### Splitting Criteria

#### **1. Classification**
The algorithm uses measures like **Gini Impurity** or **Entropy** to decide the best split.

- **Gini Impurity**: Measures the likelihood of a randomly chosen sample being incorrectly classified.
  Formula:
  ```
  Gini = 1 - Σ(p_i^2)
  ```
  where p_i is the proportion of samples of class i in the node.

- **Entropy**: Measures the level of impurity or disorder in the node.
  Formula:
  ```
  Entropy = -Σ(p_i * log2(p_i))
  ```

#### **2. Regression**
The decision tree minimizes the **Variance**:
  ```
  MSE = (1/N) * Σ(y_i - y_mean)^2
  ```
  where y_i is the actual value, y_mean is the mean of the values in the node, and N is the number of samples in the node.

---

### Strengths and Limitations

#### **Strengths**:
1. Easy to understand and interpret.
2. Handles both numerical and categorical data.
3. No need for feature scaling or normalization.

#### **Limitations**:
1. Prone to overfitting, especially with deep trees.
2. Sensitive to small changes in the data, leading to different trees (high variance).

---

### Example with Scikit-learn

#### **Classification**
```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Train a decision tree classifier
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X, y)

# Plot the tree
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.show()

# Predict
predictions = clf.predict(X[:5])
print("Predictions:", predictions)
```

#### **Regression**
```python
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np

# Generate a regression dataset
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Train a decision tree regressor
reg = DecisionTreeRegressor(criterion='squared_error', max_depth=3, random_state=42)
reg.fit(X, y)

# Predict and plot
X_grid = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
y_pred = reg.predict(X_grid)

plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_grid, y_pred, color='red', label='Decision Tree Prediction')
plt.legend()
plt.show()
```

---

### Key Parameters in Scikit-learn
1. **criterion**:
   - "gini" (default): For classification.
   - "entropy": For classification.
   - "squared_error": For regression.

2. **max_depth**: Maximum depth of the tree to control overfitting.

3. **min_samples_split**: Minimum number of samples required to split an internal node.

4. **min_samples_leaf**: Minimum number of samples required in a leaf node.

5. **random_state**: Ensures reproducibility.