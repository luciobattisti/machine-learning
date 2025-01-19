"""
Basic Decision Tree Implementation

Tree Structure:

Each node contains:
feature_idx: Index of the feature used for splitting.
threshold: Value used for splitting.
left: Left subtree.
right: Right subtree.
If the node is a leaf, it contains label (majority class).
Entropy:

Used to measure the impurity of the node.
Formula: entropy = -Σ (p_i * log2(p_i)) where p_i is the proportion of samples of class i.
Information Gain:

Difference between the entropy of the parent node and the weighted average entropy of the child nodes.
Stopping Conditions:

Tree stops growing if:
Maximum depth is reached.
All samples in a node belong to the same class.
Prediction:

Each test sample is passed through the tree based on the feature and threshold at each node until it reaches a leaf.
"""

import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth):
        # Stop splitting if depth exceeds max depth or only one class remains
        if len(set(y)) == 1 or (self.max_depth and depth >= self.max_depth):
            return {"label": self._majority_class(y)}

        # Find the best split
        feature_idx, threshold = self._best_split(X, y)
        if feature_idx is None:
            return {"label": self._majority_class(y)}

        # Split the dataset
        left_idx = X[:, feature_idx] <= threshold
        right_idx = X[:, feature_idx] > threshold
        left_tree = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_tree = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return {
            "feature_idx": feature_idx,
            "threshold": threshold,
            "left": left_tree,
            "right": right_tree,
        }

    def _best_split(self, X, y):
        best_gain = -float("inf")
        best_split = None

        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature_idx, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_idx, threshold)

        return best_split if best_gain > 0 else (None, None)

    def _information_gain(self, X, y, feature_idx, threshold):
        parent_entropy = self._entropy(y)

        left_idx = X[:, feature_idx] <= threshold
        right_idx = X[:, feature_idx] > threshold
        if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
            return 0

        left_entropy = self._entropy(y[left_idx])
        right_entropy = self._entropy(y[right_idx])
        weighted_avg = (len(y[left_idx]) / len(y)) * left_entropy + (len(y[right_idx]) / len(y)) * right_entropy

        return parent_entropy - weighted_avg

    def _entropy(self, y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def _majority_class(self, y):
        return np.bincount(y).argmax()

    def _traverse_tree(self, x, node):
        if "label" in node:
            return node["label"]

        if x[node["feature_idx"]] <= node["threshold"]:
            return self._traverse_tree(x, node["left"])
        else:
            return self._traverse_tree(x, node["right"])

# Create a simple dataset
X = np.array([[2.7, 2.5],
              [1.3, 3.1],
              [3.1, 1.5],
              [1.5, 2.7],
              [3.2, 3.3],
              [1.0, 1.0]])

y = np.array([0, 1, 0, 1, 0, 1])  # Binary classification

# Train the decision tree
tree = DecisionTree(max_depth=3)
tree.fit(X, y)

# Make predictions
X_test = np.array([[3.0, 2.5], [1.2, 2.5]])
predictions = tree.predict(X_test)
print("Predictions:", predictions)
