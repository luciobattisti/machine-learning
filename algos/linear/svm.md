### **Support Vector Machine (SVM)**: A Detailed Description

Support Vector Machine (SVM) is a supervised learning algorithm widely used for classification and regression tasks. It works by finding the optimal boundary (hyperplane) that separates data points of different classes in a feature space.

---

### **Key Concepts**

1. **Hyperplane**:
   - A hyperplane is the decision boundary that separates different classes.
   - In 2D, it's a line; in 3D, it's a plane.

2. **Support Vectors**:
   - These are the data points closest to the hyperplane.
   - The SVM algorithm uses these points to determine the hyperplane.

3. **Margin**:
   - The margin is the distance between the hyperplane and the nearest data points from either class.
   - SVM aims to maximize this margin for better generalization.

4. **Kernel Trick**:
   - When data is not linearly separable in the current space, the kernel trick projects the data into a higher-dimensional space where it becomes linearly separable.
   - Common kernel functions include:
     - **Linear Kernel**: For linearly separable data.
     - **Polynomial Kernel**: Maps data into a higher-degree feature space.
     - **Radial Basis Function (RBF) Kernel**: Projects data into an infinite-dimensional space, handling non-linear separation.

---

### **Mathematical Formulation**

1. **Objective Function**:
   - Minimize the norm of the weight vector (w) to maximize the margin, while correctly classifying all points:
     ```
     Minimize ||w||^2
     Subject to: y_i * (w · x_i + b) >= 1 for all i
     ```
     Where:
     - `y_i` is the class label (+1 or -1),
     - `x_i` is the feature vector,
     - `w` is the weight vector,
     - `b` is the bias.



2. **Soft Margin SVM**:
   - Introduces a slack variable (ξ) to handle misclassified points:
     ```
     Minimize ||w||^2 + C * Σξ_i
     Subject to: y_i * (w · x_i + b) >= 1 - ξ_i
     ```
     - `C` is the regularization parameter balancing margin maximization and misclassification tolerance.

3. **Kernel Function**:
   - When using a kernel, the objective function becomes:
     ```
     Minimize Σα_i - 0.5 * ΣΣα_i * α_j * y_i * y_j * K(x_i, x_j)
     ```
     Where `K(x_i, x_j)` is the kernel function.
   - The first term of the equation maximizes the margin
   - The second term minimizes the overlap/misclassification

---

### **Scikit-learn Examples**

#### **Linear SVM for Binary Classification**

```python
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

# Example data
X = np.array([[1, 2], [2, 3], [3, 3], [6, 5], [7, 8], [8, 8]])
y = np.array([0, 0, 0, 1, 1, 1])  # Labels: 0 and 1

# Create and train SVM model
model = SVC(kernel='linear', C=1)
model.fit(X, y)

# Print details
print("Support Vectors:", model.support_vectors_)
print("Intercept (b):", model.intercept_)
print("Weights (w):", model.coef_)

# Plot decision boundary
xx = np.linspace(0, 10, 100)
yy = (-model.coef_[0][0] * xx - model.intercept_) / model.coef_[0][1]

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=50)
plt.plot(xx, yy, color='black', linestyle='--')
plt.title("Linear SVM Decision Boundary")
plt.show()
```

---

#### **Non-linear SVM Using RBF Kernel**

```python
from sklearn.datasets import make_moons
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

# Generate non-linear data
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

# Train SVM with RBF kernel
model = SVC(kernel='rbf', C=1, gamma=0.5)
model.fit(X, y)

# Visualize decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')
plt.title("Non-linear SVM with RBF Kernel")
plt.show()
```

---

### **When to Use SVM**

- Works well for small to medium-sized datasets with clear boundaries.
- Handles linear and non-linear problems effectively (with the kernel trick).
- May not scale well for very large datasets.
- Performance depends on choosing the right kernel and hyperparameters (`C`, `gamma`, etc.).

The **`gamma` parameter** in the scikit-learn implementation of Support Vector Machines (SVM) plays a crucial role in defining how the **Radial Basis Function (RBF)** kernel and other non-linear kernels behave. It controls the influence of individual training examples on the decision boundary.

---

### **Key Role of `gamma`**

1. **Gamma in RBF Kernel**:
   In the RBF kernel, the formula is:
   ```
   K(x_i, x_j) = exp(-gamma * ||x_i - x_j||^2)
   ```
   Where:
   - `x_i` and `x_j` are feature vectors of two data points.
   - `||x_i - x_j||^2` is the squared Euclidean distance.
   - `gamma` determines the "spread" of the kernel.

2. **Effect of `gamma`**:
   - **Low gamma (small value)**:
     - The kernel has a **wide spread**, meaning each training example influences a larger area.
     - Results in a **smoother decision boundary** but may underfit the data.
   - **High gamma (large value)**:
     - The kernel has a **narrow spread**, meaning each training example influences only nearby points.
     - Leads to a **more complex decision boundary** but may overfit the data.

---

### **Choosing the Right Gamma**

- A **low `gamma`** generalizes well but might miss patterns in the data.
- A **high `gamma`** captures more details but risks overfitting.
- **Default Value** in scikit-learn:
  - If `gamma='scale'` (default), it's computed as:
    ```
    1 / (n_features * X.var())
    ```
    This adjusts gamma based on the number of features and their variance.
  - If `gamma='auto'`, it's computed as:
    ```
    1 / n_features
    ```

---

### **Practical Example: Effect of Gamma**

```python
from sklearn.datasets import make_circles
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

# Generate non-linear data
X, y = make_circles(n_samples=100, factor=0.3, noise=0.1, random_state=42)

# Train SVM with different gamma values
gamma_values = [0.1, 1, 10]
models = [SVC(kernel='rbf', C=1, gamma=g).fit(X, y) for g in gamma_values]

# Plot decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

plt.figure(figsize=(15, 5))
for i, (model, gamma) in enumerate(zip(models, gamma_values)):
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.subplot(1, 3, i + 1)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')
    plt.title(f"Gamma = {gamma}")
plt.show()
```

---

### **Summary**

- `gamma` controls the **flexibility** of the decision boundary in SVMs with kernels.
- **Low gamma** → broader influence → smoother boundaries → lower risk of overfitting.
- **High gamma** → localized influence → complex boundaries → higher risk of overfitting.
- Proper tuning of `gamma` (e.g., using grid search or cross-validation) is critical for optimal model performance.