Certainly! Let me provide an explanation and rewrite the cost functions using plain text.

---

### **Lasso Regression** (L1 Regularization)

- **Definition**: Lasso adds a penalty equivalent to the **sum of the absolute values of the coefficients** to the linear regression cost function.

- **Cost Function** (plain text):
  ```
  Minimize:
  Sum of squared errors (SSE) + λ * Sum of absolute values of coefficients
  ```

  Where:
  - `SSE` = ∑ (y_i - predicted_y_i)^2 (the residual sum of squares),
  - `λ` is the regularization parameter (controls the strength of the penalty).

- **Key Feature**: Lasso shrinks some coefficients to exactly zero, performing **feature selection** by eliminating less important features.

---

### **Ridge Regression** (L2 Regularization)

- **Definition**: Ridge adds a penalty equivalent to the **sum of the squared values of the coefficients** to the linear regression cost function.

- **Cost Function** (plain text):
  ```
  Minimize:
  Sum of squared errors (SSE) + λ * Sum of squares of coefficients
  ```

  Where:
  - `SSE` = ∑ (y_i - predicted_y_i)^2 (the residual sum of squares),
  - `λ` is the regularization parameter (controls the strength of the penalty).

- **Key Feature**: Ridge regression reduces the magnitude of coefficients but does not set any coefficients exactly to zero, meaning all features are retained.

---

### **Comparison**

| **Aspect**               | **Lasso**                          | **Ridge**                       |
|--------------------------|-------------------------------------|----------------------------------|
| **Penalty Term**         | Sum of absolute values of coefficients | Sum of squares of coefficients |
| **Feature Selection**    | Yes, shrinks some coefficients to zero | No, retains all coefficients    |
| **Use Case**             | Sparse models, when irrelevant features exist | Multicollinearity issues, all features important |

---

### **Scikit-Learn Example**

```python
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Example Data
X = np.random.rand(100, 5)  # 100 samples, 5 features
y = 3*X[:, 0] - 2*X[:, 1] + X[:, 2] + np.random.rand(100)  # Target variable with noise

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lasso Regression
lasso = Lasso(alpha=0.1)  # Set regularization strength
lasso.fit(X_train, y_train)
lasso_predictions = lasso.predict(X_test)

print("Lasso Regression Results")
print("Coefficients:", lasso.coef_)
print("Mean Squared Error:", mean_squared_error(y_test, lasso_predictions))

# Ridge Regression
ridge = Ridge(alpha=0.1)  # Set regularization strength
ridge.fit(X_train, y_train)
ridge_predictions = ridge.predict(X_test)

print("\nRidge Regression Results")
print("Coefficients:", ridge.coef_)
print("Mean Squared Error:", mean_squared_error(y_test, ridge_predictions))
```

---

### **Takeaways**
- Use **Lasso** when some features are irrelevant, and you need a sparse model.
- Use **Ridge** when multicollinearity is a concern and you want to retain all features.
- Combining Lasso (L1 regularization) and Ridge (L2 regularization) results in a technique called **Elastic Net Regression**.

---

### **Elastic Net Regression**

- **Definition**: Elastic Net adds both L1 and L2 penalties to the linear regression cost function, allowing it to combine the benefits of Lasso and Ridge regression.

- **Cost Function** (plain text):
  ```
  Minimize:
  Sum of squared errors (SSE) 
  + α * [(1 - r) * Sum of squares of coefficients (L2)] 
  + α * [r * Sum of absolute values of coefficients (L1)]
  ```

  Where:
  - `SSE` = ∑ (y_i - predicted_y_i)^2 (the residual sum of squares),
  - `α` controls the overall strength of regularization,
  - `r` controls the trade-off between L1 (Lasso) and L2 (Ridge):
    - `r = 1` → Elastic Net behaves like Lasso.
    - `r = 0` → Elastic Net behaves like Ridge.

- **Key Features**:
  - Retains feature selection capabilities (like Lasso).
  - Handles multicollinearity effectively (like Ridge).
  - Prevents over-penalization of correlated features, a limitation of Lasso.

---

### **Scikit-Learn Implementation**

Here’s an example of how to use **ElasticNet** in Scikit-Learn:

```python
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Example Data
X = np.random.rand(100, 5)  # 100 samples, 5 features
y = 3*X[:, 0] - 2*X[:, 1] + X[:, 2] + np.random.rand(100)  # Target variable with noise

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Elastic Net Regression
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)  # α = 0.1, r = 0.5 (equal L1 and L2 mix)
elastic_net.fit(X_train, y_train)
predictions = elastic_net.predict(X_test)

print("Elastic Net Regression Results")
print("Coefficients:", elastic_net.coef_)
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
```

---

### **Use Cases for Elastic Net**

1. **High-Dimensional Data**: Works well when the number of features is much larger than the number of observations (e.g., genomics, text data).
2. **Correlated Features**: Balances the penalties from Ridge and Lasso to prevent biased estimates.
3. **Sparse Models**: Offers feature selection while mitigating some limitations of Lasso with multicollinearity.

---

### **Comparison**

| **Aspect**              | **Lasso**     | **Ridge**       | **Elastic Net**           |
|-------------------------|---------------|-----------------|---------------------------|
| **Penalty**             | L1            | L2              | Combination of L1 and L2  |
| **Feature Selection**   | Yes           | No              | Yes                       |
| **Multicollinearity**   | Poor handling | Good handling   | Good handling             |
| **Use Case**            | Sparse models | Multicollinearity | High-dimensional, sparse models |

---

Elastic Net is a versatile regression technique that often provides better performance than Lasso or Ridge alone when faced with real-world data challenges.