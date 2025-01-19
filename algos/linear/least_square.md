The **Least Squares Method** is a fundamental technique in regression analysis used to find the line (or hyperplane) that best fits a set of data points by minimizing the **sum of squared differences** between observed values and the values predicted by the model.

---

### **Concept**

1. **Goal**: Minimize the **residual sum of squares (RSS)**, where:
   - Residuals are the differences between observed values (actual) and predicted values by the model.
   - RSS = sum of squared residuals = \( \sum (observed - predicted)^2 \).

2. **Objective**:
   - Find the best-fitting line by determining the optimal coefficients \( \beta_0, \beta_1, ..., \beta_n \) that minimize RSS.

3. **Result**:
   - The line (or hyperplane) represents the relationship between independent variables \( X \) and the dependent variable \( y \).

---

### **Steps**
1. Define the relationship \( y = X \cdot \beta + \epsilon \), where:
   - \( y \): Dependent variable.
   - \( X \): Matrix of independent variables.
   - \( \beta \): Coefficients to be estimated.
   - \( \epsilon \): Error term (residuals).

2. Use calculus to minimize RSS, solving for the coefficients \( \beta \).

---

### **Why Use the Least Squares Method?**
- It's computationally efficient and provides the "best linear unbiased estimate" under certain conditions (Gauss-Markov theorem).
- Works well for linear relationships and datasets with low multicollinearity.

---

### **Example Using `scikit-learn`**

Here’s how to apply least squares regression using Python’s `scikit-learn` library:

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Example data
X = np.array([[1], [2], [3], [4], [5]])  # Independent variable
y = np.array([2.1, 4.1, 6.0, 8.2, 10.1])  # Dependent variable

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Coefficients
intercept = model.intercept_
slope = model.coef_[0]

# Predictions
predictions = model.predict(X)

print(f"Intercept: {intercept}")
print(f"Slope: {slope}")
print(f"Predicted values: {predictions}")
```

---

### **Output**
- **Intercept**: The value of \( y \) when \( X = 0 \).
- **Slope**: The rate of change in \( y \) for a unit change in \( X \).
- **Predicted Values**: The model’s predictions for the dependent variable based on \( X \).

---

### **Example Explained**
- Input data \( X = [[1], [2], [3], [4], [5]] \), \( y = [2.1, 4.1, 6.0, 8.2, 10.1] \).
- The model calculates the slope and intercept to minimize RSS:
   - Slope \( \approx 2 \) (e.g., \( y \) increases by 2 units for each unit increase in \( X \)).
   - Intercept \( \approx 0.1 \) (e.g., the starting point of \( y \) when \( X = 0 \)).
- The line equation becomes \( y = 0.1 + 2 \cdot X \).

---

### **Benefits**
- Simple and interpretable.
- Forms the foundation for more advanced regression techniques.
