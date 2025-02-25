## **📌 NumPy Coding Challenge: Data Processing & Transformation**
### **📝 Problem Description**  
You are given a **NumPy 2D array** representing a dataset where:  
✅ Each **row** is a data sample  
✅ Each **column** is a feature  

Your task is to implement a function that:  
1️⃣ **Normalizes the dataset** (scaling values between 0 and 1).  
2️⃣ **Finds and replaces missing values (NaN)** with the **column mean**.  
3️⃣ **Computes the mean & standard deviation** for each column.  
4️⃣ **Returns the transformed dataset** and computed statistics.  

---

### **📌 Function Signature**
```python
import numpy as np

def process_numpy_data(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Processes a NumPy dataset by normalizing, handling missing values, and computing statistics.

    :param data: NumPy 2D array representing the dataset.
    :return: (processed_data, column_means, column_stddevs)
    """
```

---

### **📌 Example Input**
```python
data = np.array([
    [10,  200,  np.nan],
    [15,  np.nan,  50],
    [20,  180,  60],
    [25,  220,  70]
])
```

---

### **📌 Expected Output**
```python
Processed Data:
[[0.   0.6667 0.   ]
 [0.3333 0.    0.5  ]
 [0.6667 0.4  0.6667]
 [1.   1.   1.   ]]

Column Means: [17.5 200 60]
Column Std Devs: [5.59 18.26 8.16]
```
📌 **Note**: The normalization formula is:  
\[
X' = \frac{X - X_{\min}}{X_{\max} - X_{\min}}
\]

---

### **📌 Constraints & Requirements**
✅ Use **NumPy** (no Pandas or Sklearn).  
✅ Use **vectorized operations** (avoid loops).  
✅ Handle **NaN values** dynamically.  
✅ Normalize each column to **[0,1] range**.  

---

### **⏳ Time: 30 minutes**  
Write your code in the **canvas** and let me know when you're done! 🚀