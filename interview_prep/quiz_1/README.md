### **📌 Machine Learning Coding Challenge: Implement K-Means Clustering**
**Difficulty:** Medium  
**Time Limit:** 30 minutes  

---

### **📝 Problem Description**  
You need to **implement the K-Means clustering algorithm from scratch** without using `sklearn.cluster.KMeans`.  
Your implementation should:  
✅ Accept a dataset of 2D points (e.g., `[(2, 3), (5, 8), (1, 4)]`)  
✅ Randomly initialize cluster centroids  
✅ Assign each point to the nearest centroid  
✅ Recalculate centroids based on cluster assignments  
✅ Repeat the process until centroids no longer change (or max iterations reached)  

---

### **📌 Function Signature**
```python
def k_means(data: list[tuple], k: int, max_iters: int = 100) -> dict:
    """
    Implements the K-Means clustering algorithm.

    :param data: List of tuples representing 2D points (e.g., [(2,3), (5,8), (1,4)])
    :param k: Number of clusters
    :param max_iters: Maximum number of iterations
    :return: A dictionary where keys are cluster centroids and values are lists of points assigned to them
    """
```

---

### **📌 Example Input & Output**
#### **Input**
```python
data = [(2, 3), (5, 8), (1, 4), (6, 7), (3, 5), (8, 9)]
k = 2
clusters = k_means(data, k)
print(clusters)
```

#### **Output (Example)**
```python
{
    (2.5, 4.0): [(2, 3), (1, 4), (3, 5)],
    (6.5, 8.0): [(5, 8), (6, 7), (8, 9)]
}
```
📌 **Note:** The centroid values may vary depending on random initialization.  

---

### **📌 Constraints & Requirements**
- **Use only NumPy & basic Python (no sklearn)**
- **Implement Euclidean distance manually**
- **Use a stopping condition when centroids don't change**
- **Bonus:** Handle different values of `k` dynamically

---

### **⏳ Time: 30 minutes**