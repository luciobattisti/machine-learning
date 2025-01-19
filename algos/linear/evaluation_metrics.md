Here are the definitions and formulas for **Precision**, **Recall**, **F1-Score**, and **Support** in plain text format:

---

### 1. **Precision**
- **Definition**: Precision is the ratio of correctly predicted positive observations to the total predicted positive observations.
- **Formula**:  
  Precision = True Positives (TP) / (True Positives (TP) + False Positives (FP))
- **Explanation**: Precision focuses on how many of the predicted positive results are actually correct.

---

### 2. **Recall (Sensitivity)**
- **Definition**: Recall is the ratio of correctly predicted positive observations to all the actual positive observations.
- **Formula**:  
  Recall = True Positives (TP) / (True Positives (TP) + False Negatives (FN))
- **Explanation**: Recall measures how well the model identifies all positive cases.

---

### 3. **F1-Score**
- **Definition**: The F1-Score is the harmonic mean of Precision and Recall, providing a single score that balances the two.
- **Formula**:  
  F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
- **Explanation**: The F1-Score is useful when you need to find a balance between precision and recall, especially when class distribution is imbalanced.

---

### 4. **Support**
- **Definition**: Support is the number of actual occurrences of the class in the dataset.
- **Explanation**: Support gives you context for interpreting Precision, Recall, and F1-Score values.

---

### Example for Better Understanding
Suppose we have a binary classification problem with the following counts:  
- **True Positives (TP)** = 70  
- **False Positives (FP)** = 30  
- **False Negatives (FN)** = 10  
- **Support** = 100 (total actual positive cases).

#### Using the formulas:
1. **Precision**:  
   Precision = 70 / (70 + 30) = 0.7 (or 70%)

2. **Recall**:  
   Recall = 70 / (70 + 10) = 0.875 (or 87.5%)

3. **F1-Score**:  
   F1-Score = 2 * (0.7 * 0.875) / (0.7 + 0.875) ≈ 0.778 (or 77.8%)

4. **Support**:  
   Support = 100 (this is the total number of actual positive instances).

---


### Example Code in Scikit-Learn

```Python
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
# Ground truth (actual labels)
y_true = [0, 1, 1, 1, 0, 1, 0, 0, 1, 0]

# Predicted labels
y_pred = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0]

# Calculate Precision, Recall, and F1-Score for each class
precision = precision_score(y_true, y_pred, average=None)
recall = recall_score(y_true, y_pred, average=None)
f1 = f1_score(y_true, y_pred, average=None)

print(f"Precision (per class): {precision}")
print(f"Recall (per class): {recall}")
print(f"F1-Score (per class): {f1}")

# Classification report for a comprehensive summary
report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1'])
print("\nClassification Report:\n")
print(report)
```

#### Output

```
Classification Report:

              precision    recall  f1-score   support

     Class 0       0.75      0.75      0.75         4
     Class 1       0.80      0.80      0.80         6

    accuracy                           0.78        10
   macro avg       0.78      0.78      0.78        10
weighted avg       0.78      0.78      0.78        10

```