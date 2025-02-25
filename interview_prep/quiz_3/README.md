## **📌 Pandas Coding Challenge: Sales Data Analysis**  
**Difficulty:** Easy-Medium  
**Time Limit:** 30 minutes  

---

### **📝 Problem Description**  
You are given a **CSV file** (`sales_data.csv`) containing sales information. Your task is to **analyze and process the data** using **Pandas**.

The dataset contains the following columns:  
✅ `OrderID` (int) - Unique order identifier  
✅ `Product` (string) - Name of the product  
✅ `Category` (string) - Product category  
✅ `Quantity` (int) - Number of units sold  
✅ `Price` (float) - Price per unit  
✅ `TotalSales` (float) - Total sales amount (`Quantity * Price`)  

Your task is to implement a function that:  
1️⃣ **Loads the dataset** into a Pandas DataFrame.  
2️⃣ **Fills missing values** (`NaN`) in the `TotalSales` column using `Quantity * Price`.  
3️⃣ **Finds the top-selling product** (highest total sales).  
4️⃣ **Computes total revenue per category** (sum of `TotalSales` per `Category`).  
5️⃣ **Returns a summary DataFrame** with the aggregated results.  

---

### **📌 Function Signature**
```python
import pandas as pd

def analyze_sales_data(filepath: str) -> pd.DataFrame:
    """
    Processes a sales dataset to analyze revenue and top-selling products.

    :param filepath: Path to the CSV file
    :return: DataFrame with aggregated revenue per category
    """
```

---

### **📌 Example Input**
CSV file (`sales_data.csv`):
```plaintext
OrderID,Product,Category,Quantity,Price,TotalSales
101,Phone,Electronics,3,500.0,
102,Laptop,Electronics,2,1200.0,
103,Headphones,Accessories,5,100.0,500.0
104,TV,Electronics,1,800.0,
105,Mouse,Accessories,4,50.0,
```

---

### **📌 Expected Output**
```plaintext
Top-Selling Product: Laptop ($2400.0)
Revenue per Category:
    Category       Revenue
0  Accessories    700.0
1  Electronics    4900.0
```

📌 **Notes:**  
- Missing `TotalSales` values should be filled using `Quantity * Price`.  
- The **top-selling product** is determined by the **highest total sales value**.  
- The output DataFrame should contain **total revenue per category**.  

---

### **📌 Constraints & Requirements**
✅ Use **Pandas only** (no NumPy required).  
✅ Use **vectorized operations** (avoid loops where possible).  
✅ Ensure missing values in `TotalSales` are correctly calculated.  
✅ Use **groupby()** to compute revenue per category.  

---

### **⏳ Time: 30 minutes**