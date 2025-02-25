import pandas as pd


def analyze_sales_data(filepath: str) -> pd.DataFrame:
    """
    Processes a sales dataset to analyze revenue and top-selling products.

    :param filepath: Path to the CSV file
    :return: DataFrame with aggregated revenue per category
    """

    df = pd.read_csv(filepath)
    df["TotalSales"] = df["TotalSales"].fillna(df["Quantity"]*df["Price"])
    max_sale = df["TotalSales"].max()
    top_selling_product = df[df["TotalSales"]==max_sale]["Product"].values[0]
    print(f"Top-Selling Product: {top_selling_product} (${max_sale})")

    df_grouped_category = df.groupby("Category")["TotalSales"].sum().reset_index()

    return df_grouped_category.rename(columns={"TotalSales": "Revenue"})


if __name__ == "__main__":

    print(analyze_sales_data("data.csv"))