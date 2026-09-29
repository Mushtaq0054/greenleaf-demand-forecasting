"""
GreenLeaf Grocery - Raw Sales Data Generator
Generates realistic 3-month daily sales data for fresh produce perishable goods.
"""

import os
import numpy as np
import pandas as pd

def generate_greenleaf_sales_data(
    start_date: str = "2026-06-01",
    end_date: str = "2026-08-31",
    random_seed: int = 42,
    output_path: str = "data/raw/sales_data.csv"
) -> pd.DataFrame:
    """
    Generates synthetic daily sales records for GreenLeaf Grocery fresh produce.
    Includes deliberate missing values in non-target fields to test imputation.
    """
    np.random.seed(random_seed)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    
    # 12 representative fresh produce SKUs
    products = [
        {"sku_id": "SKU_001", "product_name": "Organic Bananas", "category": "Fruit", "base_price": 1.99, "base_demand": 85},
        {"sku_id": "SKU_002", "product_name": "Baby Spinach", "category": "Leafy Greens", "base_price": 3.49, "base_demand": 45},
        {"sku_id": "SKU_003", "product_name": "Roma Tomatoes", "category": "Vegetables", "base_price": 2.29, "base_demand": 60},
        {"sku_id": "SKU_004", "product_name": "Hass Avocados", "category": "Fruit", "base_price": 4.99, "base_demand": 50},
        {"sku_id": "SKU_005", "product_name": "Organic Gala Apples", "category": "Fruit", "base_price": 3.99, "base_demand": 55},
        {"sku_id": "SKU_006", "product_name": "Fresh Strawberries", "category": "Berries", "base_price": 4.49, "base_demand": 40},
        {"sku_id": "SKU_007", "product_name": "Organic Broccoli", "category": "Vegetables", "base_price": 2.89, "base_demand": 35},
        {"sku_id": "SKU_008", "product_name": "Red Bell Peppers", "category": "Vegetables", "base_price": 3.19, "base_demand": 30},
        {"sku_id": "SKU_009", "product_name": "English Cucumbers", "category": "Vegetables", "base_price": 1.79, "base_demand": 40},
        {"sku_id": "SKU_010", "product_name": "Organic Carrots", "category": "Vegetables", "base_price": 2.19, "base_demand": 50},
        {"sku_id": "SKU_011", "product_name": "Fresh Blueberries", "category": "Berries", "base_price": 4.99, "base_demand": 30},
        {"sku_id": "SKU_012", "product_name": "Organic Tuscan Kale", "category": "Leafy Greens", "base_price": 2.99, "base_demand": 25},
    ]

    records = []
    for current_date in date_range:
        day_of_week = current_date.dayofweek
        is_weekend = 1 if day_of_week in [5, 6] else 0
        weekend_multiplier = 1.35 if is_weekend else 1.0

        for prod in products:
            # Price variation +/- 10%
            price_noise = np.random.uniform(-0.15, 0.15)
            unit_price = round(prod["base_price"] + price_noise, 2)
            
            # Promotion flag (15% chance)
            is_promo = np.random.choice(["Yes", "No"], p=[0.15, 0.85])
            promo_boost = 1.25 if is_promo == "Yes" else 1.0

            # Calculate units sold with realistic noise and seasonality
            expected_demand = prod["base_demand"] * weekend_multiplier * promo_boost
            units_sold = int(np.random.normal(loc=expected_demand, scale=max(5, expected_demand * 0.12)))
            units_sold = max(0, units_sold)

            # Inventory level
            inventory_level = int(units_sold + np.random.randint(10, 40))

            records.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "sku_id": prod["sku_id"],
                "product_name": prod["product_name"],
                "category": prod["category"],
                "unit_price": unit_price,
                "inventory_level": inventory_level,
                "promotion": is_promo,
                "is_weekend": is_weekend,
                "units_sold": units_sold
            })

    df = pd.DataFrame(records)

    # Introduce realistic missing values (~4% in numerical and categorical features) for imputation testing
    missing_indices_price = np.random.choice(df.index, size=int(len(df) * 0.04), replace=False)
    df.loc[missing_indices_price, "unit_price"] = np.nan

    missing_indices_inv = np.random.choice(df.index, size=int(len(df) * 0.03), replace=False)
    df.loc[missing_indices_inv, "inventory_level"] = np.nan

    missing_indices_promo = np.random.choice(df.index, size=int(len(df) * 0.03), replace=False)
    df.loc[missing_indices_promo, "promotion"] = np.nan

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created at: {output_path} ({len(df)} rows, {len(df.columns)} columns)")
    return df

if __name__ == "__main__":
    generate_greenleaf_sales_data()
