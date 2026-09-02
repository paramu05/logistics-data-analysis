# Week 2 Task: Data Collection, Cleaning, and Preprocessing for Logistics Analysis
# Technology: Python, Pandas, NumPy
#
# Dataset:
# Save your selected logistics/supply-chain CSV file in the same folder
# as this Python file and name it: logistics_data.csv

import pandas as pd
import numpy as np


# ============================================================
# 1. LOAD THE DATASET
# ============================================================

df = pd.read_csv("logistics_data.csv")

print("\n========== FIRST FIVE ROWS ==========")
print(df.head())


# ============================================================
# 2. INITIAL DATA INSPECTION
# ============================================================

print("\n========== DATASET SHAPE ==========")
print("Rows and Columns:", df.shape)

print("\n========== DATA INFORMATION ==========")
df.info()

print("\n========== DESCRIPTIVE STATISTICS ==========")
print(df.describe())


# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================

print("\n========== MISSING VALUES BEFORE CLEANING ==========")
print(df.isnull().sum())


# ============================================================
# 4. HANDLE MISSING NUMERICAL VALUES
# ============================================================
# Delivery Days is a numerical column.
# Median is used because it is less affected by extreme values.

if "Delivery Days" in df.columns:
    df["Delivery Days"] = df["Delivery Days"].fillna(
        df["Delivery Days"].median()
    )


# ============================================================
# 5. HANDLE MISSING CATEGORICAL VALUES
# ============================================================
# Shipping Mode is a categorical column.
# Mode represents the most frequently occurring value.

if "Shipping Mode" in df.columns:
    mode_value = df["Shipping Mode"].mode()

    if not mode_value.empty:
        df["Shipping Mode"] = df["Shipping Mode"].fillna(
            mode_value[0]
        )


# ============================================================
# 6. REMOVE DUPLICATE RECORDS
# ============================================================

print("\n========== DUPLICATES ==========")
print("Duplicate records before removal:", df.duplicated().sum())

df = df.drop_duplicates()

print("Duplicate records after removal:", df.duplicated().sum())


# ============================================================
# 7. CONVERT DATE COLUMNS
# ============================================================
# Converting text dates to datetime allows date calculations.

if "Order Date" in df.columns:
    df["Order Date"] = pd.to_datetime(
        df["Order Date"], errors="coerce"
    )

if "Shipping Date" in df.columns:
    df["Shipping Date"] = pd.to_datetime(
        df["Shipping Date"], errors="coerce"
    )


# ============================================================
# 8. STANDARDIZE CATEGORICAL VALUES
# ============================================================
# Removes unnecessary spaces and standardizes capitalization.

if "Shipping Mode" in df.columns:
    df["Shipping Mode"] = (
        df["Shipping Mode"]
        .astype(str)
        .str.strip()
        .str.title()
    )


# ============================================================
# 9. DETECT OUTLIERS USING IQR
# ============================================================

if "Delivery Days" in df.columns:

    Q1 = df["Delivery Days"].quantile(0.25)
    Q3 = df["Delivery Days"].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    print("\n========== OUTLIER ANALYSIS ==========")
    print("Q1:", Q1)
    print("Q3:", Q3)
    print("IQR:", IQR)
    print("Lower Limit:", lower_limit)
    print("Upper Limit:", upper_limit)

    outliers = df[
        (df["Delivery Days"] < lower_limit)
        | (df["Delivery Days"] > upper_limit)
    ]

    print("\nPotential outliers:")
    print(outliers)


    # ========================================================
    # 10. HANDLE OUTLIERS
    # ========================================================
    # In a real logistics project, an outlier should be
    # investigated before removing it. Here, we demonstrate
    # removal of values outside the selected IQR limits.

    df = df[
        (df["Delivery Days"] >= lower_limit)
        & (df["Delivery Days"] <= upper_limit)
    ]


# ============================================================
# 11. MIN-MAX NORMALIZATION
# ============================================================
# Converts Sales approximately to a range between 0 and 1.

if "Sales" in df.columns:

    sales_min = df["Sales"].min()
    sales_max = df["Sales"].max()

    if sales_max != sales_min:
        df["Sales_Normalized"] = (
            (df["Sales"] - sales_min)
            / (sales_max - sales_min)
        )
    else:
        df["Sales_Normalized"] = 0


# ============================================================
# 12. FINAL DATA VALIDATION
# ============================================================

print("\n========== FINAL DATASET ==========")

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDuplicate records after cleaning:")
print(df.duplicated().sum())

print("\nFinal dataset information:")
df.info()

print("\nFinal descriptive statistics:")
print(df.describe())

print("\nFirst five rows of cleaned data:")
print(df.head())


# ============================================================
# 13. SAVE CLEANED DATASET
# ============================================================
# The cleaned dataset is saved for future logistics analysis.

df.to_csv(
    "cleaned_logistics_data.csv",
    index=False
)

print("\n==========================================")
print("Preprocessing completed successfully!")
print("Cleaned dataset saved as:")
print("cleaned_logistics_data.csv")
print("==========================================")
