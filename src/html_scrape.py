import os
import re

import pandas as pd

# Define input and output directory paths
raw_dir = "../data/raw/"
patch_dir = "../data/patches/"
os.makedirs(patch_dir, exist_ok=True)

# Define metadata mappings and expected HTML filenames
targets = {
    "LEHMQ": {
        "file": "LEHMQ_daily.html",
        "Name": "Lehman Brothers",
        "Category": "Distressed_2008",
        "Sector": "Financials",
    },
    "BSC": {
        "file": "BSC_daily.html",
        "Name": "Bear Stearns",
        "Category": "Distressed_2008",
        "Sector": "Financials",
    },
    "SIVB": {
        "file": "SVB_daily.html",  # Maps to SVB_daily.html if named so locally
        "Name": "Silicon Valley Bank",
        "Category": "Distressed_2023",
        "Sector": "Financials",
    },
    "SBNY": {
        "file": "SBNY_daily.html",
        "Name": "Signature Bank",
        "Category": "Distressed_2023",
        "Sector": "Financials",
    },
    "PACW": {
        "file": "PACW_daily.html",
        "Name": "PacWest Bancorp",
        "Category": "Distressed_2023",
        "Sector": "Financials",
    },
    "CHK": {
        "file": "CHK_daily.html",
        "Name": "Chesapeake Energy",
        "Category": "Distressed_2020",
        "Sector": "Energy",
    },
}

print("--- EXTRACTING DATA FROM LOCAL HTML FILES ---")

# Regex to match {"d": <epoch>, "v": <price>} with optional quotes and arbitrary whitespace
pattern = re.compile(r'{\s*"d"\s*:\s*(\d+)\s*,\s*"v"\s*:\s*"?([\d.]+)"?\s*}')

for ticker, meta in targets.items():
    html_path = os.path.join(raw_dir, meta["file"])

    # Fallback check if filename uses SIVB instead of SVB
    if not os.path.exists(html_path) and ticker == "SIVB":
        alt_path = os.path.join(raw_dir, "SIVB_daily.html")
        if os.path.exists(alt_path):
            html_path = alt_path

    if not os.path.exists(html_path):
        print(f"❌ [File Not Found] {html_path}")
        continue

    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    matches = pattern.findall(content)

    if matches:
        records = [
            {
                "Date": pd.to_datetime(int(d_val), unit="s"),
                "Close": float(v_val),
                "Adj_Close": float(v_val),
            }
            for d_val, v_val in matches
        ]

        df_entity = pd.DataFrame(records)
        df_entity = df_entity.sort_values("Date").drop_duplicates(subset=["Date"])

        # Populate schema and metadata fields
        df_entity["Ticker"] = ticker
        df_entity["Name"] = meta["Name"]
        df_entity["Category"] = meta["Category"]
        df_entity["Sector"] = meta["Sector"]
        df_entity["Open"] = df_entity["Close"]
        df_entity["High"] = df_entity["Close"]
        df_entity["Low"] = df_entity["Close"]
        df_entity["Volume"] = 0

        # Enforce canonical column ordering
        df_entity = df_entity[
            [
                "Date",
                "Ticker",
                "Name",
                "Category",
                "Sector",
                "Open",
                "High",
                "Low",
                "Close",
                "Adj_Close",
                "Volume",
            ]
        ]

        output_csv = os.path.join(patch_dir, f"{ticker}.csv")
        df_entity.to_csv(output_csv, index=False)
        print(
            f"✅ [Parsed] {meta['Name']} ({ticker}): {len(df_entity):,} rows -> {output_csv}"
        )
    else:
        print(f"⚠️ [Warning] No regex pattern match in {meta['file']}")
