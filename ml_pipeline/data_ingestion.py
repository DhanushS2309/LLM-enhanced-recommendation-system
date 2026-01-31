"""
Data ingestion and cleaning for Online Retail II dataset.
Optimized for processing 540K+ transactions efficiently.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple
import os
import urllib.request


DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"
DATA_DIR = "data"
RAW_DATA_PATH = os.path.join(DATA_DIR, "online_retail_II.xlsx")


def download_dataset():
    """Download Online Retail II dataset from UCI repository."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if os.path.exists(RAW_DATA_PATH):
        print(f"✓ Dataset already exists: {RAW_DATA_PATH}")
        return
    
    print(f"Downloading dataset from {DATASET_URL}...")
    urllib.request.urlretrieve(DATASET_URL, RAW_DATA_PATH)
    print(f"✓ Dataset downloaded: {RAW_DATA_PATH}")


def load_and_clean_data() -> Tuple[pd.DataFrame, dict]:
    """
    Load and clean the Online Retail II dataset.
    
    Returns:
        Cleaned DataFrame and statistics dict
    """
    print("Loading dataset...")
    
    # Load both sheets from Excel file
    df1 = pd.read_excel(RAW_DATA_PATH, sheet_name='Year 2009-2010')
    df2 = pd.read_excel(RAW_DATA_PATH, sheet_name='Year 2010-2011')
    df = pd.concat([df1, df2], ignore_index=True)
    
    initial_count = len(df)
    print(f"Initial records: {initial_count:,}")
    
    # Data cleaning steps
    # 1. Remove cancelled orders (negative quantities)
    df = df[df['Quantity'] > 0]
    
    # 2. Remove invalid prices
    df = df[df['Price'] > 0]
    
    # 3. Remove missing customer IDs (we need them for recommendations)
    df = df[df['Customer ID'].notna()]
    
    # 4. Remove missing descriptions
    df = df[df['Description'].notna()]
    
    # 5. Convert Customer ID to string
    df['Customer ID'] = df['Customer ID'].astype(str).str.replace('.0', '', regex=False)
    
    # 6. Clean description text
    df['Description'] = df['Description'].str.strip().str.upper()
    
    # 7. Remove duplicates
    df = df.drop_duplicates()
    
    # 8. Add total amount column
    df['TotalAmount'] = df['Quantity'] * df['Price']
    
    # 9. Ensure InvoiceDate is datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    final_count = len(df)
    removed = initial_count - final_count
    
    stats = {
        'initial_records': initial_count,
        'final_records': final_count,
        'removed_records': removed,
        'unique_customers': df['Customer ID'].nunique(),
        'unique_products': df['StockCode'].nunique(),
        'date_range': (df['InvoiceDate'].min(), df['InvoiceDate'].max()),
        'total_revenue': df['TotalAmount'].sum()
    }
    
    print(f"\n{'='*60}")
    print(f"DATA CLEANING SUMMARY")
    print(f"{'='*60}")
    print(f"Initial records:     {stats['initial_records']:,}")
    print(f"Final records:       {stats['final_records']:,}")
    print(f"Removed:             {stats['removed_records']:,}")
    print(f"Unique customers:    {stats['unique_customers']:,}")
    print(f"Unique products:     {stats['unique_products']:,}")
    print(f"Date range:          {stats['date_range'][0]} to {stats['date_range'][1]}")
    print(f"Total revenue:       £{stats['total_revenue']:,.2f}")
    print(f"{'='*60}\n")
    
    return df, stats


def categorize_products(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorize products based on description keywords.
    Simple but effective categorization for recommendations.
    """
    def get_category(description: str) -> str:
        description = description.upper()
        
        # Define category keywords
        categories = {
            'HOME_DECOR': ['DECORATION', 'DECOR', 'ORNAMENT', 'VINTAGE', 'ANTIQUE'],
            'KITCHEN': ['KITCHEN', 'LUNCH', 'DINNER', 'PLATE', 'CUP', 'MUG', 'BOWL'],
            'GARDEN': ['GARDEN', 'PLANT', 'FLOWER', 'OUTDOOR'],
            'TOYS': ['TOY', 'GAME', 'PUZZLE', 'DOLL'],
            'STATIONERY': ['PAPER', 'CARD', 'NOTEBOOK', 'PEN', 'PENCIL'],
            'BAGS': ['BAG', 'POUCH', 'HOLDER'],
            'LIGHTING': ['LIGHT', 'LAMP', 'CANDLE'],
            'TEXTILE': ['FABRIC', 'CUSHION', 'TOWEL', 'BLANKET'],
            'PARTY': ['PARTY', 'BIRTHDAY', 'CELEBRATION', 'BUNTING'],
            'CHRISTMAS': ['CHRISTMAS', 'XMAS', 'SANTA'],
        }
        
        for category, keywords in categories.items():
            if any(keyword in description for keyword in keywords):
                return category
        
        return 'OTHER'
    
    df['Category'] = df['Description'].apply(get_category)
    
    print(f"✓ Products categorized into {df['Category'].nunique()} categories")
    print(f"Category distribution:")
    print(df['Category'].value_counts())
    
    return df


def save_cleaned_data(df: pd.DataFrame):
    """Save cleaned data to CSV for faster loading."""
    output_path = os.path.join(DATA_DIR, "cleaned_data.csv")
    df.to_csv(output_path, index=False)
    print(f"✓ Cleaned data saved: {output_path}")


def run_ingestion_pipeline():
    """Run complete data ingestion pipeline."""
    print("\n" + "="*60)
    print("STARTING DATA INGESTION PIPELINE")
    print("="*60 + "\n")
    
    # Download dataset
    download_dataset()
    
    # Load and clean
    df, stats = load_and_clean_data()
    
    # Categorize products
    df = categorize_products(df)
    
    # Save cleaned data
    save_cleaned_data(df)
    
    print(f"\n{'='*60}")
    print(f"✓ Processed {stats['final_records']:,} transactions")
    print(f"{'='*60}\n")
    
    return df, stats


if __name__ == "__main__":
    df, stats = run_ingestion_pipeline()
