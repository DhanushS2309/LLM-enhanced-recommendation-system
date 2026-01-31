"""
Feature engineering: Build user profiles and product metadata.
Optimized for large-scale transaction processing.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
from typing import Dict, List


def build_user_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build comprehensive user profiles from transaction data.
    Critical for personalized recommendations.
    """
    print("\nBuilding user profiles...")
    
    user_profiles = []
    
    # Group by customer
    for customer_id, group in df.groupby('Customer ID'):
        # Calculate spending metrics
        total_spend = group['TotalAmount'].sum()
        purchase_count = len(group)
        avg_order_value = total_spend / purchase_count
        
        # Calculate purchase frequency (purchases per month)
        date_range = (group['InvoiceDate'].max() - group['InvoiceDate'].min()).days
        months = max(date_range / 30, 1)  # At least 1 month
        purchase_frequency = purchase_count / months
        
        # Top categories
        category_counts = group['Category'].value_counts()
        top_categories = category_counts.head(3).index.tolist()
        
        # Brand affinity (using StockCode prefix as proxy for brand)
        stock_codes = group['StockCode'].astype(str)
        brand_prefixes = stock_codes.str[:2].value_counts()
        brand_affinity = {
            brand: count / len(group) 
            for brand, count in brand_prefixes.head(5).items()
        }
        
        # Price sensitivity
        avg_price = group['Price'].mean()
        if avg_price < 2:
            price_sensitivity = "high"
        elif avg_price < 5:
            price_sensitivity = "medium"
        else:
            price_sensitivity = "low"
        
        user_profiles.append({
            'customer_id': customer_id,
            'total_spend': round(total_spend, 2),
            'purchase_count': purchase_count,
            'avg_order_value': round(avg_order_value, 2),
            'purchase_frequency': round(purchase_frequency, 2),
            'top_categories': top_categories,
            'brand_affinity': brand_affinity,
            'price_sensitivity': price_sensitivity,
            'first_purchase': group['InvoiceDate'].min(),
            'last_purchase': group['InvoiceDate'].max()
        })
    
    profiles_df = pd.DataFrame(user_profiles)
    
    print(f"✓ Created {len(profiles_df):,} user profiles")
    print(f"\nUser Profile Statistics:")
    print(f"  Avg spend per user: £{profiles_df['total_spend'].mean():.2f}")
    print(f"  Avg purchases per user: {profiles_df['purchase_count'].mean():.1f}")
    print(f"  Avg order value: £{profiles_df['avg_order_value'].mean():.2f}")
    
    return profiles_df


def build_product_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build product metadata including popularity scores.
    Essential for content-based filtering.
    """
    print("\nBuilding product metadata...")
    
    product_data = []
    
    # Group by product
    for stock_code, group in df.groupby('StockCode'):
        # Get most common description (in case of variations)
        description = group['Description'].mode()[0] if len(group['Description'].mode()) > 0 else group['Description'].iloc[0]
        
        # Category
        category = group['Category'].mode()[0] if len(group['Category'].mode()) > 0 else group['Category'].iloc[0]
        
        # Price (median to handle outliers)
        price = group['Price'].median()
        
        # Popularity score (based on purchase frequency and quantity)
        total_quantity = group['Quantity'].sum()
        unique_customers = group['Customer ID'].nunique()
        popularity_score = np.log1p(total_quantity) * np.log1p(unique_customers)
        
        product_data.append({
            'stock_code': stock_code,
            'description': description,
            'category': category,
            'price': round(price, 2),
            'popularity_score': round(popularity_score, 2),
            'total_sold': int(total_quantity),
            'unique_buyers': unique_customers
        })
    
    products_df = pd.DataFrame(product_data)
    
    # Normalize popularity scores to 0-100
    max_pop = products_df['popularity_score'].max()
    products_df['popularity_score'] = (products_df['popularity_score'] / max_pop * 100).round(2)
    
    print(f"✓ Created metadata for {len(products_df):,} products")
    print(f"\nProduct Metadata Statistics:")
    print(f"  Avg price: £{products_df['price'].mean():.2f}")
    print(f"  Avg popularity: {products_df['popularity_score'].mean():.1f}")
    print(f"  Top 5 products by popularity:")
    top_products = products_df.nlargest(5, 'popularity_score')[['description', 'popularity_score', 'total_sold']]
    print(top_products.to_string(index=False))
    
    return products_df


def create_user_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create user-item interaction matrix for collaborative filtering.
    Uses implicit feedback (purchase count).
    """
    print("\nCreating user-item interaction matrix...")
    
    # Aggregate purchases per user-product pair
    interactions = df.groupby(['Customer ID', 'StockCode']).agg({
        'Quantity': 'sum',
        'TotalAmount': 'sum'
    }).reset_index()
    
    # Create pivot table
    user_item_matrix = interactions.pivot_table(
        index='Customer ID',
        columns='StockCode',
        values='Quantity',
        fill_value=0
    )
    
    print(f"✓ Matrix shape: {user_item_matrix.shape[0]:,} users × {user_item_matrix.shape[1]:,} products")
    print(f"  Sparsity: {(1 - user_item_matrix.astype(bool).sum().sum() / (user_item_matrix.shape[0] * user_item_matrix.shape[1])) * 100:.2f}%")
    
    return user_item_matrix


def save_features(user_profiles: pd.DataFrame, products: pd.DataFrame, user_item_matrix: pd.DataFrame):
    """Save engineered features for ML pipeline."""
    import os
    
    os.makedirs('data/features', exist_ok=True)
    
    user_profiles.to_csv('data/features/user_profiles.csv', index=False)
    products.to_csv('data/features/products.csv', index=False)
    user_item_matrix.to_csv('data/features/user_item_matrix.csv')
    
    print("\n✓ Features saved to data/features/")


def run_feature_engineering(df: pd.DataFrame):
    """Run complete feature engineering pipeline."""
    print("\n" + "="*60)
    print("STARTING FEATURE ENGINEERING")
    print("="*60)
    
    # Build user profiles
    user_profiles = build_user_profiles(df)
    
    # Build product metadata
    products = build_product_metadata(df)
    
    # Create user-item matrix
    user_item_matrix = create_user_item_matrix(df)
    
    # Save features
    save_features(user_profiles, products, user_item_matrix)
    
    print("\n" + "="*60)
    print("✓ FEATURE ENGINEERING COMPLETE")
    print("="*60 + "\n")
    
    return user_profiles, products, user_item_matrix


if __name__ == "__main__":
    # Load cleaned data
    df = pd.read_csv('data/cleaned_data.csv')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Run feature engineering
    user_profiles, products, user_item_matrix = run_feature_engineering(df)
