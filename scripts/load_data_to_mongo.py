"""
Script to load processed data from ML pipeline into MongoDB.
Run after training pipeline completes.
"""
import asyncio
import pandas as pd
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.config import get_settings

settings = get_settings()


async def load_data_to_mongo():
    """Load all processed data into MongoDB."""
    print("\n" + "="*60)
    print("LOADING DATA TO MONGODB")
    print("="*60 + "\n")
    
    # Connect to MongoDB
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.DATABASE_NAME]
    
    # Load data files
    print("Loading data files...")
    
    try:
        df = pd.read_csv('data/cleaned_data.csv')
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        user_profiles = pd.read_csv('data/features/user_profiles.csv')
        products = pd.read_csv('data/features/products.csv')
        
        print(f"✓ Loaded {len(df):,} transactions")
        print(f"✓ Loaded {len(user_profiles):,} user profiles")
        print(f"✓ Loaded {len(products):,} products")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run ml_pipeline/train_pipeline.py first")
        return
    
    # Clear existing collections
    print("\nClearing existing collections...")
    await db.users.delete_many({})
    await db.products.delete_many({})
    await db.transactions.delete_many({})
    await db.user_profiles.delete_many({})
    print("✓ Collections cleared")
    
    # Insert users
    print("\nInserting users...")
    unique_customers = df['Customer ID'].unique()
    users_data = [
        {
            'customer_id': str(cid),
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        for cid in unique_customers
    ]
    result = await db.users.insert_many(users_data)
    user_id_map = {
        users_data[i]['customer_id']: result.inserted_ids[i]
        for i in range(len(users_data))
    }
    print(f"✓ Inserted {len(users_data):,} users")
    
    # Insert products
    print("Inserting products...")
    products_data = []
    for _, row in products.iterrows():
        products_data.append({
            'stock_code': str(row['stock_code']),
            'description': str(row['description']),
            'category': str(row['category']),
            'price': float(row['price']),
            'popularity_score': float(row['popularity_score']),
            'created_at': datetime.utcnow()
        })
    
    result = await db.products.insert_many(products_data)
    product_id_map = {
        products_data[i]['stock_code']: result.inserted_ids[i]
        for i in range(len(products_data))
    }
    print(f"✓ Inserted {len(products_data):,} products")
    
    # Insert transactions (in batches for performance)
    print("Inserting transactions...")
    batch_size = 1000
    total_inserted = 0
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        transactions_data = []
        
        for _, row in batch.iterrows():
            customer_id = str(row['Customer ID'])
            stock_code = str(row['StockCode'])
            
            if customer_id in user_id_map and stock_code in product_id_map:
                transactions_data.append({
                    'user_id': user_id_map[customer_id],
                    'product_id': product_id_map[stock_code],
                    'quantity': int(row['Quantity']),
                    'unit_price': float(row['Price']),
                    'invoice_date': row['InvoiceDate'],
                    'country': str(row['Country'])
                })
        
        if transactions_data:
            await db.transactions.insert_many(transactions_data)
            total_inserted += len(transactions_data)
        
        if (i + batch_size) % 10000 == 0:
            print(f"  Progress: {total_inserted:,} transactions inserted...")
    
    print(f"✓ Inserted {total_inserted:,} transactions")
    
    # Insert user profiles
    print("Inserting user profiles...")
    profiles_data = []
    
    for _, row in user_profiles.iterrows():
        customer_id = str(row['customer_id'])
        
        if customer_id in user_id_map:
            profiles_data.append({
                'user_id': user_id_map[customer_id],
                'total_spend': float(row['total_spend']),
                'avg_order_value': float(row['avg_order_value']),
                'purchase_frequency': float(row['purchase_frequency']),
                'top_categories': eval(row['top_categories']) if isinstance(row['top_categories'], str) else [],
                'brand_affinity': eval(row['brand_affinity']) if isinstance(row['brand_affinity'], str) else {},
                'price_sensitivity': str(row['price_sensitivity']),
                'updated_at': datetime.utcnow()
            })
    
    if profiles_data:
        await db.user_profiles.insert_many(profiles_data)
    print(f"✓ Inserted {len(profiles_data):,} user profiles")
    
    # Create indexes
    print("\nCreating indexes...")
    await db.transactions.create_index('user_id')
    await db.transactions.create_index('product_id')
    await db.transactions.create_index('invoice_date')
    await db.products.create_index('stock_code', unique=True)
    await db.products.create_index('category')
    await db.products.create_index('price')
    await db.user_profiles.create_index('user_id', unique=True)
    print("✓ Indexes created")
    
    # Summary
    print("\n" + "="*60)
    print("DATA LOADING COMPLETE")
    print("="*60)
    print(f"\nDatabase: {settings.DATABASE_NAME}")
    print(f"Users: {len(users_data):,}")
    print(f"Products: {len(products_data):,}")
    print(f"Transactions: {total_inserted:,}")
    print(f"User Profiles: {len(profiles_data):,}")
    print("\n" + "="*60 + "\n")
    
    client.close()


if __name__ == "__main__":
    asyncio.run(load_data_to_mongo())
