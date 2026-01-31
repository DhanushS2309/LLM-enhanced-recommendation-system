"""
MongoDB async database connection using Motor.
Optimized for high-throughput transaction processing.
"""
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from typing import Optional
from backend.config import get_settings

settings = get_settings()


class Database:
    client: Optional[AsyncIOMotorClient] = None
    db: Optional[AsyncIOMotorDatabase] = None


db = Database()


async def connect_to_mongo():
    """Connect to MongoDB with connection pooling."""
    db.client = AsyncIOMotorClient(
        settings.MONGODB_URL,
        maxPoolSize=50,  # Handle concurrent requests
        minPoolSize=10,
        maxIdleTimeMS=45000,
    )
    db.db = db.client[settings.DATABASE_NAME]
    
    # Create indexes for performance
    await create_indexes()
    print(f"✓ Connected to MongoDB: {settings.DATABASE_NAME}")


async def close_mongo_connection():
    """Close MongoDB connection."""
    if db.client:
        db.client.close()
        print("✓ Closed MongoDB connection")


async def create_indexes():
    """Create database indexes for optimal query performance."""
    # Transactions collection indexes
    await db.db.transactions.create_index("user_id")
    await db.db.transactions.create_index("product_id")
    await db.db.transactions.create_index("invoice_date")
    await db.db.transactions.create_index([("user_id", 1), ("invoice_date", -1)])
    
    # Products collection indexes
    await db.db.products.create_index("stock_code", unique=True)
    await db.db.products.create_index("category")
    await db.db.products.create_index("price")
    await db.db.products.create_index([("category", 1), ("price", 1)])
    
    # User profiles index
    await db.db.user_profiles.create_index("user_id", unique=True)
    
    # Product embeddings index
    await db.db.product_embeddings.create_index("product_id", unique=True)
    
    print("✓ Database indexes created")


def get_database() -> AsyncIOMotorDatabase:
    """Get database instance."""
    return db.db
