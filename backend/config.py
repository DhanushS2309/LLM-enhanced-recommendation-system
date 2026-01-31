"""
Configuration management for the recommendation system.
Optimized for performance and scalability.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # MongoDB Configuration
    MONGODB_URL: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "recommendation_system"
    
    # LLM Configuration (Groq)
    GROQ_API_KEY: str = ""
    LLM_MODEL: str = "llama-3.3-70b-versatile"  # Fast model for explanations
    LLM_MODEL_ADVANCED: str = "llama-3.3-70b-versatile"  # For complex cold-start reasoning
    LLM_MAX_TOKENS: int = 150
    LLM_TEMPERATURE: float = 0.7
    
    # Performance Constraints
    RECOMMENDATION_TIMEOUT_MS: int = 500
    SEARCH_TIMEOUT_MS: int = 2000
    
    # ML Configuration
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    TOP_K_RECOMMENDATIONS: int = 10
    
    # Hybrid Recommender Weights
    COLLABORATIVE_WEIGHT: float = 0.6
    CONTENT_BASED_WEIGHT: float = 0.4
    
    # Cold Start Configuration
    MIN_PURCHASES_FOR_CF: int = 5  # Minimum purchases to use collaborative filtering
    COLD_START_POPULAR_ITEMS: int = 20
    
    # Cache Configuration
    CACHE_TTL_SECONDS: int = 3600  # 1 hour
    MAX_CACHE_SIZE: int = 10000
    
    # Data Processing
    BATCH_SIZE: int = 1000
    MIN_TRANSACTION_AMOUNT: float = 0.01
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
