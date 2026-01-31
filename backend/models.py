"""
Pydantic models for MongoDB documents.
Optimized for fast serialization and validation.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
from bson import ObjectId


class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema):
        field_schema.update(type="string")


class User(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    customer_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class Product(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    stock_code: str
    description: str
    category: str
    price: float
    popularity_score: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class Transaction(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    user_id: PyObjectId
    product_id: PyObjectId
    quantity: int
    unit_price: float
    invoice_date: datetime
    country: str

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class UserProfile(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    user_id: PyObjectId
    total_spend: float
    avg_order_value: float
    purchase_frequency: float  # purchases per month
    top_categories: List[str]
    brand_affinity: Dict[str, float]  # brand -> affinity score
    price_sensitivity: str  # "low", "medium", "high"
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class ProductEmbedding(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    product_id: PyObjectId
    embedding_vector: List[float]
    model_version: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# API Response Models
class RecommendationItem(BaseModel):
    product_id: str
    product_name: str
    price: float
    match_score: float
    explanation: str


class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[RecommendationItem]
    processing_time_ms: float


class SearchResult(BaseModel):
    product_id: str
    product_name: str
    category: str
    price: float
    relevance_score: float
    explanation: str


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    processing_time_ms: float


class UserInsight(BaseModel):
    user_id: str
    total_spend: float
    purchase_count: int
    top_categories: List[str]
    insight_text: str
