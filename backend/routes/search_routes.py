"""
FastAPI routes for natural language search.
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from backend.services.search_service import search_service

router = APIRouter(prefix="/api/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str
    user_id: str = None
    top_k: int = 10


@router.post("/natural")
async def natural_language_search(request: SearchRequest):
    """
    Search products using natural language query.
    
    - **query**: Natural language search query
    - **user_id**: Optional user ID for personalization
    - **top_k**: Number of results (default: 10)
    
    Returns results with <2s latency.
    """
    try:
        result = await search_service.natural_language_search(
            query=request.query,
            user_id=request.user_id,
            top_k=request.top_k
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
