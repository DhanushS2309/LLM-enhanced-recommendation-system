"""
FastAPI routes for recommendation endpoints.
"""
from fastapi import APIRouter, HTTPException, Query
from backend.services.recommendation_service import recommendation_service
from backend.models import RecommendationResponse

router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])


@router.get("/{user_id}")
async def get_recommendations(
    user_id: str,
    top_k: int = Query(10, ge=1, le=50),
    include_explanations: bool = Query(True)
):
    """
    Get personalized recommendations for a user.
    
    - **user_id**: Customer ID
    - **top_k**: Number of recommendations (1-50)
    - **include_explanations**: Include LLM explanations
    
    Returns recommendations with <500ms latency.
    """
    try:
        result = await recommendation_service.get_recommendations(
            user_id=user_id,
            top_k=top_k,
            include_explanations=include_explanations
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}/insight")
async def get_user_insight(user_id: str):
    """
    Get LLM-generated insight about user's shopping behavior.
    
    - **user_id**: Customer ID
    
    Returns user profile insights and shopping patterns.
    """
    try:
        result = await recommendation_service.get_user_insight(user_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
