"""
FastAPI routes for cold-start handling.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict
from backend.services.cold_start_service import cold_start_service

router = APIRouter(prefix="/api/cold-start", tags=["cold-start"])


class ResponseSubmission(BaseModel):
    session_id: str
    question_index: int
    response: str


class RefinementRequest(BaseModel):
    session_id: str
    feedback: Dict  # {'liked': [...], 'disliked': [...]}


@router.post("/init")
async def initialize_cold_start(session_id: str):
    """
    Initialize cold-start session for new user.
    
    - **session_id**: Unique session identifier
    
    Returns initial questions to understand user preferences.
    """
    try:
        result = await cold_start_service.initialize_cold_start(session_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/respond")
async def submit_response(request: ResponseSubmission):
    """
    Submit response to cold-start question.
    
    - **session_id**: Session identifier
    - **question_index**: Index of current question
    - **response**: User's response
    
    Returns next question or final recommendations.
    """
    try:
        result = await cold_start_service.submit_response(
            session_id=request.session_id,
            question_index=request.question_index,
            response=request.response
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refine")
async def refine_recommendations(request: RefinementRequest):
    """
    Refine recommendations based on user feedback.
    
    - **session_id**: Session identifier
    - **feedback**: Dict with liked/disliked product IDs
    
    Returns refined recommendations.
    """
    try:
        result = await cold_start_service.refine_recommendations(
            session_id=request.session_id,
            feedback=request.feedback
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
