"""
FastAPI main application.
Production-ready recommendation system backend.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.database import connect_to_mongo, close_mongo_connection
from backend.config import get_settings
from backend.middleware.performance_monitor import PerformanceMonitorMiddleware
from backend.routes import recommendation_routes, search_routes, cold_start_routes

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("\n" + "="*60)
    print("STARTING RECOMMENDATION SYSTEM API")
    print("="*60 + "\n")
    
    await connect_to_mongo()
    
    print("\n✓ API Ready")
    print(f"  Docs: http://localhost:8000/docs")
    print(f"  Performance targets:")
    print(f"    - Recommendations: <500ms")
    print(f"    - Search: <2000ms")
    print("\n" + "="*60 + "\n")
    
    yield
    
    # Shutdown
    await close_mongo_connection()
    print("\n✓ API Shutdown Complete\n")


# Create FastAPI app
app = FastAPI(
    title="LLM-Enhanced Recommendation System",
    description="Production-ready recommendation engine with hybrid ML and LLM explanations",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add performance monitoring
app.add_middleware(PerformanceMonitorMiddleware)

# Include routers
app.include_router(recommendation_routes.router)
app.include_router(search_routes.router)
app.include_router(cold_start_routes.router)


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "LLM-Enhanced Recommendation System",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "recommendations": "/api/recommendations/{user_id}",
            "user_insight": "/api/recommendations/{user_id}/insight",
            "search": "/api/search/natural",
            "cold_start_init": "/api/cold-start/init",
            "cold_start_respond": "/api/cold-start/respond",
            "cold_start_refine": "/api/cold-start/refine"
        },
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
