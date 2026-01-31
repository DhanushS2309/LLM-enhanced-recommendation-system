"""
Performance monitoring middleware.
Tracks request latency and ensures performance constraints are met.
"""
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class PerformanceMonitorMiddleware(BaseHTTPMiddleware):
    """
    Middleware to monitor API performance.
    Logs warnings if endpoints exceed their latency targets.
    """
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Add latency header
        response.headers["X-Processing-Time-Ms"] = str(round(latency_ms, 2))
        
        # Check performance constraints
        path = request.url.path
        
        if "/recommendations/" in path and latency_ms > 500:
            print(f"⚠ PERFORMANCE WARNING: Recommendation latency {latency_ms:.2f}ms > 500ms target")
        
        if "/search/natural" in path and latency_ms > 2000:
            print(f"⚠ PERFORMANCE WARNING: Search latency {latency_ms:.2f}ms > 2000ms target")
        
        # Log request
        print(f"{request.method} {path} - {latency_ms:.2f}ms")
        
        return response
