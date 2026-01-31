"""
Recommendation Service - Core business logic for generating recommendations.
Optimized for <500ms response time with caching and efficient model loading.
"""
import time
from typing import List, Dict, Optional
from bson import ObjectId
from backend.database import get_database
from backend.cache import (
    get_cached_recommendations, cache_recommendations,
    get_cached_user_profile, cache_user_profile
)
from backend.services.llm_service import llm_service
from ml_pipeline.hybrid_engine import HybridRecommender
import os


class RecommendationService:
    """
    High-performance recommendation service with caching and cold-start handling.
    """
    
    def __init__(self):
        self.recommender: Optional[HybridRecommender] = None
        self._load_recommender()
    
    def _load_recommender(self):
        """Load hybrid recommender models (lazy loading)."""
        if os.path.exists('models/collaborative_filter.pkl'):
            self.recommender = HybridRecommender()
            self.recommender.load_models()
            print("✓ Recommendation models loaded")
        else:
            print("⚠ Warning: Models not found. Run ml_pipeline/train_pipeline.py first")
    
    async def get_recommendations(self, user_id: str, top_k: int = 10,
                                 include_explanations: bool = True) -> Dict:
        """
        Get personalized recommendations for a user.
        
        Args:
            user_id: Customer ID
            top_k: Number of recommendations
            include_explanations: Whether to generate LLM explanations
            
        Returns:
            Dict with recommendations and metadata
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = f"{user_id}_{top_k}_{include_explanations}"
        cached = get_cached_recommendations(user_id)
        if cached and len(cached) >= top_k:
            return {
                'user_id': user_id,
                'recommendations': cached[:top_k],
                'processing_time_ms': 0,  # Cached
                'cached': True
            }
        
        # Get recommendations from hybrid engine
        if not self.recommender:
            return self._cold_start_fallback(user_id, top_k)
        
        recommendations = self.recommender.recommend(user_id, top_k=top_k)
        
        # Get user profile for explanations
        user_profile = await self._get_user_profile(user_id)
        
        # Generate LLM explanations if requested
        if include_explanations and recommendations:
            recommendations = await self._add_explanations(
                recommendations, user_profile
            )
        
        # Cache results
        cache_recommendations(user_id, recommendations)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return {
            'user_id': user_id,
            'recommendations': recommendations,
            'processing_time_ms': round(processing_time_ms, 2),
            'cached': False,
            'strategy': recommendations[0].get('method', 'unknown') if recommendations else 'none'
        }
    
    async def _get_user_profile(self, user_id: str) -> Dict:
        """Get or build user profile."""
        # Check cache
        cached_profile = get_cached_user_profile(user_id)
        if cached_profile:
            return cached_profile
        
        # Get from database
        db = get_database()
        profile = await db.user_profiles.find_one({'user_id': ObjectId(user_id)})
        
        if profile:
            profile_dict = {
                'user_id': str(profile['user_id']),
                'total_spend': profile.get('total_spend', 0),
                'purchase_count': profile.get('purchase_frequency', 0) * 12,  # Approximate
                'top_categories': profile.get('top_categories', []),
                'avg_price': profile.get('avg_order_value', 0),
                'price_sensitivity': profile.get('price_sensitivity', 'medium')
            }
            
            # Cache it
            cache_user_profile(user_id, profile_dict)
            return profile_dict
        
        # Return empty profile for new users
        return {
            'user_id': user_id,
            'total_spend': 0,
            'purchase_count': 0,
            'top_categories': [],
            'avg_price': 0,
            'price_sensitivity': 'unknown'
        }
    
    async def _add_explanations(self, recommendations: List[Dict], 
                               user_profile: Dict) -> List[Dict]:
        """Add LLM-generated explanations to recommendations."""
        # Use batch explanation for efficiency
        try:
            explanations = await llm_service.explain_recommendations_batch(
                recommendations, user_profile
            )
            
            for rec in recommendations:
                product_id = rec.get('product_id')
                rec['explanation'] = explanations.get(
                    product_id,
                    "Recommended based on your shopping preferences."
                )
        except Exception as e:
            print(f"Error generating explanations: {e}")
            # Fallback to generic explanations
            for rec in recommendations:
                rec['explanation'] = "Recommended based on your shopping preferences."
        
        return recommendations
    
    def _cold_start_fallback(self, user_id: str, top_k: int) -> Dict:
        """Fallback for when models aren't loaded."""
        return {
            'user_id': user_id,
            'recommendations': [],
            'processing_time_ms': 0,
            'cached': False,
            'error': 'Models not loaded. Please train models first.'
        }
    
    async def get_user_insight(self, user_id: str) -> Dict:
        """
        Generate LLM-based insight about user's shopping behavior.
        
        Args:
            user_id: Customer ID
            
        Returns:
            Dict with insight text and user stats
        """
        user_profile = await self._get_user_profile(user_id)
        
        if user_profile['purchase_count'] == 0:
            return {
                'user_id': user_id,
                'insight': "New customer - no purchase history yet. Great opportunity to explore our catalog!",
                'is_new_user': True
            }
        
        # Generate LLM insight
        insight_text = await llm_service.generate_user_insight(user_profile)
        
        return {
            'user_id': user_id,
            'insight': insight_text,
            'total_spend': user_profile['total_spend'],
            'purchase_count': user_profile['purchase_count'],
            'top_categories': user_profile['top_categories'],
            'is_new_user': False
        }


# Global service instance
recommendation_service = RecommendationService()
