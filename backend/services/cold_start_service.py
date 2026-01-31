"""
Cold Start Service - Handle new users with no purchase history.
Uses LLM reasoning and iterative refinement.
"""
from typing import List, Dict, Optional
from backend.database import get_database
from backend.services.llm_service import llm_service


class ColdStartService:
    """
    Service for handling cold-start scenarios with new users.
    """
    
    def __init__(self):
        self.sessions = {}  # In-memory session storage (use Redis in production)
    
    async def initialize_cold_start(self, session_id: str) -> Dict:
        """
        Initialize cold-start session for a new user.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Dict with initial questions
        """
        # Generate questions using LLM
        questions = await llm_service.generate_cold_start_questions()
        
        # Initialize session
        self.sessions[session_id] = {
            'questions': questions,
            'responses': {},
            'current_question_index': 0
        }
        
        return {
            'session_id': session_id,
            'questions': questions,
            'current_question': questions[0] if questions else None
        }
    
    async def submit_response(self, session_id: str, question_index: int,
                             response: str) -> Dict:
        """
        Submit response to a cold-start question.
        
        Args:
            session_id: Session identifier
            question_index: Index of the question being answered
            response: User's response
            
        Returns:
            Dict with next question or recommendations
        """
        if session_id not in self.sessions:
            return {'error': 'Invalid session'}
        
        session = self.sessions[session_id]
        questions = session['questions']
        
        # Store response
        if question_index < len(questions):
            session['responses'][questions[question_index]] = response
        
        # Check if more questions
        next_index = question_index + 1
        if next_index < len(questions):
            return {
                'session_id': session_id,
                'next_question': questions[next_index],
                'question_index': next_index,
                'complete': False
            }
        
        # All questions answered - generate recommendations
        recommendations = await self._generate_cold_start_recommendations(session)
        
        return {
            'session_id': session_id,
            'complete': True,
            'recommendations': recommendations
        }
    
    async def _generate_cold_start_recommendations(self, session: Dict) -> List[Dict]:
        """Generate recommendations based on cold-start responses."""
        db = get_database()
        
        # Get available categories and popular products
        categories = await db.products.distinct('category')
        popular_products_cursor = db.products.find().sort('popularity_score', -1).limit(20)
        popular_products = await popular_products_cursor.to_list(length=20)
        
        popular_list = [
            {
                'name': p['description'],
                'price': p['price'],
                'category': p['category']
            }
            for p in popular_products
        ]
        
        # Use LLM to reason about recommendations
        llm_recommendations = await llm_service.generate_cold_start_recommendations(
            user_responses=session['responses'],
            available_categories=categories,
            popular_products=popular_list
        )
        
        # Match LLM recommendations to actual products
        final_recommendations = []
        
        for llm_rec in llm_recommendations:
            category = llm_rec.get('product_category')
            
            # Find products in this category
            products_cursor = db.products.find(
                {'category': category}
            ).sort('popularity_score', -1).limit(3)
            
            products = await products_cursor.to_list(length=3)
            
            for product in products:
                final_recommendations.append({
                    'product_id': product['stock_code'],
                    'product_name': product['description'],
                    'category': product['category'],
                    'price': product['price'],
                    'reasoning': llm_rec.get('reasoning', 'Recommended for you'),
                    'priority': llm_rec.get('priority', 'medium')
                })
        
        return final_recommendations[:10]
    
    async def refine_recommendations(self, session_id: str, feedback: Dict) -> Dict:
        """
        Refine recommendations based on user feedback.
        
        Args:
            session_id: Session identifier
            feedback: Dict with liked/disliked product IDs
            
        Returns:
            Dict with refined recommendations
        """
        if session_id not in self.sessions:
            return {'error': 'Invalid session'}
        
        db = get_database()
        
        # Get liked/disliked products
        liked_ids = feedback.get('liked', [])
        disliked_ids = feedback.get('disliked', [])
        
        # Find similar products to liked ones
        refined = []
        
        if liked_ids:
            for product_id in liked_ids:
                product = await db.products.find_one({'stock_code': product_id})
                
                if product:
                    # Find similar products in same category
                    similar_cursor = db.products.find({
                        'category': product['category'],
                        'stock_code': {'$nin': liked_ids + disliked_ids}
                    }).sort('popularity_score', -1).limit(5)
                    
                    similar = await similar_cursor.to_list(length=5)
                    
                    for sim_product in similar:
                        refined.append({
                            'product_id': sim_product['stock_code'],
                            'product_name': sim_product['description'],
                            'category': sim_product['category'],
                            'price': sim_product['price'],
                            'reasoning': f"Similar to products you liked in {product['category']}"
                        })
        
        return {
            'session_id': session_id,
            'refined_recommendations': refined[:10]
        }


# Global service instance
cold_start_service = ColdStartService()
