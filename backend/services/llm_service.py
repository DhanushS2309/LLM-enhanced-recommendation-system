"""
LLM Service for generating explanations and handling cold-start scenarios.
Optimized with caching to minimize API calls and latency.
Uses Groq API for fast, cost-effective inference.
"""
from groq import AsyncGroq
import json
import hashlib
from typing import Dict, List, Optional
from backend.config import get_settings
from backend.cache import cache_llm_response, get_cached_llm_response, generate_cache_key
from backend.services.prompts import (
    format_user_insight_prompt,
    format_recommendation_explanation_prompt,
    format_query_understanding_prompt,
    format_search_explanation_prompt,
    format_batch_explanation_prompt,
    COLD_START_QUESTIONS_PROMPT,
    COLD_START_REASONING_PROMPT
)

settings = get_settings()


class LLMService:
    """
    LLM service for generating insights, explanations, and handling cold-start.
    Uses Groq API with caching to improve performance and reduce API costs.
    """
    
    def __init__(self):
        self.client = None
        if settings.GROQ_API_KEY:
            self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self.model = settings.LLM_MODEL
        self.model_advanced = settings.LLM_MODEL_ADVANCED
        self.max_tokens = settings.LLM_MAX_TOKENS
        self.temperature = settings.LLM_TEMPERATURE
    
    async def _call_llm(self, prompt: str, use_advanced: bool = False, 
                       max_tokens: Optional[int] = None) -> str:
        """
        Call Groq LLM API with caching.
        
        Args:
            prompt: The prompt to send
            use_advanced: Whether to use advanced model
            max_tokens: Override default max tokens
            
        Returns:
            LLM response text
        """
        # Check cache first
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        cached_response = get_cached_llm_response(prompt_hash)
        
        if cached_response:
            return cached_response
        
        # Check if client is initialized
        if not self.client:
            print("⚠ Groq API key not configured, using fallback responses")
            return self._get_fallback_response(prompt)
        
        # Call Groq API
        try:
            model = self.model_advanced if use_advanced else self.model
            tokens = max_tokens or self.max_tokens
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful e-commerce recommendation assistant. Provide concise, relevant responses."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=tokens,
                temperature=self.temperature
            )
            
            result = response.choices[0].message.content.strip()
            
            # Cache the response
            cache_llm_response(prompt_hash, result)
            
            return result
            
        except Exception as e:
            print(f"Groq API error: {e}")
            return self._get_fallback_response(prompt)
    
    def _get_fallback_response(self, prompt: str) -> str:
        """Provide fallback response when LLM is unavailable."""
        if "insight" in prompt.lower():
            return "Valued customer with diverse shopping interests."
        elif "explain" in prompt.lower() or "recommendation" in prompt.lower():
            return "This product matches your shopping preferences and price range."
        elif "query" in prompt.lower():
            return json.dumps({"category": null, "intent": "general search", "features": [], "constraints": []})
        else:
            return "Based on your preferences, this is a good match."
    
    async def generate_user_insight(self, user_profile: Dict) -> str:
        """
        Generate insight about user's shopping behavior.
        
        Args:
            user_profile: Dict with total_spend, purchase_count, top_categories, etc.
            
        Returns:
            Insight text
        """
        prompt = format_user_insight_prompt(user_profile)
        insight = await self._call_llm(prompt, use_advanced=False)
        return insight
    
    async def explain_recommendation(self, product: Dict, user_profile: Dict, 
                                    match_score: float) -> str:
        """
        Generate explanation for why a product is recommended.
        
        Args:
            product: Product dict with name, category, price
            user_profile: User profile dict
            match_score: Recommendation match score
            
        Returns:
            Explanation text
        """
        prompt = format_recommendation_explanation_prompt(product, user_profile, match_score)
        explanation = await self._call_llm(prompt, use_advanced=False)
        return explanation
    
    async def explain_recommendations_batch(self, products: List[Dict], 
                                           user_profile: Dict) -> Dict[str, str]:
        """
        Generate explanations for multiple products in one call (more efficient).
        
        Args:
            products: List of product dicts
            user_profile: User profile dict
            
        Returns:
            Dict mapping product_id to explanation
        """
        prompt = format_batch_explanation_prompt(user_profile, products)
        
        try:
            response = await self._call_llm(prompt, use_advanced=False, max_tokens=500)
            explanations_list = json.loads(response)
            
            # Convert to dict
            explanations = {
                item['product_id']: item['explanation']
                for item in explanations_list
            }
            return explanations
            
        except json.JSONDecodeError:
            # Fallback to individual explanations
            explanations = {}
            for product in products:
                explanations[product['product_id']] = "Recommended based on your preferences."
            return explanations
    
    async def understand_query(self, query: str) -> Dict:
        """
        Extract structured information from natural language query.
        
        Args:
            query: Natural language search query
            
        Returns:
            Dict with category, intent, price range, features, constraints
        """
        prompt = format_query_understanding_prompt(query)
        
        try:
            response = await self._call_llm(prompt, use_advanced=False)
            # Parse JSON response
            parsed = json.loads(response)
            return parsed
        except json.JSONDecodeError:
            # Fallback to basic parsing
            return {
                "category": None,
                "intent": query,
                "max_price": None,
                "min_price": None,
                "features": [],
                "constraints": []
            }
    
    async def explain_search_result(self, query: str, product: Dict) -> str:
        """
        Explain why a product matches a search query.
        
        Args:
            query: Search query
            product: Product dict
            
        Returns:
            Explanation text
        """
        prompt = format_search_explanation_prompt(query, product)
        explanation = await self._call_llm(prompt, use_advanced=False)
        return explanation
    
    async def generate_cold_start_questions(self) -> List[str]:
        """
        Generate questions to ask new users for cold-start.
        
        Returns:
            List of questions
        """
        try:
            response = await self._call_llm(COLD_START_QUESTIONS_PROMPT, use_advanced=False)
            questions = json.loads(response)
            return questions
        except json.JSONDecodeError:
            # Fallback questions
            return [
                "What type of products are you interested in?",
                "What's your typical budget range?",
                "Are you shopping for yourself or as a gift?",
                "Do you prefer trendy or classic styles?"
            ]
    
    async def generate_cold_start_recommendations(self, user_responses: Dict,
                                                  available_categories: List[str],
                                                  popular_products: List[Dict]) -> List[Dict]:
        """
        Generate recommendations for new users based on their responses.
        
        Args:
            user_responses: Dict of question -> answer
            available_categories: List of available product categories
            popular_products: List of popular products
            
        Returns:
            List of recommendation dicts with category, reasoning, priority
        """
        prompt = COLD_START_REASONING_PROMPT.format(
            user_responses=json.dumps(user_responses, indent=2),
            available_categories=', '.join(available_categories),
            popular_products=json.dumps([
                f"{p['name']} (£{p['price']}, {p['category']})"
                for p in popular_products[:10]
            ], indent=2)
        )
        
        try:
            response = await self._call_llm(prompt, use_advanced=True, max_tokens=300)
            recommendations = json.loads(response)
            return recommendations
        except json.JSONDecodeError:
            # Fallback to popular products
            return [
                {
                    "product_category": cat,
                    "reasoning": "Popular category based on overall trends",
                    "priority": "medium"
                }
                for cat in available_categories[:3]
            ]


# Global LLM service instance
llm_service = LLMService()
