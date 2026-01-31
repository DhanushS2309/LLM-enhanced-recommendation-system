"""
Search Service - Natural language search with LLM query understanding.
Optimized for <2s response time.
"""
import time
from typing import List, Dict
from backend.database import get_database
from backend.services.llm_service import llm_service
from ml_pipeline.embedding_generator import EmbeddingGenerator
import os


class SearchService:
    """
    Natural language search service with semantic understanding.
    """
    
    def __init__(self):
        self.embedding_gen: EmbeddingGenerator = None
        self._load_embeddings()
    
    def _load_embeddings(self):
        """Load embedding model and FAISS index."""
        if os.path.exists('models/embeddings.pkl'):
            self.embedding_gen = EmbeddingGenerator()
            self.embedding_gen.load_embeddings()
            print("✓ Embedding model loaded for search")
        else:
            print("⚠ Warning: Embeddings not found. Run ml_pipeline/train_pipeline.py first")
    
    async def natural_language_search(self, query: str, user_id: str = None,
                                     top_k: int = 10) -> Dict:
        """
        Search products using natural language query.
        
        Args:
            query: Natural language search query
            user_id: Optional user ID for personalization
            top_k: Number of results to return
            
        Returns:
            Dict with search results and metadata
        """
        start_time = time.time()
        
        # Step 1: Understand query using LLM
        query_understanding = await llm_service.understand_query(query)
        
        # Step 2: Search using embeddings
        if self.embedding_gen:
            semantic_results = self.embedding_gen.search_by_text(query, top_k=top_k * 2)
        else:
            semantic_results = []
        
        # Step 3: Filter by extracted criteria
        db = get_database()
        filtered_results = await self._filter_by_criteria(
            semantic_results, query_understanding, db
        )
        
        # Step 4: Rank and format results
        results = await self._format_results(
            filtered_results[:top_k], query, query_understanding
        )
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return {
            'query': query,
            'query_understanding': query_understanding,
            'results': results,
            'result_count': len(results),
            'processing_time_ms': round(processing_time_ms, 2)
        }
    
    async def _filter_by_criteria(self, semantic_results: List[tuple],
                                  criteria: Dict, db) -> List[Dict]:
        """Filter semantic search results by extracted criteria."""
        filtered = []
        
        for product_id, distance in semantic_results:
            # Get product from database
            product = await db.products.find_one({'stock_code': product_id})
            
            if not product:
                continue
            
            # Apply filters
            if criteria.get('category') and product.get('category') != criteria['category']:
                continue
            
            if criteria.get('max_price') and product.get('price', 0) > criteria['max_price']:
                continue
            
            if criteria.get('min_price') and product.get('price', 0) < criteria['min_price']:
                continue
            
            filtered.append({
                'product_id': product['stock_code'],
                'product_name': product['description'],
                'category': product['category'],
                'price': product['price'],
                'popularity': product.get('popularity_score', 0),
                'semantic_distance': distance
            })
        
        return filtered
    
    async def _format_results(self, results: List[Dict], query: str,
                             criteria: Dict) -> List[Dict]:
        """Format and add explanations to search results."""
        formatted = []
        
        for result in results:
            # Calculate relevance score (lower distance = higher relevance)
            relevance_score = max(0, 1 - (result['semantic_distance'] / 10))
            
            # Generate explanation
            try:
                explanation = await llm_service.explain_search_result(query, result)
            except:
                explanation = f"Matches your search for {criteria.get('intent', query)}"
            
            formatted.append({
                'product_id': result['product_id'],
                'product_name': result['product_name'],
                'category': result['category'],
                'price': result['price'],
                'relevance_score': round(relevance_score, 3),
                'explanation': explanation
            })
        
        return formatted


# Global service instance
search_service = SearchService()
