"""
Hybrid Recommendation Engine combining Collaborative and Content-Based filtering.
Optimized for <500ms response time with intelligent cold-start handling.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import time
from ml_pipeline.collaborative_filtering import CollaborativeFilter
from ml_pipeline.content_based_filtering import ContentBasedFilter


class HybridRecommender:
    """
    Hybrid recommendation engine with adaptive weighting.
    Automatically handles cold-start scenarios.
    """
    
    def __init__(self, 
                 collaborative_weight: float = 0.6,
                 content_weight: float = 0.4,
                 min_purchases_for_cf: int = 5):
        """
        Initialize hybrid recommender.
        
        Args:
            collaborative_weight: Weight for collaborative filtering (0-1)
            content_weight: Weight for content-based filtering (0-1)
            min_purchases_for_cf: Minimum purchases to use collaborative filtering
        """
        self.collaborative_weight = collaborative_weight
        self.content_weight = content_weight
        self.min_purchases_for_cf = min_purchases_for_cf
        
        self.cf_model: Optional[CollaborativeFilter] = None
        self.cbf_model: Optional[ContentBasedFilter] = None
        self.products_df: Optional[pd.DataFrame] = None
        self.user_item_matrix: Optional[pd.DataFrame] = None
        
    def load_models(self, 
                   cf_path: str = 'models/collaborative_filter.pkl',
                   cbf_path: str = 'models/content_based_filter.pkl',
                   products_path: str = 'data/features/products.csv',
                   user_item_path: str = 'data/features/user_item_matrix.csv'):
        """Load pre-trained models and data."""
        print("Loading hybrid recommender models...")
        
        self.cf_model = CollaborativeFilter.load_model(cf_path)
        self.cbf_model = ContentBasedFilter.load_model(cbf_path)
        self.products_df = pd.read_csv(products_path)
        self.user_item_matrix = pd.read_csv(user_item_path, index_col=0)
        
        print("✓ Hybrid recommender ready")
    
    def get_user_purchase_history(self, user_id: str) -> List[str]:
        """Get list of products user has purchased."""
        if str(user_id) not in self.user_item_matrix.index:
            return []
        
        user_row = self.user_item_matrix.loc[str(user_id)]
        purchased = user_row[user_row > 0].index.tolist()
        return purchased
    
    def recommend(self, user_id: str, top_k: int = 10, exclude_purchased: bool = True) -> List[Dict]:
        """
        Generate hybrid recommendations for a user.
        
        Args:
            user_id: Customer ID
            top_k: Number of recommendations to return
            exclude_purchased: Whether to exclude already purchased items
            
        Returns:
            List of recommendation dicts with product_id, score, method
        """
        start_time = time.time()
        
        # Get user's purchase history
        purchase_history = self.get_user_purchase_history(user_id)
        num_purchases = len(purchase_history)
        
        # All available products
        all_products = self.products_df['stock_code'].tolist()
        
        # Determine strategy based on purchase history
        if num_purchases == 0:
            # Cold start: Use only content-based (popular items)
            recommendations = self._cold_start_recommend(all_products, top_k)
            method = "cold_start"
            
        elif num_purchases < self.min_purchases_for_cf:
            # Warm start: Favor content-based (70/30)
            recommendations = self._warm_start_recommend(
                user_id, purchase_history, all_products, top_k
            )
            method = "warm_start"
            
        else:
            # Full hybrid: Use configured weights
            recommendations = self._hybrid_recommend(
                user_id, purchase_history, all_products, top_k
            )
            method = "hybrid"
        
        # Exclude purchased items if requested
        if exclude_purchased:
            recommendations = [
                rec for rec in recommendations 
                if rec['product_id'] not in purchase_history
            ]
        
        # Add product details
        recommendations = self._enrich_recommendations(recommendations[:top_k])
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Add metadata
        for rec in recommendations:
            rec['method'] = method
            rec['processing_time_ms'] = round(elapsed_ms, 2)
        
        return recommendations
    
    def _cold_start_recommend(self, all_products: List[str], top_k: int) -> List[Dict]:
        """Recommend popular items for new users."""
        popular = self.products_df.nlargest(top_k, 'popularity_score')
        
        recommendations = []
        for _, row in popular.iterrows():
            recommendations.append({
                'product_id': row['stock_code'],
                'score': row['popularity_score'] / 100,
                'source': 'popularity'
            })
        
        return recommendations
    
    def _warm_start_recommend(self, user_id: str, purchase_history: List[str], 
                              all_products: List[str], top_k: int) -> List[Dict]:
        """Recommend with emphasis on content-based for users with few purchases."""
        # 70% content-based, 30% collaborative
        cbf_recs = self.cbf_model.recommend_for_user(
            purchase_history, all_products, top_k * 2
        )
        
        try:
            cf_recs = self.cf_model.predict_for_user(
                user_id, all_products, top_k * 2
            )
        except:
            cf_recs = []
        
        # Combine with adjusted weights
        combined = self._combine_scores(cbf_recs, cf_recs, cbf_weight=0.7, cf_weight=0.3)
        return combined[:top_k]
    
    def _hybrid_recommend(self, user_id: str, purchase_history: List[str],
                         all_products: List[str], top_k: int) -> List[Dict]:
        """Full hybrid recommendation with configured weights."""
        # Get recommendations from both models
        cbf_recs = self.cbf_model.recommend_for_user(
            purchase_history, all_products, top_k * 2
        )
        
        cf_recs = self.cf_model.predict_for_user(
            user_id, all_products, top_k * 2
        )
        
        # Combine scores
        combined = self._combine_scores(
            cbf_recs, cf_recs, 
            cbf_weight=self.content_weight,
            cf_weight=self.collaborative_weight
        )
        
        return combined[:top_k]
    
    def _combine_scores(self, cbf_recs: List[Tuple[str, float]], 
                       cf_recs: List[Tuple[str, float]],
                       cbf_weight: float, cf_weight: float) -> List[Dict]:
        """Combine scores from both models with normalization."""
        # Normalize scores to 0-1 range
        cbf_dict = dict(cbf_recs)
        cf_dict = dict(cf_recs)
        
        # Normalize CBF scores
        if cbf_dict:
            max_cbf = max(cbf_dict.values())
            cbf_dict = {k: v/max_cbf if max_cbf > 0 else v for k, v in cbf_dict.items()}
        
        # Normalize CF scores
        if cf_dict:
            max_cf = max(cf_dict.values())
            cf_dict = {k: v/max_cf if max_cf > 0 else v for k, v in cf_dict.items()}
        
        # Combine scores
        all_products = set(cbf_dict.keys()) | set(cf_dict.keys())
        combined = []
        
        for product_id in all_products:
            cbf_score = cbf_dict.get(product_id, 0)
            cf_score = cf_dict.get(product_id, 0)
            
            final_score = (cbf_weight * cbf_score) + (cf_weight * cf_score)
            
            combined.append({
                'product_id': product_id,
                'score': final_score,
                'cbf_score': cbf_score,
                'cf_score': cf_score
            })
        
        # Sort by final score
        combined.sort(key=lambda x: x['score'], reverse=True)
        return combined
    
    def _enrich_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """Add product details to recommendations."""
        enriched = []
        
        for rec in recommendations:
            product_id = rec['product_id']
            product_info = self.products_df[
                self.products_df['stock_code'] == product_id
            ]
            
            if not product_info.empty:
                product_info = product_info.iloc[0]
                rec['product_name'] = product_info['description']
                rec['category'] = product_info['category']
                rec['price'] = float(product_info['price'])
                rec['popularity'] = float(product_info['popularity_score'])
                enriched.append(rec)
        
        return enriched
    
    def get_recommendation_stats(self, user_id: str) -> Dict:
        """Get statistics about user for recommendation context."""
        purchase_history = self.get_user_purchase_history(user_id)
        
        if not purchase_history:
            return {
                'user_id': user_id,
                'purchase_count': 0,
                'is_cold_start': True,
                'recommendation_strategy': 'cold_start'
            }
        
        # Calculate user stats
        num_purchases = len(purchase_history)
        
        # Get categories
        purchased_products = self.products_df[
            self.products_df['stock_code'].isin(purchase_history)
        ]
        
        top_categories = purchased_products['category'].value_counts().head(3).index.tolist()
        avg_price = purchased_products['price'].mean()
        
        # Determine strategy
        if num_purchases < self.min_purchases_for_cf:
            strategy = 'warm_start'
        else:
            strategy = 'hybrid'
        
        return {
            'user_id': user_id,
            'purchase_count': num_purchases,
            'is_cold_start': False,
            'recommendation_strategy': strategy,
            'top_categories': top_categories,
            'avg_price': round(avg_price, 2) if not pd.isna(avg_price) else 0
        }


if __name__ == "__main__":
    # Initialize hybrid recommender
    recommender = HybridRecommender()
    recommender.load_models()
    
    # Test with different user scenarios
    user_item_matrix = pd.read_csv('data/features/user_item_matrix.csv', index_col=0)
    
    # Test 1: Existing user with many purchases
    test_user_1 = user_item_matrix.index[0]
    print(f"\n{'='*60}")
    print(f"Test 1: User with purchase history ({test_user_1})")
    print(f"{'='*60}")
    
    stats = recommender.get_recommendation_stats(test_user_1)
    print(f"Purchase count: {stats['purchase_count']}")
    print(f"Strategy: {stats['recommendation_strategy']}")
    
    recs = recommender.recommend(test_user_1, top_k=5)
    print(f"\nTop 5 recommendations:")
    for i, rec in enumerate(recs, 1):
        print(f"{i}. {rec['product_name'][:40]} - £{rec['price']:.2f} (score: {rec['score']:.3f})")
    
    print(f"\nProcessing time: {recs[0]['processing_time_ms']:.2f}ms")
