"""
Content-Based Filtering using TF-IDF and cosine similarity.
Fast and effective for product recommendations based on features.
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from typing import List, Tuple, Dict


class ContentBasedFilter:
    """
    Content-based filtering using product descriptions and categories.
    Works well for cold-start scenarios.
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.product_vectors = None
        self.products_df = None
        self.similarity_matrix = None
        
    def train(self, products_df: pd.DataFrame):
        """
        Train content-based model on product metadata.
        
        Args:
            products_df: DataFrame with columns: stock_code, description, category, price
        """
        print("\n" + "="*60)
        print("TRAINING CONTENT-BASED FILTERING MODEL")
        print("="*60 + "\n")
        
        self.products_df = products_df.copy()
        
        # Create combined text features
        print("Creating product feature vectors...")
        self.products_df['combined_features'] = (
            self.products_df['description'] + ' ' +
            self.products_df['category'] + ' ' +
            self.products_df['category']  # Weight category more
        )
        
        # Create TF-IDF vectors
        self.product_vectors = self.vectorizer.fit_transform(
            self.products_df['combined_features']
        )
        
        print(f"✓ Created TF-IDF vectors: {self.product_vectors.shape}")
        
        # Precompute similarity matrix for faster recommendations
        print("Computing product similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.product_vectors)
        
        print(f"✓ Similarity matrix computed: {self.similarity_matrix.shape}")
        
        print("\n" + "="*60)
        print("✓ CONTENT-BASED FILTERING TRAINING COMPLETE")
        print("="*60 + "\n")
    
    def get_similar_products(self, product_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get products similar to a given product.
        
        Args:
            product_id: Stock code of the product
            top_k: Number of similar products to return
            
        Returns:
            List of (product_id, similarity_score) tuples
        """
        if product_id not in self.products_df['stock_code'].values:
            return []
        
        # Get index of the product
        idx = self.products_df[self.products_df['stock_code'] == product_id].index[0]
        
        # Get similarity scores
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        
        # Sort by similarity (excluding the product itself)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_k+1]
        
        # Get product IDs and scores
        recommendations = [
            (self.products_df.iloc[i]['stock_code'], score)
            for i, score in sim_scores
        ]
        
        return recommendations
    
    def recommend_for_user(self, user_purchase_history: List[str], 
                          all_products: List[str], 
                          top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Recommend products based on user's purchase history.
        
        Args:
            user_purchase_history: List of product IDs user has purchased
            all_products: List of all available product IDs
            top_k: Number of recommendations to return
            
        Returns:
            List of (product_id, score) tuples
        """
        if not user_purchase_history:
            # Return popular products for cold start
            return self._get_popular_products(top_k)
        
        # Get indices of purchased products
        purchased_indices = []
        for product_id in user_purchase_history:
            if product_id in self.products_df['stock_code'].values:
                idx = self.products_df[self.products_df['stock_code'] == product_id].index[0]
                purchased_indices.append(idx)
        
        if not purchased_indices:
            return self._get_popular_products(top_k)
        
        # Compute average similarity to purchased products
        avg_similarity = self.similarity_matrix[purchased_indices].mean(axis=0)
        
        # Create recommendations excluding already purchased
        recommendations = []
        for idx, score in enumerate(avg_similarity):
            product_id = self.products_df.iloc[idx]['stock_code']
            if product_id not in user_purchase_history and product_id in all_products:
                recommendations.append((product_id, float(score)))
        
        # Sort by score and return top-k
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_k]
    
    def _get_popular_products(self, top_k: int) -> List[Tuple[str, float]]:
        """Get most popular products as fallback."""
        top_products = self.products_df.nlargest(top_k, 'popularity_score')
        return [
            (row['stock_code'], row['popularity_score'] / 100)
            for _, row in top_products.iterrows()
        ]
    
    def save_model(self, path: str = 'models/content_based_filter.pkl'):
        """Save trained model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"✓ Model saved: {path}")
    
    @staticmethod
    def load_model(path: str = 'models/content_based_filter.pkl'):
        """Load trained model from disk."""
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Model loaded: {path}")
        return model


if __name__ == "__main__":
    # Load product metadata
    print("Loading product metadata...")
    products_df = pd.read_csv('data/features/products.csv')
    
    # Train model
    cbf = ContentBasedFilter()
    cbf.train(products_df)
    
    # Save model
    cbf.save_model()
    
    # Test similar products
    test_product = products_df.iloc[0]['stock_code']
    similar = cbf.get_similar_products(test_product, top_k=5)
    
    print(f"\nProducts similar to {test_product}:")
    for product_id, score in similar:
        product_info = products_df[products_df['stock_code'] == product_id].iloc[0]
        print(f"  {product_id} ({product_info['description'][:40]}): {score:.3f}")
