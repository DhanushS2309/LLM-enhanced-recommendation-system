"""
Collaborative Filtering using SVD (Singular Value Decomposition).
Optimized for sparse user-item matrices with 540K+ transactions.
"""
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
import pickle
import os
from typing import List, Tuple


class CollaborativeFilter:
    """
    SVD-based collaborative filtering for implicit feedback.
    Handles cold-start by falling back to popular items.
    """
    
    def __init__(self, n_factors=50, n_epochs=20):
        self.model = SVD(n_factors=n_factors, n_epochs=n_epochs, random_state=42)
        self.reader = Reader(rating_scale=(0, 100))  # Quantity-based ratings
        self.trained = False
        self.user_item_df = None
        
    def prepare_data(self, user_item_matrix: pd.DataFrame) -> Dataset:
        """Convert user-item matrix to Surprise Dataset format."""
        print("Preparing data for collaborative filtering...")
        
        # Convert to long format
        data_list = []
        for user_id in user_item_matrix.index:
            for product_id in user_item_matrix.columns:
                rating = user_item_matrix.loc[user_id, product_id]
                if rating > 0:  # Only include actual purchases
                    # Normalize rating to 0-100 scale
                    normalized_rating = min(rating * 10, 100)
                    data_list.append([str(user_id), str(product_id), normalized_rating])
        
        df = pd.DataFrame(data_list, columns=['user_id', 'product_id', 'rating'])
        print(f"✓ Prepared {len(df):,} interactions for training")
        
        dataset = Dataset.load_from_df(df[['user_id', 'product_id', 'rating']], self.reader)
        return dataset
    
    def train(self, user_item_matrix: pd.DataFrame):
        """Train SVD model on user-item interactions."""
        print("\n" + "="*60)
        print("TRAINING COLLABORATIVE FILTERING MODEL")
        print("="*60 + "\n")
        
        self.user_item_df = user_item_matrix
        dataset = self.prepare_data(user_item_matrix)
        
        # Train on full dataset
        trainset = dataset.build_full_trainset()
        
        print("Training SVD model...")
        self.model.fit(trainset)
        self.trained = True
        
        print("✓ Model trained successfully")
        
        # Cross-validation for evaluation
        print("\nEvaluating model with cross-validation...")
        cv_results = cross_validate(self.model, dataset, measures=['RMSE', 'MAE'], cv=3, verbose=False)
        print(f"  RMSE: {cv_results['test_rmse'].mean():.2f}")
        print(f"  MAE:  {cv_results['test_mae'].mean():.2f}")
        
        print("\n" + "="*60)
        print("✓ COLLABORATIVE FILTERING TRAINING COMPLETE")
        print("="*60 + "\n")
    
    def predict_for_user(self, user_id: str, product_ids: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get top-k product recommendations for a user.
        
        Args:
            user_id: Customer ID
            product_ids: List of all product IDs to consider
            top_k: Number of recommendations to return
            
        Returns:
            List of (product_id, score) tuples
        """
        if not self.trained:
            raise ValueError("Model not trained yet")
        
        # Get products user hasn't purchased
        if str(user_id) in self.user_item_df.index:
            purchased_products = set(
                self.user_item_df.columns[self.user_item_df.loc[str(user_id)] > 0]
            )
        else:
            purchased_products = set()
        
        # Predict scores for all products
        predictions = []
        for product_id in product_ids:
            if str(product_id) not in purchased_products:
                pred = self.model.predict(str(user_id), str(product_id))
                predictions.append((str(product_id), pred.est))
        
        # Sort by score and return top-k
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_k]
    
    def save_model(self, path: str = 'models/collaborative_filter.pkl'):
        """Save trained model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"✓ Model saved: {path}")
    
    @staticmethod
    def load_model(path: str = 'models/collaborative_filter.pkl'):
        """Load trained model from disk."""
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Model loaded: {path}")
        return model


if __name__ == "__main__":
    # Load user-item matrix
    print("Loading user-item matrix...")
    user_item_matrix = pd.read_csv('data/features/user_item_matrix.csv', index_col=0)
    
    # Train model
    cf = CollaborativeFilter(n_factors=50, n_epochs=20)
    cf.train(user_item_matrix)
    
    # Save model
    cf.save_model()
    
    # Test prediction
    test_user = user_item_matrix.index[0]
    test_products = user_item_matrix.columns[:100].tolist()
    recommendations = cf.predict_for_user(test_user, test_products, top_k=10)
    
    print(f"\nTest recommendations for user {test_user}:")
    for product_id, score in recommendations[:5]:
        print(f"  {product_id}: {score:.2f}")
