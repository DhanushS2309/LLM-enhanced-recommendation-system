"""
Complete ML training pipeline orchestration.
Runs all steps: ingestion, feature engineering, model training, embeddings.
"""
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_pipeline.data_ingestion import run_ingestion_pipeline
from ml_pipeline.feature_engineering import run_feature_engineering
from ml_pipeline.collaborative_filtering import CollaborativeFilter
from ml_pipeline.content_based_filtering import ContentBasedFilter
from ml_pipeline.embedding_generator import EmbeddingGenerator
import pandas as pd


def run_complete_pipeline():
    """
    Run the complete ML training pipeline.
    This processes 540K+ transactions and trains all models.
    """
    print("\n" + "="*70)
    print(" " * 15 + "ML TRAINING PIPELINE")
    print("="*70 + "\n")
    
    start_time = datetime.now()
    
    # Step 1: Data Ingestion
    print("STEP 1/5: Data Ingestion")
    df, stats = run_ingestion_pipeline()
    print(f"✓ Processed {stats['final_records']:,} transactions\n")
    
    # Step 2: Feature Engineering
    print("STEP 2/5: Feature Engineering")
    user_profiles, products, user_item_matrix = run_feature_engineering(df)
    print(f"✓ Created {len(user_profiles):,} user profiles")
    print(f"✓ Created {len(products):,} product profiles\n")
    
    # Step 3: Train Collaborative Filtering
    print("STEP 3/5: Training Collaborative Filtering")
    cf_model = CollaborativeFilter(n_factors=50, n_epochs=20)
    cf_model.train(user_item_matrix)
    cf_model.save_model()
    print()
    
    # Step 4: Train Content-Based Filtering
    print("STEP 4/5: Training Content-Based Filtering")
    cbf_model = ContentBasedFilter()
    cbf_model.train(products)
    cbf_model.save_model()
    print()
    
    # Step 5: Generate Embeddings
    print("STEP 5/5: Generating Product Embeddings")
    embedding_gen = EmbeddingGenerator()
    embedding_gen.generate_product_embeddings(products)
    embedding_gen.save_embeddings()
    print()
    
    # Pipeline complete
    elapsed = datetime.now() - start_time
    
    print("="*70)
    print(" " * 20 + "PIPELINE COMPLETE")
    print("="*70)
    print(f"\nTotal time: {elapsed}")
    print(f"\nModels saved in: models/")
    print(f"Features saved in: data/features/")
    print("\n" + "="*70 + "\n")
    
    # Summary statistics
    print("SUMMARY:")
    print(f"  Transactions processed: {stats['final_records']:,}")
    print(f"  Unique customers: {stats['unique_customers']:,}")
    print(f"  Unique products: {stats['unique_products']:,}")
    print(f"  User profiles created: {len(user_profiles):,}")
    print(f"  Product embeddings: {len(products):,}")
    print(f"  Models trained: 3 (Collaborative, Content-Based, Embeddings)")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    run_complete_pipeline()
