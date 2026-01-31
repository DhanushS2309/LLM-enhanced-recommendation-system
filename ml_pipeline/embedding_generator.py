"""
Embedding generation using sentence-transformers.
Creates semantic embeddings for products and users.
"""
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from typing import List, Tuple


class EmbeddingGenerator:
    """
    Generate and manage product/user embeddings using sentence-transformers.
    Uses FAISS for fast similarity search.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Sentence-transformer model name (384-dim embeddings)
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = 384
        self.product_embeddings = None
        self.product_ids = None
        self.faiss_index = None
        
    def generate_product_embeddings(self, products_df: pd.DataFrame) -> np.ndarray:
        """
        Generate embeddings for all products.
        
        Args:
            products_df: DataFrame with stock_code, description, category
            
        Returns:
            Array of embeddings (n_products, embedding_dim)
        """
        print("\n" + "="*60)
        print("GENERATING PRODUCT EMBEDDINGS")
        print("="*60 + "\n")
        
        # Create combined text for embedding
        texts = []
        self.product_ids = []
        
        for _, row in products_df.iterrows():
            # Combine description and category for richer embeddings
            text = f"{row['description']} {row['category']}"
            texts.append(text)
            self.product_ids.append(row['stock_code'])
        
        print(f"Generating embeddings for {len(texts):,} products...")
        
        # Generate embeddings in batches for efficiency
        self.product_embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"✓ Generated embeddings: {self.product_embeddings.shape}")
        
        # Build FAISS index for fast similarity search
        self._build_faiss_index()
        
        print("\n" + "="*60)
        print("✓ PRODUCT EMBEDDINGS COMPLETE")
        print("="*60 + "\n")
        
        return self.product_embeddings
    
    def _build_faiss_index(self):
        """Build FAISS index for fast nearest neighbor search."""
        print("Building FAISS index...")
        
        # Use L2 distance (can also use inner product)
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        self.faiss_index.add(self.product_embeddings.astype('float32'))
        
        print(f"✓ FAISS index built with {self.faiss_index.ntotal:,} vectors")
    
    def find_similar_products(self, product_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find products similar to a given product using embeddings.
        
        Args:
            product_id: Stock code of the product
            top_k: Number of similar products to return
            
        Returns:
            List of (product_id, distance) tuples
        """
        if product_id not in self.product_ids:
            return []
        
        # Get embedding of the query product
        idx = self.product_ids.index(product_id)
        query_embedding = self.product_embeddings[idx:idx+1].astype('float32')
        
        # Search FAISS index
        distances, indices = self.faiss_index.search(query_embedding, top_k + 1)
        
        # Convert to list (exclude the query product itself)
        similar = []
        for i, dist in zip(indices[0][1:], distances[0][1:]):
            similar.append((self.product_ids[i], float(dist)))
        
        return similar
    
    def search_by_text(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search products by natural language query.
        
        Args:
            query: Natural language search query
            top_k: Number of results to return
            
        Returns:
            List of (product_id, distance) tuples
        """
        # Generate embedding for query
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype('float32')
        
        # Search FAISS index
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        
        # Convert to list
        results = []
        for i, dist in zip(indices[0], distances[0]):
            results.append((self.product_ids[i], float(dist)))
        
        return results
    
    def save_embeddings(self, path: str = 'models/embeddings.pkl'):
        """Save embeddings and FAISS index."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data = {
            'product_embeddings': self.product_embeddings,
            'product_ids': self.product_ids,
            'embedding_dim': self.embedding_dim
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        # Save FAISS index separately
        faiss_path = path.replace('.pkl', '_faiss.index')
        faiss.write_index(self.faiss_index, faiss_path)
        
        print(f"✓ Embeddings saved: {path}")
        print(f"✓ FAISS index saved: {faiss_path}")
    
    def load_embeddings(self, path: str = 'models/embeddings.pkl'):
        """Load embeddings and FAISS index."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.product_embeddings = data['product_embeddings']
        self.product_ids = data['product_ids']
        self.embedding_dim = data['embedding_dim']
        
        # Load FAISS index
        faiss_path = path.replace('.pkl', '_faiss.index')
        self.faiss_index = faiss.read_index(faiss_path)
        
        print(f"✓ Embeddings loaded: {path}")
        print(f"✓ FAISS index loaded: {faiss_path}")


if __name__ == "__main__":
    # Load product data
    print("Loading product metadata...")
    products_df = pd.read_csv('data/features/products.csv')
    
    # Generate embeddings
    generator = EmbeddingGenerator()
    embeddings = generator.generate_product_embeddings(products_df)
    
    # Save embeddings
    generator.save_embeddings()
    
    # Test similarity search
    test_product = products_df.iloc[0]['stock_code']
    similar = generator.find_similar_products(test_product, top_k=5)
    
    print(f"\nProducts similar to {test_product}:")
    for product_id, distance in similar:
        product_info = products_df[products_df['stock_code'] == product_id].iloc[0]
        print(f"  {product_id} ({product_info['description'][:40]}): distance={distance:.3f}")
    
    # Test text search
    query = "christmas decoration"
    results = generator.search_by_text(query, top_k=5)
    
    print(f"\nSearch results for '{query}':")
    for product_id, distance in results:
        product_info = products_df[products_df['stock_code'] == product_id].iloc[0]
        print(f"  {product_info['description'][:50]}: distance={distance:.3f}")
