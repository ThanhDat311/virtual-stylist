"""
Vector Database module.
Handles initialization and querying of the vector database using FAISS.
"""

import os
import faiss
import numpy as np
import time

class VectorDatabase:
    def __init__(self, index_path="data/vector_index.bin"):
        """
        Initialize the FAISS index by loading it from disk.
        """
        self.index_path = index_path
        self.index = None
        self.dimension = None
        self.load_index()

    def load_index(self):
        """Loads FAISS index from disk if it exists."""
        if os.path.exists(self.index_path):
            print(f"[VectorDatabase] Loading index from {self.index_path}...")
            t0 = time.time()
            self.index = faiss.read_index(self.index_path)
            self.dimension = self.index.d
            print(f"[VectorDatabase] Index loaded in {time.time()-t0:.2f}s! Total vectors: {self.index.ntotal}, Dim: {self.dimension}")
        else:
            print(f"[VectorDatabase] WARNING: Index file {self.index_path} not found. Call create_index() first.")

    def create_index(self, dimension):
        """
        Initialize an empty FAISS index (IndexFlatIP for cosine similarity).
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        print(f"[VectorDatabase] Created new empty FAISS index with dimension {dimension}.")

    def add_features(self, features):
        """
        Add extracted features to the index.
        Features should be L2-normalized numpy array of shape (N, dim).
        """
        if self.index is None:
            self.create_index(features.shape[1])
            
        self.index.add(features.astype(np.float32))
        
    def save_index(self):
        """Saves current index to disk."""
        if self.index:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            faiss.write_index(self.index, self.index_path)

    def search(self, query_features, top_k=5):
        """
        Search the index for the most similar features.
        
        Args:
            query_features: numpy array of shape (dim,) or (1, dim) -> L2-normalized!
            top_k: Number of nearest neighbors to retrieve.
            
        Returns:
            similarities: array of shape (1, top_k)
            indices: array of shape (1, top_k)
        """
        if self.index is None:
            raise ValueError("FAISS Index is not initialized or loaded.")
            
        # Ensure shape is (1, dim)
        if len(query_features.shape) == 1:
            query_features = np.expand_dims(query_features, axis=0)
            
        query_features = query_features.astype(np.float32)
        
        # FAISS search
        similarities, indices = self.index.search(query_features, top_k)
        return similarities, indices
