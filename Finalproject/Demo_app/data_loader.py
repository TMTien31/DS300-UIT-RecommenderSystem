"""
Data Loader Module
Handles loading of models, embeddings, and recipe data
"""
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from pathlib import Path
from config import RECIPES_CSV, EMBEDDINGS_FILE, SBERT_MODEL


class DataLoader:
    """Handles loading and caching of data and models"""
    
    def __init__(self):
        self._model = None
        self._recipes_df = None
        self._embeddings_list = None
    
    def load_model(self) -> SentenceTransformer:
        """Load Vietnamese SBERT model (cached)"""
        if self._model is None:
            print(f"Loading SBERT model: {SBERT_MODEL}...")
            self._model = SentenceTransformer(SBERT_MODEL)
            print(f"Model loaded. Embedding dimension: {self._model.get_sentence_embedding_dimension()}")
        return self._model
    
    def load_recipes_data(self) -> pd.DataFrame:
        """Load recipes CSV (cached)"""
        if self._recipes_df is None:
            print(f"Loading recipes from: {RECIPES_CSV}")
            self._recipes_df = pd.read_csv(RECIPES_CSV)
            print(f"Loaded {len(self._recipes_df)} recipes")
            print(f"Columns: {self._recipes_df.columns.tolist()}")
        return self._recipes_df
    
    def load_embeddings(self) -> list:
        """Load recipe embeddings (cached)"""
        if self._embeddings_list is None:
            print(f"Loading embeddings from: {EMBEDDINGS_FILE}")
            with open(EMBEDDINGS_FILE, "rb") as f:
                self._embeddings_list = pickle.load(f)
            print(f"Loaded {len(self._embeddings_list)} recipe embeddings")
        return self._embeddings_list
    
    def load_all(self) -> tuple:
        """
        Load all data and models
        
        Returns:
            Tuple of (model, recipes_df, embeddings_list)
        """
        model = self.load_model()
        recipes_df = self.load_recipes_data()
        embeddings_list = self.load_embeddings()
        return model, recipes_df, embeddings_list
    
    def is_data_available(self) -> bool:
        """Check if all required files exist"""
        return (
            RECIPES_CSV.exists() and 
            EMBEDDINGS_FILE.exists()
        )


# Singleton instance
_data_loader = None

def get_data_loader() -> DataLoader:
    """Get singleton data loader instance"""
    global _data_loader
    if _data_loader is None:
        _data_loader = DataLoader()
    return _data_loader
