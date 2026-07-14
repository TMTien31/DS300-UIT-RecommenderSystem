"""
Search Recommender Module
Handles recipe search with 7 different algorithms
"""
import pandas as pd
import numpy as np
import pickle
import gc
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json

from config import SAVED_MODELS_DIR


class SearchRecommender:
    """Handles recipe search with multiple algorithms"""
    
    # Algorithm names
    ALGORITHMS = [
        "TF-IDF",
        "Keyword",
        "Ingredient TF-IDF",
        "SBERT + FAISS",
        "Hybrid TF-IDF + SBERT",
        "Hybrid General",
        "RA-Rec (Late Fusion)"
    ]
    
    def __init__(self, df: pd.DataFrame, base_path: str | Path = SAVED_MODELS_DIR):
        """
        Initialize search recommender
        
        Args:
            df: DataFrame with all recipes
            base_path: Path to saved models directory
        """
        self.df = df
        self.base_path = Path(base_path).resolve()
        self.current_algorithm = None
        self.current_model = None
        
    def load_algorithm(self, algorithm_name: str):
        """Load selected algorithm"""
        if algorithm_name == self.current_algorithm:
            return  # Already loaded
        
        # Clean up previous model
        self.cleanup()
        
        # Load new algorithm
        if algorithm_name == "TF-IDF":
            self._load_tfidf()
        elif algorithm_name == "Keyword":
            self._load_keyword()
        elif algorithm_name == "Ingredient TF-IDF":
            self._load_ingredient_tfidf()
        elif algorithm_name == "SBERT + FAISS":
            self._load_sbert()
        elif algorithm_name == "Hybrid TF-IDF + SBERT":
            self._load_hybrid_tfidf_sbert()
        elif algorithm_name == "Hybrid General":
            self._load_hybrid_general()
        elif algorithm_name == "RA-Rec (Late Fusion)":
            self._load_rarec()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        self.current_algorithm = algorithm_name
    
    def _load_tfidf(self):
        """Load TF-IDF model"""
        path = self.base_path / "TFIDF"
        
        with (path / "tfidf_vectorizer.pkl").open("rb") as f:
            vectorizer = pickle.load(f)
        
        similarity_matrix = np.load(path / "tfidf_similarity.npy")
        
        self.current_model = {
            'type': 'similarity_matrix',
            'vectorizer': vectorizer,
            'similarity': similarity_matrix
        }
    
    def _load_keyword(self):
        """Load Keyword model"""
        # Keyword uses simple text matching, no preloading needed
        self.current_model = {
            'type': 'keyword'
        }
    
    def _load_ingredient_tfidf(self):
        """Load Ingredient TF-IDF model"""
        path = self.base_path / "Ingredient_TFIDF"
        
        with (path / "ingredient_tfidf_vectorizer.pkl").open("rb") as f:
            vectorizer = pickle.load(f)
        
        similarity_matrix = np.load(path / "ingredient_tfidf_similarity.npy")
        
        self.current_model = {
            'type': 'similarity_matrix',
            'vectorizer': vectorizer,
            'similarity': similarity_matrix
        }
    
    def _load_sbert(self):
        """Load SBERT + FAISS model"""
        path = self.base_path / "SBERT_FAISS"
        
        with (path / "model_info.json").open("r", encoding="utf-8") as f:
            info = json.load(f)
        
        model = SentenceTransformer(info['model_name'])
        embeddings = np.load(path / "recipe_embeddings.npy")
        similarity_matrix = cosine_similarity(embeddings, embeddings)
        
        self.current_model = {
            'type': 'similarity_matrix',
            'model': model,
            'embeddings': embeddings,
            'similarity': similarity_matrix
        }
    
    def _load_hybrid_tfidf_sbert(self):
        """Load Hybrid TF-IDF + SBERT model"""
        hybrid_path = self.base_path / "Hybrid_TFIDF_SBERT"
        
        with (hybrid_path / "config.json").open("r", encoding="utf-8") as f:
            config = json.load(f)
        
        alpha = config['alpha']
        
        # Load TF-IDF similarity
        tfidf_sim = np.load(self.base_path / "TFIDF" / "tfidf_similarity.npy")
        
        # Load SBERT embeddings and compute similarity
        sbert_embeddings = np.load(hybrid_path / "sbert_embeddings.npy")
        sbert_sim = cosine_similarity(sbert_embeddings, sbert_embeddings)
        
        # Combine
        combined_sim = alpha * tfidf_sim + (1 - alpha) * sbert_sim
        
        self.current_model = {
            'type': 'similarity_matrix',
            'similarity': combined_sim,
            'alpha': alpha
        }
    
    def _load_hybrid_general(self):
        """Load Hybrid General model"""
        path = self.base_path / "Hybrid"
        similarity_matrix = np.load(path / "hybrid_similarity.npy")
        
        self.current_model = {
            'type': 'similarity_matrix',
            'similarity': similarity_matrix
        }
    
    def _load_rarec(self):
        """Load RA-Rec Late Fusion model"""
        rarec_path = self.base_path / "RA_Rec"
        if not rarec_path.exists():
            rarec_path = self.base_path.parent / "RA_Rec"
        
        model = SentenceTransformer('keepitreal/vietnamese-sbert')
        
        with (rarec_path / "recipes_embeddings_list.pkl").open("rb") as f:
            embeddings_list = pickle.load(f)
        
        self.current_model = {
            'type': 'late_fusion',
            'model': model,
            'embeddings_list': embeddings_list
        }
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search recipes using current algorithm
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of recipe dictionaries with scores
        """
        if self.current_model is None:
            raise ValueError("No algorithm loaded. Call load_algorithm() first.")
        
        if self.current_model['type'] == 'similarity_matrix':
            return self._search_similarity_matrix(query, top_k)
        elif self.current_model['type'] == 'keyword':
            return self._search_keyword(query, top_k)
        elif self.current_model['type'] == 'late_fusion':
            return self._search_late_fusion(query, top_k)
        else:
            raise ValueError(f"Unknown model type: {self.current_model['type']}")
    
    def _search_similarity_matrix(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search using precomputed similarity matrix"""
        # Find best matching recipe in dataset
        query_lower = query.lower()
        best_match_idx = None
        best_match_score = 0
        
        for idx, row in self.df.iterrows():
            score = 0
            title_lower = str(row['title']).lower()
            ingredients_lower = str(row.get('ingredients', '')).lower()
            
            # Count keyword matches
            for word in query_lower.split():
                if len(word) > 2:
                    if word in title_lower:
                        score += 2
                    if word in ingredients_lower:
                        score += 1
            
            if score > best_match_score:
                best_match_score = score
                best_match_idx = idx
        
        if best_match_idx is None:
            return []
        
        # Get recommendations from similarity matrix
        similarity_matrix = self.current_model['similarity']
        scores = similarity_matrix[best_match_idx].copy()
        scores[best_match_idx] = -1  # Exclude self
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            recipe = self.df.iloc[idx]
            results.append({
                'title': recipe['title'],
                'type_of_food': recipe.get('type_of_food', 'N/A'),
                'ingredients': recipe.get('ingredients', 'N/A'),
                'description': recipe.get('description', 'N/A'),
                'step': recipe.get('step', 'N/A'),
                'cook_time': recipe.get('cook_time', 'N/A'),
                'num_of_people': recipe.get('num_of_people', 'N/A'),
                'calories': recipe.get('calories', 'N/A'),
                'source': recipe.get('source', 'N/A'),
                'link': recipe.get('link', 'N/A'),
                'score': float(scores[idx])
            })
        
        return results
    
    def _search_keyword(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search using keyword matching"""
        query_lower = query.lower()
        scores = []
        
        for idx, row in self.df.iterrows():
            score = 0
            title_lower = str(row['title']).lower()
            ingredients_lower = str(row.get('ingredients', '')).lower()
            
            for word in query_lower.split():
                if len(word) > 2:
                    if word in title_lower:
                        score += 2
                    if word in ingredients_lower:
                        score += 1
            
            scores.append(score)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                recipe = self.df.iloc[idx]
                results.append({
                    'title': recipe['title'],
                    'type_of_food': recipe.get('type_of_food', 'N/A'),
                    'ingredients': recipe.get('ingredients', 'N/A'),
                    'description': recipe.get('description', 'N/A'),
                    'step': recipe.get('step', 'N/A'),
                    'cook_time': recipe.get('cook_time', 'N/A'),
                    'num_of_people': recipe.get('num_of_people', 'N/A'),
                    'calories': recipe.get('calories', 'N/A'),
                    'source': recipe.get('source', 'N/A'),
                    'link': recipe.get('link', 'N/A'),
                    'score': float(scores[idx])
                })
        
        return results
    
    def _search_late_fusion(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search using Late Fusion (RA-Rec)"""
        model = self.current_model['model']
        embeddings_list = self.current_model['embeddings_list']
        
        # Encode query
        query_embedding = model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Calculate average similarity for each recipe
        recipe_scores = []
        
        for recipe_idx, dish_embeds in enumerate(embeddings_list):
            if len(dish_embeds) == 0:
                continue
            
            # Normalize embeddings
            dish_embeds_norm = dish_embeds / np.linalg.norm(dish_embeds, axis=1, keepdims=True)
            
            # Compute similarity
            similarities = np.dot(dish_embeds_norm, query_embedding.T).flatten()
            avg_similarity = np.mean(similarities)
            
            recipe_scores.append((recipe_idx, avg_similarity))
        
        # Sort and get top k
        recipe_scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in recipe_scores[:top_k]:
            recipe = self.df.iloc[idx]
            results.append({
                'title': recipe['title'],
                'type_of_food': recipe.get('type_of_food', 'N/A'),
                'ingredients': recipe.get('ingredients', 'N/A'),
                'description': recipe.get('description', 'N/A'),
                'step': recipe.get('step', 'N/A'),
                'cook_time': recipe.get('cook_time', 'N/A'),
                'num_of_people': recipe.get('num_of_people', 'N/A'),
                'calories': recipe.get('calories', 'N/A'),
                'source': recipe.get('source', 'N/A'),
                'link': recipe.get('link', 'N/A'),
                'score': float(score)
            })
        
        return results
    
    def cleanup(self):
        """Clean up current model from memory"""
        if self.current_model is not None:
            self.current_model = None
            self.current_algorithm = None
            gc.collect()


# Singleton instance
_search_recommender = None


def get_search_recommender(df: pd.DataFrame = None, base_path: str | Path = SAVED_MODELS_DIR):
    """Get or create SearchRecommender instance"""
    global _search_recommender
    
    if _search_recommender is None:
        if df is None:
            raise ValueError("DataFrame required for first initialization")
        _search_recommender = SearchRecommender(df, base_path)
    
    return _search_recommender
