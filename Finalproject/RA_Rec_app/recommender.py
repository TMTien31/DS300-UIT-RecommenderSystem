"""
Recommender Module
Handles recipe search and recommendation logic
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
from config import TOP_K_RESULTS


class RecipeRecommender:
    """Handles recipe search using late fusion strategy"""
    
    def __init__(self, model, recipes_embeddings_list, all_recipes_df):
        """
        Initialize recommender
        
        Args:
            model: SentenceTransformer model
            recipes_embeddings_list: List of embeddings per recipe
            all_recipes_df: DataFrame with all recipes
        """
        self.model = model
        self.recipes_embeddings_list = recipes_embeddings_list
        self.all_recipes_df = all_recipes_df
    
    def search_recipes(self, query: str, top_k: int = TOP_K_RESULTS) -> pd.DataFrame:
        """
        Search recipes using LATE FUSION strategy (Average Similarity)
        
        Args:
            query: User's search query (Vietnamese)
            top_k: Number of results to return
            
        Returns:
            DataFrame with top_k recipes and similarity scores
        """
        # 1. Encode query
        query_embedding = self.model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize
        
        # 2. Calculate average similarity for EACH recipe
        recipe_scores = []
        
        for recipe_idx, dish_embeds in enumerate(self.recipes_embeddings_list):
            if len(dish_embeds) == 0:
                continue
            
            # Normalize dish embeddings
            dish_embeds_norm = dish_embeds / np.linalg.norm(dish_embeds, axis=1, keepdims=True)
            
            # Compute cosine similarity with ALL sentences
            similarities = np.dot(dish_embeds_norm, query_embedding.T).flatten()
            
            # LATE FUSION: Average similarity
            avg_similarity = np.mean(similarities)
            
            recipe_scores.append({
                'recipe_idx': recipe_idx,
                'avg_similarity': float(avg_similarity),
                'max_similarity': float(np.max(similarities)),
                'min_similarity': float(np.min(similarities)),
                'num_sentences': len(similarities)
            })
        
        # 3. Sort by average similarity
        recipe_scores.sort(key=lambda x: x['avg_similarity'], reverse=True)
        top_recipes = recipe_scores[:top_k]
        
        # 4. Create results dataframe with FULL recipe info
        results = []
        for item in top_recipes:
            recipe_idx = item['recipe_idx']
            recipe = self.all_recipes_df.iloc[recipe_idx]
            
            results.append({
                'recipe_idx': recipe_idx,
                'avg_similarity': item['avg_similarity'],
                'max_similarity': item['max_similarity'],
                'min_similarity': item['min_similarity'],
                'num_sentences': item['num_sentences'],
                'title': recipe['title'],
                'type_of_food': recipe['type_of_food'],
                'cook_time': recipe['cook_time'],
                'num_of_people': recipe['num_of_people'],
                'ingredients': recipe['ingredients'],
                'step': recipe['step'],
                'note': recipe['note'],
                'description': recipe['description'],
                'link': recipe['link']
            })
        
        return pd.DataFrame(results)
    
    def generate_query_from_state(self, state: Dict[str, Any]) -> str:
        """
        Generate search query from state constraints using rule-based approach
        
        Args:
            state: Dictionary from state (with hard_constraints, soft_constraints)
            
        Returns:
            Generated query string
        """
        parts = []
        
        # Hard constraints (priority)
        if "hard_constraints" in state:
            # Type of food
            if state["hard_constraints"].get("type_of_food"):
                type_food = state["hard_constraints"]["type_of_food"]
                if type_food and type_food[0] != "none":
                    parts.append(type_food[0])
            
            # Ingredients
            if state["hard_constraints"].get("ingredients"):
                ingredients = state["hard_constraints"]["ingredients"]
                if ingredients and ingredients != ["none"]:
                    if len(ingredients) == 1:
                        parts.append(f"có {ingredients[0]}")
                    else:
                        parts.append(f"có {', '.join(ingredients)}")
        
        # Soft constraints
        if "soft_constraints" in state:
            # Number of people
            if state["soft_constraints"].get("num_of_people"):
                num_people = state["soft_constraints"]["num_of_people"]
                if num_people and num_people[0] != "none":
                    parts.append(f"cho {num_people[0]} người")
            
            # Cook time
            if state["soft_constraints"].get("cook_time"):
                cook_time = state["soft_constraints"]["cook_time"]
                if cook_time and cook_time[0] != "none":
                    parts.append(f"có thời gian nấu {cook_time[0]} phút")
        
        # Combine parts into natural query
        if not parts:
            return "Món ăn"
        
        query = " ".join(parts)
        return query


# Singleton instance
_recommender = None

def get_recommender(model=None, recipes_embeddings_list=None, all_recipes_df=None) -> RecipeRecommender:
    """Get singleton recommender instance"""
    global _recommender
    if _recommender is None:
        if model is None or recipes_embeddings_list is None or all_recipes_df is None:
            raise ValueError("First call to get_recommender must provide model, embeddings, and dataframe")
        _recommender = RecipeRecommender(model, recipes_embeddings_list, all_recipes_df)
    return _recommender
