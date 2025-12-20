"""
State Management Module
Handles conversation state and recipe buffer
"""
import json
from pathlib import Path
from typing import Dict, List, Any
from config import DEFAULT_STATE, STATE_FILE


class StateManager:
    """Manages dialogue state and recipe buffer"""
    
    def __init__(self, state_file: Path = STATE_FILE):
        self.state_file = state_file
        self.recipes_buffer = []
    
    def load_state(self) -> Dict[str, Any]:
        """Load dialogue state from JSON file"""
        if self.state_file.exists():
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            self.save_state(DEFAULT_STATE)
            return DEFAULT_STATE.copy()
    
    def save_state(self, state: Dict[str, Any]) -> None:
        """Save dialogue state to JSON file"""
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4, ensure_ascii=False)
    
    def reset_state(self) -> Dict[str, Any]:
        """Reset state to default and clear buffer"""
        self.recipes_buffer = []
        self.save_state(DEFAULT_STATE)
        return DEFAULT_STATE.copy()
    
    def add_recipes_to_buffer(self, recipes_data: List[Dict[str, Any]]) -> None:
        """Add recipes to buffer (replaces existing)"""
        self.recipes_buffer = recipes_data
    
    def get_recipes_buffer_text(self) -> str:
        """Get formatted text representation of recipe buffer"""
        if not self.recipes_buffer:
            return "Chưa có món nào được gợi ý."
        
        buffer_text = []
        for recipe in self.recipes_buffer:
            recipe_info = f"""
                            Món: {recipe['title']}
                            - Loại: {recipe['type_of_food']}
                            - Thời gian: {recipe['cook_time']}
                            - Số người: {recipe['num_of_people']}
                            - Nguyên liệu: {recipe.get('ingredients', 'N/A')}
                            - Mô tả: {recipe.get('description', 'N/A')}
                            - Các bước nấu: {recipe.get('step', 'N/A')}
                            - Lưu ý: {recipe.get('note', 'N/A')}
                            - Link: {recipe['link']}
                          """
            buffer_text.append(recipe_info.strip())
        
        return "\n\n".join(buffer_text)
    
    def get_recipes_buffer(self) -> List[Dict[str, Any]]:
        """Get raw recipe buffer data"""
        return self.recipes_buffer
    
    def is_info_complete(self, state: Dict[str, Any]) -> bool:
        """Check if all required information is collected"""
        # Check hard constraints
        for key in state["hard_constraints"]:
            if len(state["hard_constraints"][key]) == 0:
                return False
        
        # Check if at least one soft constraint is filled
        all_soft_empty = all(
            len(state["soft_constraints"][key]) == 0 
            for key in state["soft_constraints"]
        )
        
        return not all_soft_empty
    
    def get_missing_field(self, state: Dict[str, Any]) -> str:
        """Get the next missing field to ask about"""
        # Check hard constraints first
        for key in state["hard_constraints"]:
            if len(state["hard_constraints"][key]) == 0:
                return key
        
        # Then check soft constraints
        for key in state["soft_constraints"]:
            if len(state["soft_constraints"][key]) == 0:
                return key
        
        return None


# Singleton instance
_state_manager = None

def get_state_manager() -> StateManager:
    """Get singleton state manager instance"""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager
