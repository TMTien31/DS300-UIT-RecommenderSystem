"""
Dialogue Manager
Orchestrates the conversation flow
"""
from typing import Dict, Any, Tuple
from state_manager import get_state_manager
from llm_handler import get_llm_handler
from recommender import get_recommender
from config import RESTART_KEYWORDS, TOP_K_DISPLAY


class DialogueManager:
    """Manages conversation flow and dialogue logic"""
    
    def __init__(self):
        self.state_manager = get_state_manager()
        self.llm_handler = get_llm_handler()
        self.recommender = None  # Will be set when data is loaded
    
    def set_recommender(self, recommender):
        """Set the recommender instance"""
        self.recommender = recommender
    
    def check_restart_intent(self, user_utterance: str) -> bool:
        """Check if user wants to restart conversation"""
        return any(keyword in user_utterance.lower() for keyword in RESTART_KEYWORDS)
    
    def select_action(self, intents: list, state: Dict[str, Any]) -> str:
        """
        Select action based on intents and state
        
        Returns:
            Action string: "Answer", "Request Information", or "Info Complete"
        """
        # If user is asking a question
        if "Inquire" in intents:
            return "Answer"
        
        # Check if all hard constraints are filled
        for key in state["hard_constraints"]:
            if len(state["hard_constraints"][key]) == 0:
                return "Request Information"
        
        # Check if soft constraints are filled
        all_soft_empty = all(
            len(state["soft_constraints"][key]) == 0 
            for key in state["soft_constraints"]
        )
        
        if all_soft_empty:
            return "Request Information"
        
        # All information collected
        return "Info Complete"
    
    def generate_question(self, state: Dict[str, Any]) -> str:
        """Generate question for missing information"""
        questions = {
            "type_of_food": "Bạn muốn tìm loại món gì? (ví dụ: Món kho, món luộc, món xào...)",
            "ingredients": "Bạn muốn món có nguyên liệu gì? (ví dụ: thịt lợn, hải sản, rau...)",
            "cook_time": "Bạn muốn món nấu trong bao lâu? (nếu không quan tâm hãy nói 'không')",
            "num_of_people": "Bạn muốn món ăn nấu cho bao nhiêu người? (nếu không quan tâm hãy nói 'không')",
            "calories": "Bạn quan tâm đến mức calories không? (nếu không quan tâm hãy nói 'không')",
            "algeric": "Bạn có dị ứng với thành phần nào không? (nếu không thì nói 'không')"
        }
        
        # Find missing field
        missing_field = self.state_manager.get_missing_field(state)
        
        # get này là method của dict, trả về giá trị tương ứng với key truyền vào
        return questions.get(missing_field, "Tôi cần thêm thông tin để giúp bạn.")
    
    def process_message(self, user_message: str) -> Tuple[str, Dict[str, Any], str, list]:
        """
        Process user message and return response
        
        Args:
            user_message: User's input message
            
        Returns:
            Tuple of (response, state, phase, recommendations)
        """
        # Check for restart
        if self.check_restart_intent(user_message):
            state = self.state_manager.reset_state()
            return "Đã reset hệ thống!", state, "info_gathering", []
        
        # Load current state
        state = self.state_manager.load_state()
        
        # Classify intent
        intents = self.llm_handler.classify_intent(user_message)
        
        # Special handling for info-gathering phase
        asking_hard = any(
            len(state["hard_constraints"][key]) == 0 
            for key in state["hard_constraints"]
        )
        if asking_hard and "Provide Preference" not in intents:
            intents = ["Provide Preference"]
        
        # Special handling for "không" responses
        if any(word in user_message.lower() for word in ["không", "khong", "không có", "không dị ứng"]):
            if "algeric" in state["soft_constraints"] and state["soft_constraints"]["algeric"] == []:
                state["soft_constraints"]["algeric"] = ["none"]
                self.state_manager.save_state(state)
                if "Provide Preference" not in intents:
                    intents.append("Provide Preference")
        
        # Update state
        state = self.llm_handler.update_state(user_message, intents, state)
        self.state_manager.save_state(state)
        
        # Select action
        action = self.select_action(intents, state)
        
        # Handle different actions
        if action == "Answer":
            # Answer question
            recipes_context = self.state_manager.get_recipes_buffer_text()
            response = self.llm_handler.answer_question(user_message, recipes_context, state)
            return response, state, "feedback", []
        
        elif action == "Info Complete":
            # Generate recommendations
            if self.recommender is None:
                return "Lỗi: Recommender chưa được khởi tạo", state, "error", []
            
            query = self.recommender.generate_query_from_state(state)
            results = self.recommender.search_recipes(query)
            
            # Present recommendations
            presentation, recipes_data = self.llm_handler.present_recommendations(
                results.head(TOP_K_DISPLAY), 
                state
            )
            
            # Save to buffer
            self.state_manager.add_recipes_to_buffer(recipes_data[:TOP_K_DISPLAY])
            
            # Update state
            state["recommended_items"] = [recipe["title"] for recipe in recipes_data[:TOP_K_DISPLAY]]
            self.state_manager.save_state(state)
            
            return presentation, state, "recommending", recipes_data[:TOP_K_DISPLAY]
        
        else:  # Request Information
            response = self.generate_question(state)
            return response, state, "info_gathering", []
    
    def process_feedback(self, user_message: str) -> Tuple[str, Dict[str, Any], list]:
        """
        Process feedback after recommendations
        
        Args:
            user_message: User's feedback message
            
        Returns:
            Tuple of (response, state, recommendations)
        """
        # Load state
        state = self.state_manager.load_state()
        
        # Classify intent
        intents = self.llm_handler.classify_intent(user_message)
        
        # Update state
        state = self.llm_handler.update_state(user_message, intents, state)
        self.state_manager.save_state(state)
        
        # Generate response
        if "Accept Recommendation" in intents:
            response = f"Tuyệt vời! Tôi sẽ lưu lại sở thích này của bạn.\nĐã thích: {', '.join(state['accepted_items'])}"
            return response, state, []
            
        elif "Reject Recommendation" in intents:
            response = "Được rồi! Để tôi tìm món khác phù hợp hơn..."
            
            # Get new recommendations
            if self.recommender is None:
                return "Lỗi: Recommender chưa được khởi tạo", state, []
            
            query = self.recommender.generate_query_from_state(state)
            results = self.recommender.search_recipes(query)
            
            # Filter out rejected items
            filtered_results = results[~results['title'].isin(state['rejected_items'])]
            
            if len(filtered_results) > 0:
                presentation, recipes_data = self.llm_handler.present_recommendations(
                    filtered_results.head(TOP_K_DISPLAY), 
                    state
                )
                self.state_manager.add_recipes_to_buffer(recipes_data[:TOP_K_DISPLAY])
                response = f"{response}\n\n{presentation}"
                return response, state, recipes_data[:TOP_K_DISPLAY]
            else:
                response = "Xin lỗi, không còn món nào khác phù hợp. Bạn có thể nói 'restart' để tìm món hoàn toàn mới."
                return response, state, []
        
        elif "Inquire" in intents:
            recipes_context = self.state_manager.get_recipes_buffer_text()
            response = self.llm_handler.answer_question(user_message, recipes_context, state)
            return response, state, []
        
        else:
            response = "Tôi có thể giúp gì thêm cho bạn?"
            return response, state, []


# Singleton instance
_dialogue_manager = None

def get_dialogue_manager() -> DialogueManager:
    """Get singleton dialogue manager instance"""
    global _dialogue_manager
    if _dialogue_manager is None:
        _dialogue_manager = DialogueManager()
    return _dialogue_manager
