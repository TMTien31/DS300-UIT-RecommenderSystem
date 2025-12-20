"""
LLM Handler Module
Handles all interactions with Google Gemini LLM
"""
import json
from typing import List, Dict, Any
import google.generativeai as genai
from config import MODEL_NAME, GOOGLE_API_KEY

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)


class LLMHandler:
    """Handles LLM-based dialogue functions"""
    
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
    
    def classify_intent(self, user_utterance: str) -> List[str]:
        prompt = f"""
                  Classify the USER INTENT in a conversational recommender system.

                  Possible intents (can be multiple):
                  - "Provide Preference" - user states what they want (food type, ingredients, cooking time, servings, calories)
                  - "Inquire" - user asks questions
                  - "Accept Recommendation" - user accepts/likes a recommended dish
                  - "Reject Recommendation" - user rejects/dislikes a recommended dish

                  Examples (in Vietnamese):
                  - "Tôi muốn tìm món Tết" → ["Provide Preference"]
                  - "Món nào nấu nhanh cho 2 người" → ["Provide Preference"]
                  - "Bạn gợi ý món gì?" → ["Inquire"]
                  - "Món này hay đấy" → ["Accept Recommendation"]
                  - "Không, tôi không thích món này" → ["Reject Recommendation"]
                  - "Tôi thích món này" → ["Accept Recommendation"]
                  - "Cho tôi món khác" → ["Reject Recommendation"]

                  User says: "{user_utterance}"

                  Return ONLY a JSON array of intent strings (e.g., ["Provide Preference"]).
                """
        
        try:
            resp = self.model.generate_content(prompt)
            text = self._clean_json_response(resp.text)
            return json.loads(text)
        except Exception as e:
            print(f"[Error] parsing intent JSON: {e}")
            return []
    
    def update_state(self, user_utterance: str, intents: List[str], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update dialogue state based on user utterance and intents
        
        Args:
            user_utterance: User's input
            intents: List of classified intents
            state: Current dialogue state
            
        Returns:
            Updated state
        """
        # Handle "Provide Preference" intent
        if "Provide Preference" in intents:
            state = self._update_preferences(user_utterance, state)
        
        # Handle "Accept Recommendation" intent
        if "Accept Recommendation" in intents:
            dish_name = self._extract_dish_name(user_utterance, action="accept")
            if dish_name and dish_name != "NONE":
                if dish_name not in state["accepted_items"]:
                    state["accepted_items"].append(dish_name)
        
        # Handle "Reject Recommendation" intent
        if "Reject Recommendation" in intents:
            dish_name = self._extract_dish_name(user_utterance, action="reject")
            if dish_name and dish_name != "NONE":
                if dish_name not in state["rejected_items"]:
                    state["rejected_items"].append(dish_name)
        
        return state
    
    def _update_preferences(self, user_utterance: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Update state with user preferences using LLM"""
        # Detect if user is providing complete info upfront
        type_of_food_empty = len(state["hard_constraints"]["type_of_food"]) == 0
        ingredients_empty = len(state["hard_constraints"]["ingredients"]) == 0
        all_hard_empty = type_of_food_empty and ingredients_empty
        
        # Find current field being asked
        current_field = None
        for key in ["type_of_food", "ingredients"]:
            if len(state["hard_constraints"][key]) == 0:
                current_field = key
                break
        
        # Build context
        field_context = ""
        if current_field and not all_hard_empty:
            field_names = {
                "type_of_food": "TYPE OF FOOD (type_of_food)",
                "ingredients": "INGREDIENTS (ingredients)"
            }
            field_context = f"\n**IMPORTANT: Currently asking about {field_names[current_field]}. ONLY update this field, KEEP all others unchanged.**\n"
        elif all_hard_empty:
            field_context = "\n**IMPORTANT: User is providing complete information upfront. Update ALL fields mentioned in their utterance.**\n"
        
        prompt = f"""
                  You are updating a conversational dialogue state JSON based on user preferences.

                  User says: "{user_utterance}"
                  {field_context}
                  Current JSON state:
                  {json.dumps(state, indent=4, ensure_ascii=False)}

                  Rules:
                  - **If user provides complete info in one utterance**: Update ALL mentioned fields
                  - **If asking step-by-step**: ONLY update the field being asked, KEEP others unchanged
                  - **Analyze carefully**: Identify which fields the user mentioned

                  **Field-specific rules:**

                  - **type_of_food**: 
                    + Update if user mentions food type: "món kho", "món xào", "món luộc",...
                      → Example: "món kho" → type_of_food = ["món kho"]
                    + If NOT mentioned → KEEP as []

                  - **ingredients**: 
                    + Update if user mentions ingredients: "thịt lợn", "hải sản",...
                      - Specific: "thịt lợn" → ingredients = ["thịt lợn"]
                    + If NOT mentioned → KEEP as []
                    
                  - **cook_time**: IMPORTANT (soft_constraints)
                    + Update if user mentions time (including "don't care")
                      - Specific: "45 phút" → ["45 phút"]
                    + If NOT mentioned → KEEP as []
                    
                  - **algeric**: IMPORTANT (soft_constraints)
                    + Update if user mentions allergies (including "no allergies")
                      - "không dị ứng", "không bị dị ứng" → algeric = ["none"]
                      - Specific: "dị ứng tôm" → algeric = ["tôm"]
                    + If NOT mentioned → KEEP as []

                  - **num_of_people**: IMPORTANT (soft_constraints)
                    + Update if user mentions servings (including "don't care")
                      - "không cần quan tâm số người" → num_of_people = ["none"]
                      - Specific: "4 người" → num_of_people = ["4"]
                    + If NOT mentioned → KEEP as []

                  - **calories**: IMPORTANT (soft_constraints)
                    + Update if user mentions calories (including "don't care")
                      - "không cần quan tâm kcal" → calories = ["none"]
                      - Specific: "1000 kcal" → calories = ["1000 kcal"]
                    + If NOT mentioned → KEEP as []

                  IMPORTANT: Return ONLY pure JSON, NO markdown, NO ```json, NO explanation.
                  Return valid JSON with all commas and brackets in correct positions.
                  """
        
        try:
            resp = self.model.generate_content(prompt)
            text = self._clean_json_response(resp.text)
            new_state = json.loads(text)
            
            # Validate structure
            if "hard_constraints" in new_state and "soft_constraints" in new_state:
                return new_state
            else:
                print("⚠️ LLM returned incomplete JSON structure, keeping old state")
                return state
                
        except Exception as e:
            print(f"❌ Error updating state: {e}")
            return state
    
    def _extract_dish_name(self, user_utterance: str, action: str = "accept") -> str:
        """Extract dish name from user utterance"""
        action_text = "accepting/liking" if action == "accept" else "rejecting/disliking"
        
        prompt = f"""
                  User says: "{user_utterance}"

                  Extract the DISH NAME that the user is {action_text}.
                  Return ONLY the dish name, NO explanation.
                  If no dish name found, return "NONE".

                  Examples (Vietnamese):
                  - "Tôi thích món Nem rán" → "Nem rán"
                  - "Món bánh xèo này ok" → "Bánh xèo"
                  - "Cho tôi thêm phở bò" → "Phở bò"
                  - "Không thích món này" → "NONE" (no specific name)
                """
        
        try:
            resp = self.model.generate_content(prompt)
            return resp.text.strip()
        except Exception as e:
            print(f"[Error] extracting dish name: {e}")
            return "NONE"
    
    def answer_question(self, user_utterance: str, recipes_context: str, state: Dict[str, Any]) -> str:
        """
        Answer user question based on recipe buffer and state
        
        Args:
            user_utterance: User's question
            recipes_context: Formatted recipe buffer text
            state: Current dialogue state
            
        Returns:
            Answer text
        """
        state_context = json.dumps(state, indent=2, ensure_ascii=False)
        
        prompt = f"""
                Bạn là trợ lý gợi ý món ăn. Trả lời câu hỏi của user dựa trên:

                1. DANH SÁCH MÓN ĐÃ GỢI Ý (có đầy đủ thông tin):
                {recipes_context}

                2. YÊU CẦU CỦA USER (state.json):
                {state_context}

                3. USER HỎI: "{user_utterance}"

                NHIỆM VỤ - PHÂN TÍCH CÂU HỎI:
                - Nếu hỏi về "cách làm", "các bước", "làm thế nào", "nấu như thế nào" → Đưa ra "Các bước nấu"
                - Nếu hỏi về "nguyên liệu", "cần gì" → Đưa ra "Nguyên liệu"
                - Nếu hỏi về "thời gian" → Đưa ra "Thời gian"
                - Nếu hỏi về "lưu ý", "mẹo", "tips" → Đưa ra "Lưu ý"
                - Nếu hỏi tổng quát về món → Tóm tắt thông tin chính

                YÊU CẦU:
                - Trả lời bằng tiếng Việt, tự nhiên và thân thiện
                - KHÔNG dùng kiến thức nội tại, CHỈ dựa vào data trong danh sách
                - Nếu user hỏi về cách làm/bước nấu → PHẢI đưa ra các bước từ trường "Các bước nấu"
                - Trả lời đầy đủ, chi tiết khi cần
                - Nếu có link, đưa link cho user để xem thêm
                - Nếu không có thông tin → Nói "Tôi chưa có thông tin này trong danh sách món đã gợi ý"
                """
        
        try:
            resp = self.model.generate_content(prompt)
            return resp.text
        except Exception as e:
            print(f"[Error] answering question: {e}")
            return "Xin lỗi, tôi gặp lỗi khi trả lời câu hỏi của bạn."
    
    def present_recommendations(self, results_df, state: Dict[str, Any]) -> str:
        """
        Present recommendations using Gemini with natural language
        
        Args:
            results_df: DataFrame with recipe search results
            state: Current dialogue state
            
        Returns:
            Formatted presentation text
        """
        # Format recipe data
        recipes_data = []
        for idx, row in results_df.iterrows():
            recipe = {
                "index": idx + 1,
                "title": row['title'],
                "type_of_food": row['type_of_food'],
                "cook_time": row['cook_time'],
                "num_of_people": row['num_of_people'],
                "similarity_score": float(row['avg_similarity']),
                "ingredients": row['ingredients'] if row['ingredients'] else 'N/A',
                "description": row['description'] if row['description'] else 'N/A',
                "step": row['step'] if row['step'] else 'N/A',
                "note": row['note'] if row['note'] else 'N/A',
                "link": row['link'] if row['link'] else 'N/A'
            }
            recipes_data.append(recipe)
        
        prompt = f"""
                  Bạn là trợ lý gợi ý món ăn thông minh. Bạn nhận được:

                  1. YÊU CẦU CỦA NGƯỜI DÙNG (state.json):
                  {json.dumps(state, indent=2, ensure_ascii=False)}

                  2. KẾT QUẢ TÌM KIẾM TỪ HỆ THỐNG (Top {len(recipes_data)} món):
                  {json.dumps(recipes_data, indent=2, ensure_ascii=False)}

                  NHIỆM VỤ CỦA BẠN:

                  1. **QUAN TRỌNG - LỌC DỊ ỨNG:**
                    - Kiểm tra trường "algeric" trong soft_constraints
                    - Nếu có dị ứng (không phải "none"), loại BỎ các món có thành phần dị ứng trong "ingredients"
                    - Chỉ giới thiệu các món an toàn

                  2. **GỢI Ý THÔNG MINH:**
                    - Phân tích món nào phù hợp nhất với yêu cầu (thời gian, số người, loại món)
                    - Đề xuất 3-5 món nổi bật với lý do cụ thể
                    - Sử dụng tone thân thiện: "Tôi nghĩ món A sẽ ... Hoặc bạn cũng có thể thử món B vì..."

                  3. **FORMAT TRẢ LỜI:**
                    - Giới thiệu ngắn gọn (2-3 câu)
                    - Gợi ý 3-5 món với:
                      + Tên món (in đậm)
                      + Lý do phù hợp (ngắn gọn)
                      + Thông tin quan trọng (thời gian, nguyên liệu chính)
                      + Link để xem chi tiết
                    - Kết thúc: "Bạn muốn biết thêm chi tiết món nào không?"

                  YÊU CẦU:
                  - Viết bằng tiếng Việt tự nhiên, thân thiện
                  - Giải thích CỤ THỂ tại sao món này phù hợp
                  - Nếu có món bị loại do dị ứng, KHÔNG đề cập
                  - Đưa đầy đủ thông tin để user có thể hỏi thêm sau này
                  """
        
        try:
            resp = self.model.generate_content(prompt)
            return resp.text, recipes_data
        except Exception as e:
            print(f"[Error] presenting recommendations: {e}")
            return "Xin lỗi, tôi gặp lỗi khi trình bày món ăn.", recipes_data
    
    def _clean_json_response(self, text: str) -> str:
        """Clean JSON response from markdown code blocks"""
        text = text.strip()
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]
                if text.startswith("json") or text.startswith("JSON"):
                    text = text[4:]
                text = text.strip()
        return text.strip()


# Singleton instance
_llm_handler = None

def get_llm_handler() -> LLMHandler:
    """Get singleton LLM handler instance"""
    global _llm_handler
    if _llm_handler is None:
        _llm_handler = LLMHandler()
    return _llm_handler
