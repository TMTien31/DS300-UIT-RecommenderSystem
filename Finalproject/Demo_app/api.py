"""
FastAPI Backend for RA-Rec Chatbot
Provides REST API endpoints for the conversational recommender system
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn

from data_loader import get_data_loader
from state_manager import get_state_manager
from llm_handler import get_llm_handler
from recommender import get_recommender
from config import GREETING_MESSAGE, RESTART_KEYWORDS, TOP_K_DISPLAY

# Initialize FastAPI app
app = FastAPI(
    title="RA-Rec Chatbot API",
    description="Conversational Food Recommender System API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (initialized on startup)
state_manager = None
llm_handler = None
recommender = None


# Pydantic models
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    state: Dict[str, Any]


@app.on_event("startup")
async def startup_event():
    """Load models and data on startup"""
    global state_manager, llm_handler, recommender
    
    print("Starting RA-Rec Chatbot API...")
    
    # Initialize components
    data_loader = get_data_loader()
    
    # Check if data is available
    if not data_loader.is_data_available():
        print("ERROR: Required data files not found!")
        print("Please ensure the following files exist:")
        print(f"  - {data_loader.RECIPES_CSV}")
        print(f"  - {data_loader.EMBEDDINGS_FILE}")
        raise RuntimeError("Required data files not found")
    
    # Load data and models
    print("\nLoading data and models...")
    model, recipes_df, embeddings_list = data_loader.load_all()
    
    # Initialize components
    state_manager = get_state_manager()
    llm_handler = get_llm_handler()
    recommender = get_recommender(model, embeddings_list, recipes_df)
    
    print("\nAPI is ready!")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RA-Rec Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "/chat": "POST - Send chat message",
            "/reset": "POST - Reset conversation",
            "/state": "GET - Get current state",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "state_manager": state_manager is not None,
        "llm_handler": llm_handler is not None,
        "recommender": recommender is not None
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint
    
    Processes user message and returns bot response with state
    """
    user_message = request.message.strip()
    
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # Check for restart intent
    if any(keyword in user_message.lower() for keyword in RESTART_KEYWORDS):
        state = state_manager.reset_state()
        return ChatResponse(
            response=f"Đã reset!\n\n{GREETING_MESSAGE}",
            state=state
        )
    
    # Load current state
    state = state_manager.load_state()
    
    # Classify intent
    intents = llm_handler.classify_intent(user_message)
    
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
            state_manager.save_state(state)
            if "Provide Preference" not in intents:
                intents.append("Provide Preference")
    
    # Update state
    state = llm_handler.update_state(user_message, intents, state)
    state_manager.save_state(state)
    
    # Check if info is complete
    info_complete = state_manager.is_info_complete(state)
    
    # Handle different phases
    if "Inquire" in intents:
        # Answer question phase
        recipes_context = state_manager.get_recipes_buffer_text()
        answer = llm_handler.answer_question(user_message, recipes_context, state)
        
        return ChatResponse(
            response=answer,
            state=state
        )
    
    elif info_complete:
        # Recommendation phase
        query = recommender.generate_query_from_state(state)
        results = recommender.search_recipes(query)
        
        # Present recommendations
        presentation, recipes_data = llm_handler.present_recommendations(results.head(TOP_K_DISPLAY), state)
        
        # Save to buffer
        state_manager.add_recipes_to_buffer(recipes_data[:TOP_K_DISPLAY])
        
        # Update state with recommended items
        state["recommended_items"] = [recipe["title"] for recipe in recipes_data[:TOP_K_DISPLAY]]
        state_manager.save_state(state)
        
        return ChatResponse(
            response=presentation,
            state=state
        )
    
    else:
        # Info gathering phase
        missing_field = state_manager.get_missing_field(state)
        
        questions = {
            "type_of_food": "Bạn muốn tìm loại món gì? (ví dụ: Món kho, món luộc, món xào...)",
            "ingredients": "Bạn muốn món có nguyên liệu gì? (ví dụ: thịt lợn, hải sản, rau...)",
            "cook_time": "Bạn muốn món nấu trong bao lâu? (nếu không quan tâm hãy nói 'không')",
            "num_of_people": "Bạn muốn món ăn nấu cho bao nhiêu người? (nếu không quan tâm hãy nói 'không')",
            "calories": "Bạn quan tâm đến mức calories không? (nếu không quan tâm hãy nói 'không')",
            "algeric": "Bạn có dị ứng với thành phần nào không? (nếu không thì nói 'không')"
        }
        
        response_text = questions.get(missing_field, "Tôi cần thêm thông tin để giúp bạn.")
        
        return ChatResponse(
            response=response_text,
            state=state
        )


@app.post("/reset")
async def reset_conversation():
    """Reset conversation state"""
    state = state_manager.reset_state()
    return {
        "message": "Đã reset!",
        "state": state,
        "greeting": GREETING_MESSAGE
    }


@app.get("/state")
async def get_state():
    """Get current conversation state"""
    state = state_manager.load_state()
    return {"state": state}


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
