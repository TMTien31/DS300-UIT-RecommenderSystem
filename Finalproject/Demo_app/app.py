"""
Streamlit Frontend for Conversational Recipe Recommender System
Supports both Search and Conversational Chat modes
"""
import streamlit as st
import requests
import pandas as pd

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Recipe Recommender",
    layout="wide"
)


def call_api(endpoint, data=None):
    """Call API"""
    try:
        url = f"{API_URL}/{endpoint}"
        response = requests.post(url, json=data) if data else requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Lỗi API: {e}")
        return None


def parse_list_string(text):
    """Parse a string representation of a list into actual list items"""
    if not text or text == 'N/A':
        return []
    
    # Remove brackets and quotes, then split
    text = str(text).strip()
    if text.startswith('[') and text.endswith(']'):
        text = text[1:-1]
    
    # Split by quotes and comma
    import re
    items = re.findall(r"'([^']*)'", text)
    
    if not items:
        # Fallback: split by comma
        items = [item.strip().strip("'\"") for item in text.split(',')]
    
    return [item for item in items if item.strip()]


def display_recipe_card(recipe, index):
    """Display a single recipe as a card"""
    with st.expander(f"#{index}. {recipe['title']} - Score: {recipe['score']:.4f}"):
        # Basic Information
        st.markdown(f"**Loại món:** {recipe['type_of_food']}")
        st.markdown(f"**Nguồn:** {recipe['source']}")
        
        # Link if available
        if recipe.get('link') and recipe['link'] != 'N/A':
            st.markdown(f"**Link:** [{recipe['link']}]({recipe['link']})")
        
        st.divider()
        
        # Description
        st.markdown("**Mô tả:**")
        st.write(recipe['description'])
        
        st.divider()
        
        # Cooking Information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Thời gian nấu:** {recipe['cook_time']}")
        with col2:
            st.markdown(f"**Số người ăn:** {recipe['num_of_people']}")
        with col3:
            st.markdown(f"**Calories:** {recipe['calories']}")
        
        st.divider()
        
        # Ingredients - expandable
        with st.expander("Nguyên liệu:", expanded=True):
            ingredients = parse_list_string(recipe['ingredients'])
            if ingredients:
                for i, ingredient in enumerate(ingredients, 1):
                    st.markdown(f"{i}. {ingredient}")
            else:
                st.write(recipe['ingredients'])
        
        # Steps - expandable
        if recipe.get('step') and recipe['step'] != 'N/A':
            with st.expander("Các bước thực hiện:", expanded=False):
                steps = parse_list_string(recipe['step'])
                if steps:
                    for step in steps:
                        # Check if step already has "Bước X:" prefix
                        if step.strip().startswith('Bước'):
                            st.markdown(f"**{step.split(':', 1)[0]}:**")
                            if ':' in step:
                                st.write(step.split(':', 1)[1].strip())
                        else:
                            st.write(step)
                        st.write("")  # Add spacing between steps
                else:
                    st.write(recipe['step'])


def init_chat():
    """Initialize chat session"""
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Xin chào! Tôi là trợ lý gợi ý món ăn. Bạn muốn tìm món gì?"
        }]


def init_search():
    """Initialize search session"""
    if "search_results" not in st.session_state:
        st.session_state.search_results = None


# Main
st.title("Conversational Recipe Recommender System")

# Check API health
health = call_api("health")
if not health or health.get("status") != "healthy":
    st.error("API không hoạt động. Vui lòng chạy: `python api.py`")
    st.stop()

# Sidebar for mode selection
with st.sidebar:
    st.header("Cài đặt")
    
    mode = st.radio(
        "Chọn chế độ:",
        ["Search", "Conversational Chat"],
        index=0
    )
    
    st.divider()
    
    if "Search" in mode:
        st.info("**Chế độ Search:** Tìm kiếm nhanh với các thuật toán khác nhau")
    else:
        st.info("**Chế độ Chat:** Trò chuyện và nhận gợi ý cá nhân hóa")
    
    st.divider()
    st.caption("API Status: Healthy")


# ==================== SEARCH MODE ====================
if "Search" in mode:
    st.header("Tìm kiếm món ăn")
    
    init_search()
    
    # Get available algorithms
    algorithms_data = call_api("algorithms")
    if algorithms_data:
        algorithms = algorithms_data["algorithms"]
        
        # Algorithm selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_algorithm = st.selectbox(
                "Chọn thuật toán tìm kiếm:",
                algorithms,
                help="Mỗi thuật toán có cách tính similarity khác nhau"
            )
        
        with col2:
            top_k = st.number_input(
                "Số kết quả:",
                min_value=5,
                max_value=20,
                value=10,
                step=1
            )
        
        # Search query
        query = st.text_input(
            "Nhập từ khóa tìm kiếm:",
            placeholder="Ví dụ: Thịt kho nước dừa, Canh chua cá...",
            help="Nhập tên món, nguyên liệu hoặc mô tả món ăn"
        )
        
        # Search button
        if st.button("Tìm kiếm", type="primary", use_container_width=True):
            if query.strip():
                with st.spinner(f"Đang tìm kiếm với thuật toán {selected_algorithm}..."):
                    result = call_api("search", {
                        "algorithm": selected_algorithm,
                        "query": query,
                        "top_k": top_k
                    })
                
                if result:
                    st.session_state.search_results = result
                    st.rerun()
            else:
                st.warning("Vui lòng nhập từ khóa tìm kiếm")
        
        # Display results
        if st.session_state.search_results:
            results = st.session_state.search_results
            
            st.divider()
            st.subheader("Kết quả tìm kiếm")
            st.caption(f"Thuật toán: **{results['algorithm']}** | Query: **{results['query']}**")
            
            if results['results']:
                st.success(f"Tìm thấy {len(results['results'])} món ăn")
                
                # Display results as cards
                for idx, recipe in enumerate(results['results'], 1):
                    display_recipe_card(recipe, idx)
            else:
                st.warning("Không tìm thấy món ăn phù hợp")
        
        # Algorithm descriptions
        with st.expander("Thông tin về các thuật toán"):
            st.markdown("""
            **1. TF-IDF:** Content-based sử dụng text features (title + description + steps)
            
            **2. Keyword:** Tìm kiếm đơn giản dựa trên keyword matching
            
            **3. Ingredient TF-IDF:** TF-IDF tập trung vào nguyên liệu
            
            **4. SBERT + FAISS:** Semantic embeddings sử dụng Vietnamese sentence transformers
            
            **5. Hybrid TF-IDF + SBERT:** Kết hợp weighted (50% TF-IDF + 50% SBERT)
            
            **6. Hybrid General:** Kết hợp text TF-IDF và ingredient TF-IDF
            
            **7. RA-Rec (Late Fusion):** Sentence-level embeddings với average similarity
            """)


# ==================== CHAT MODE ====================
else:
    st.header("Trò chuyện với trợ lý")
    
    init_chat()
    
    # Reset button in chat mode
    if st.sidebar.button("Reset cuộc trò chuyện", use_container_width=True):
        call_api("reset", {})
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Đã reset! Bạn muốn tìm món gì?"
        }]
        st.rerun()
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Nhập tin nhắn..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Call API
        with st.spinner("Đang xử lý..."):
            result = call_api("chat", {"message": prompt})
        
        if result:
            # Add bot response
            bot_msg = result["response"]
            st.session_state.messages.append({"role": "assistant", "content": bot_msg})
            with st.chat_message("assistant"):
                st.write(bot_msg)
            st.rerun()

