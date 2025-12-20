"""
Streamlit Frontend for RA-Rec Chatbot - Simple Version
"""
import streamlit as st
import requests

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(page_title="RA-Rec Chatbot", page_icon="🍳")


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


def init():
    """Initialize session"""
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Xin chào! Tôi là trợ lý gợi ý món ăn. Bạn muốn tìm món gì?"
        }]


# Main
st.title("RA-Rec Chatbot")

init()

# Sidebar with reset button
if st.sidebar.button("Reset"):
    call_api("reset", {})
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Đã reset! Bạn muốn tìm món gì?"
    }]
    st.rerun()

# Check API health
health = call_api("health")
if not health or health.get("status") != "healthy":
    st.error("API không hoạt động. Chạy: python api.py")
    st.stop()
else:
    st.sidebar.success("API OK")

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

