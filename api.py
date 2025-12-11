import time
import streamlit as st
from module import construct_response

st.title("🤖 Mekari Chatbot")

# 1. Initialize memory chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Shows chat histories
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(
                f"<div style='text-align: right;'>{message['content']}</div>", 
                unsafe_allow_html=True
            )
        else:
            st.markdown(message["content"])

# 3. Input User
if prompt := st.chat_input("Type your message here..."):
    with st.chat_message("user"):
        st.markdown(
            prompt,
            unsafe_allow_html=True
        )
    
    # Store message to memory
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 4. Bot response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        llm_response = construct_response(prompt)
        
        # Typing effect (answer appears word by word)
        for chunk in llm_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
    
    # Save response to memory
    st.session_state.messages.append({"role": "assistant", "content": full_response})