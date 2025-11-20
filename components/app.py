import streamlit as st
import os
import time
import re
import uuid

# --- Import Our Modules ---
# Ensure your llm.py exports these functions!
from llm import run_llm_agent, get_thread_history, get_all_thread_ids
from audio_transcribe import transcribe_audio
from tts import generate_speech

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Voice Search Agent",
    page_icon="🎙️",
    layout="wide"
)

# --- Session State Initialization ---
if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_processed_audio" not in st.session_state:
    st.session_state.last_processed_audio = None

if "audio_widget_key" not in st.session_state:
    st.session_state.audio_widget_key = 1

# --- Helper Functions ---

def switch_thread(thread_id):
    """Loads a specific conversation history."""
    st.session_state.current_thread_id = thread_id
    st.session_state.messages = get_thread_history(thread_id)
    st.session_state.last_processed_audio = None
    # Increment widget key to reset audio input on thread switch
    st.session_state.audio_widget_key += 1

def create_new_chat():
    """Starts a fresh conversation thread."""
    new_id = str(uuid.uuid4())
    switch_thread(new_id)

# --- Sidebar Layout ---
with st.sidebar:
    st.header("🗂️ Conversations")
    
    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True):
        create_new_chat()
        st.rerun()
        
    st.divider()
    
    # History List (From SQLite DB)
    st.markdown("### Past Chats")
    all_threads = get_all_thread_ids()
    
    for t_id in reversed(all_threads):
        label = f"Chat {t_id[:8]}..."
        if t_id == st.session_state.current_thread_id:
            label = f"🟢 {label}"
            
        if st.button(label, key=f"btn_{t_id}", use_container_width=True):
            switch_thread(t_id)
            st.rerun()

# --- Main Page Layout ---
st.title("🎙️ AI Voice Search Agent")
st.caption(f"Session ID: {st.session_state.current_thread_id}")

# --- INPUT 1: Audio Recorder ---
st.subheader("1. Ask with your Voice")
audio_bytes = st.audio_input(
    "Record", 
    key=f"audio_input_{st.session_state.audio_widget_key}"
)

# Logic to handle NEW audio input
if audio_bytes:
    current_audio_data = audio_bytes.getvalue()
    
    # Check if this is actually new audio
    if st.session_state.last_processed_audio != current_audio_data:
        st.session_state.last_processed_audio = current_audio_data
        
        with st.spinner("Transcribing..."):
            user_text = transcribe_audio(current_audio_data)
        
        if "Error" not in user_text:
            # 1. User Message
            st.session_state.messages.append({"role": "user", "content": user_text})
            
            # 2. Agent Response (Pass Thread ID for Memory!)
            with st.spinner("Thinking..."):
                response_text = run_llm_agent(user_text, st.session_state.current_thread_id)
            
            # 3. Construct Assistant Message Dictionary
            msg_data = {"role": "assistant", "content": response_text}
            
            # Check for Image (JPG or PNG)
            match = re.search(r"(generated_image_\d+\.(jpg|png))", response_text)
            if match and os.path.exists(match.group(1)):
                msg_data["image"] = match.group(1)
            
            # Generate Audio
            with st.spinner("Speaking..."):
                audio_file = generate_speech(response_text, f"response_{int(time.time())}.mp3")
                msg_data["audio"] = audio_file
            
            # 4. Append to State
            st.session_state.messages.append(msg_data)
            
            # 5. Reset Audio Widget
            st.session_state.audio_widget_key += 1
            st.rerun()

# --- INPUT 2: Text Input ---
st.subheader("2. Type your Query")
with st.form("text_form", clear_on_submit=True):
    text_input = st.text_input("Type here...")
    submitted = st.form_submit_button("Send")

if submitted and text_input:
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": text_input})
    
    # 2. Agent Response (Pass Thread ID!)
    with st.spinner("Thinking..."):
        response_text = run_llm_agent(text_input, st.session_state.current_thread_id)
    
    # 3. Construct Assistant Message Dictionary
    msg_data = {"role": "assistant", "content": response_text}
    
    # Check for Image
    match = re.search(r"(generated_image_[\w-]+\.(jpg|png))", response_text)
    if match and os.path.exists(match.group(1)):
        msg_data["image"] = match.group(1)

    # Generate Audio
    with st.spinner("Speaking..."):
        audio_file = generate_speech(response_text, f"response_{int(time.time())}.mp3")
        msg_data["audio"] = audio_file
        
    # 4. Append to State
    st.session_state.messages.append(msg_data)
    
    # Clear audio state so user can switch back to voice easily
    st.session_state.last_processed_audio = None
    st.rerun()

# --- Display Conversation History ---
st.divider()
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])
            
            # Display Image if present
            if "image" in msg and msg["image"]:
                st.image(msg["image"], caption="Generated Image", width=400)
            
            # Display Audio if present
            if "audio" in msg and os.path.exists(msg["audio"]):
                with open(msg["audio"], "rb") as f:
                    audio_data = f.read()
                st.audio(audio_data, format="audio/mp3", start_time=0)