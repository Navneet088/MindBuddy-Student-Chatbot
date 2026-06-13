import streamlit as st
import os
import re
from transformers import pipeline

# 🦜 LangChain Imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 🎨 1. Page Configuration
st.set_page_config(page_title="MindBuddy - Student Companion", page_icon="🌱", layout="centered")

st.title("🌱 MindBuddy (Powered by LangChain)")
st.caption("An anonymous space for students to de-stress and find balance.")
st.markdown("---")

# 🧠 2. Initialize Emotion Model with Caching
@st.cache_resource
def load_emotion_pipeline():
    return pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion", top_k=None)

classifier = load_emotion_pipeline()

# 🔑 3. Sidebar में API Key का Input Box देना
with st.sidebar:
    st.title("⚙️ Setup")
    # यहाँ यूजर अपनी की (Key) इनपुट करेगा
    GROQ_KEY = st.text_input("Enter your Groq API Key:", type="password", help="Get your key from Groq Console")
    
    st.markdown("---")
    st.title("🔒 Privacy First")
    st.info("Conversations are fully anonymous and are cleared completely upon reloading this browser tab.")
    st.warning("**Disclaimer:** MindBuddy is an AI coping companion, not a substitute for clinical diagnosis or emergency mental healthcare.")

# अगर यूजर ने Key नहीं डाली है, तो उसे स्क्रीन पर एक मैसेज दिखेगा और ऐप आगे नहीं बढ़ेगा
if not GROQ_KEY:
    st.warning("Please enter your Groq API Key in the sidebar to activate the chatbot. 🔑")
    st.stop()

# 🧠 4. Initialize LangChain LLM using the input key
@st.cache_resource
def init_langchain_llm(api_key):
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.6,
        max_tokens=250,
        api_key=api_key
    )

# जब यूजर की (Key) डाल देगा, तब LLM चालू होगा
llm = init_langchain_llm(GROQ_KEY)

# 🛡️ 5. Safety Check & LangChain Execution
CRISIS_KEYWORDS = [r"\bsuicide\b", r"\bkill myself\b", r"\bhurt myself\b", r"\bend my life\b", r"\bself-harm\b", r"\bdie\b"]

def generate_langchain_response(user_text):
    # Hard-coded Safety Gate
    for pattern in CRISIS_KEYWORDS:
        if re.search(pattern, user_text.lower()):
            return ("⚠️ It sounds like you are navigating a severe crisis. Please know you are not alone. "
                    "Reach out immediately to professionals who can help: Call or text **988** (Crisis Lifeline). "
                    "It is free, confidential, and available 24/7.")

    # 1. Classify Emotion
    outputs = classifier(user_text)[0]
    detected_emotion = sorted(outputs, key=lambda x: x['score'], reverse=True)[0]['label']

    # 2. Define LangChain System Prompt Template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "You are 'MindBuddy', an empathetic, supportive mental health chatbot for university students. "
            "The user is dealing with stress, anxiety, or academic burnout. Your job is to validate their feelings, "
            "show deep empathy, and offer a short, practical relaxation tip or coping mechanism.\n\n"
            "CRITICAL RULES:\n"
            "1. Never give medical advice or diagnose a condition.\n"
            "2. Keep your response brief, warm, and structured (under 4-5 sentences).\n"
            "3. Do not use fake or toxic positivity.\n\n"
            f"Context: The user's primary detected emotional state is: {detected_emotion.upper()}."
        )),
        ("user", "{student_input}")
    ])

    # 3. Create Chain
    chain = prompt_template | llm | StrOutputParser()

    try:
        response = chain.invoke({"student_input": user_text})
        return response
    except Exception as e:
        return "I'm right here with you. Take a slow, deep breath. Things might feel overwhelming right now, but we can take it step-by-step."

# 💬 6. Manage Session State & Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi there! 👋 I'm MindBuddy. Academic life can be rough sometimes. How are you holding up today?"}
    ]

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 📥 7. Process User Input
if user_input := st.chat_input("How are you feeling right now?"):
    
    # Render user chat
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Listening deeply..."):
            bot_reply = generate_langchain_response(user_input)
            st.markdown(bot_reply)
            
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})