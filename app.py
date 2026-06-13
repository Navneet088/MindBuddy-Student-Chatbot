import streamlit as st
import os
import re
from transformers import pipeline

# 🦜 LangChain Imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ─────────────────────────────────────────────
# 🎨 1. Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MindBuddy - Student Companion",
    page_icon="🌱",
    layout="centered",
)

st.title("🌱 MindBuddy (Powered by LangChain)")
st.caption("An anonymous space for students to de-stress and find balance.")
st.markdown("---")

# ─────────────────────────────────────────────
# 🧠 2. Initialize Emotion Model with Caching
# ─────────────────────────────────────────────
@st.cache_resource
def load_emotion_pipeline():
    """Load and cache the BERT emotion classifier."""
    return pipeline(
        "text-classification",
        model="bhadresh-savani/bert-base-uncased-emotion",
        top_k=None,
    )

classifier = load_emotion_pipeline()

# ─────────────────────────────────────────────
# 🔑 3. Sidebar: API Key Input & Privacy Info
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Setup")
    GROQ_KEY = st.text_input(
        "Enter your Groq API Key:",
        type="password",
        help="Get your free key from https://console.groq.com/keys",
    )
    st.markdown("---")
    st.title("🔒 Privacy First")
    st.info(
        "Conversations are fully anonymous and are cleared completely "
        "upon reloading this browser tab."
    )
    st.warning(
        "**Disclaimer:** MindBuddy is an AI coping companion, not a substitute "
        "for clinical diagnosis or emergency mental healthcare."
    )

# Guard: stop the app if no API key has been entered yet
if not GROQ_KEY:
    st.warning(
        "Please enter your Groq API Key in the sidebar to activate the chatbot. 🔑"
    )
    st.stop()

# ─────────────────────────────────────────────
# 🧠 4. Initialize LangChain LLM (session-scoped)
#
# FIX: Removed @st.cache_resource from the LLM initializer.
# Using st.session_state is the correct Streamlit pattern for
# stateful objects that depend on user-supplied values (like API keys).
# @st.cache_resource with dynamic arguments can return stale instances
# if the key changes mid-session.
# ─────────────────────────────────────────────
if "llm" not in st.session_state or st.session_state.get("groq_key_used") != GROQ_KEY:
    st.session_state.llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.6,
        max_tokens=250,
        api_key=GROQ_KEY,
    )
    st.session_state.groq_key_used = GROQ_KEY

llm = st.session_state.llm

# ─────────────────────────────────────────────
# 🛡️ 5. Crisis Keywords (Safety Gate)
# ─────────────────────────────────────────────
CRISIS_KEYWORDS = [
    r"\bsuicide\b",
    r"\bkill myself\b",
    r"\bhurt myself\b",
    r"\bend my life\b",
    r"\bself-harm\b",
    r"\bself harm\b",
    r"\bwant to die\b",
    r"\bdie\b",
]

CRISIS_RESPONSE = (
    "⚠️ It sounds like you are navigating a severe crisis. "
    "Please know you are not alone. Reach out immediately to professionals who can help:\n\n"
    "📞 **Call or text 988** — Suicide & Crisis Lifeline (free, confidential, 24/7)\n"
    "💬 **Text HOME to 741741** — Crisis Text Line\n"
    "🏥 Or go to your **nearest emergency room**.\n\n"
    "You matter, and support is available right now."
)

# ─────────────────────────────────────────────
# 💡 6. Core Response Function
#
# FIX 1: `llm` is now passed as an explicit parameter instead of
#         relying on a global reference (safer, more testable).
#
# FIX 2: Emotion classifier output is unpacked defensively.
#         `pipeline(..., top_k=None)` may return [[{...}]] or [{...}]
#         depending on the transformers version. Both are handled.
# ─────────────────────────────────────────────
def generate_langchain_response(user_text: str, llm: ChatGroq) -> str:
    """
    Run the safety gate → emotion detection → LangChain pipeline
    and return the assistant's reply as a string.
    """

    # — Safety Gate —
    for pattern in CRISIS_KEYWORDS:
        if re.search(pattern, user_text.lower()):
            return CRISIS_RESPONSE

    # — Emotion Detection (defensive unpacking) —
    try:
        raw_output = classifier(user_text)
        # raw_output is either [[{label, score}, ...]] or [{label, score}, ...]
        outputs = raw_output[0] if isinstance(raw_output[0], list) else raw_output
        detected_emotion = sorted(outputs, key=lambda x: x["score"], reverse=True)[0]["label"]
    except Exception:
        detected_emotion = "neutral"

    # — LangChain Prompt + Chain —
    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are 'MindBuddy', an empathetic, supportive mental health chatbot for "
                "university students. The user is dealing with stress, anxiety, or academic burnout. "
                "Your job is to validate their feelings, show deep empathy, and offer a short, "
                "practical relaxation tip or coping mechanism.\n\n"
                "CRITICAL RULES:\n"
                "1. Never give medical advice or diagnose a condition.\n"
                "2. Keep your response brief, warm, and structured (under 4–5 sentences).\n"
                "3. Do not use fake or toxic positivity.\n\n"
                f"Context: The user's primary detected emotional state is: {detected_emotion.upper()}."
            ),
        ),
        ("user", "{student_input}"),
    ])

    chain = prompt_template | llm | StrOutputParser()

    try:
        return chain.invoke({"student_input": user_text})
    except Exception:
        return (
            "I'm right here with you. 💚 Take a slow, deep breath. "
            "Things might feel overwhelming right now, but we can work through it step-by-step."
        )


# ─────────────────────────────────────────────
# 💬 7. Session State: Chat History
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi there! 👋 I'm MindBuddy. Academic life can be rough sometimes. "
                "How are you holding up today?"
            ),
        }
    ]

# Display full chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# 📥 8. Process User Input

if user_input := st.chat_input("How are you feeling right now?"):

    # Render user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Guard: skip processing for blank/whitespace-only input
    if not user_input.strip():
        bot_reply = (
            "It looks like your message was empty. "
            "Feel free to share whatever is on your mind. 💚"
        )
    else:
        with st.chat_message("assistant"):
            with st.spinner("Listening deeply..."):
                bot_reply = generate_langchain_response(user_input, llm)
                st.markdown(bot_reply)

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
