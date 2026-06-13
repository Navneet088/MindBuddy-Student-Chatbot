# 🌱 MindBuddy — Student Mental Wellness Companion

> *An anonymous, AI-powered space for university students to de-stress, reflect, and find balance.*

---

## 📖 Table of Contents

- [About the Project](#-about-the-project)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the App](#running-the-app)
- [How It Works](#-how-it-works)
- [Configuration](#-configuration)
- [Safety & Ethics](#-safety--ethics)
- [Bug Fixes & Code Changes](#-bug-fixes--code-changes)
- [Screenshots](#-screenshots)
- [Disclaimer](#-disclaimer)
- [License](#-license)

---

## 🧠 About the Project

**MindBuddy** is a Streamlit-based mental wellness chatbot built specifically for university students navigating academic stress, anxiety, and burnout. It uses a BERT-based emotion classifier to detect the user's emotional state and routes the input through a LangChain pipeline powered by **Groq's ultra-fast LLaMA 3 inference** to generate warm, empathetic, and actionable responses.

All conversations are **fully anonymous** and **session-scoped** — nothing is stored or logged.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎭 **Emotion Detection** | Classifies emotional state (joy, sadness, anger, fear, surprise, love) using a fine-tuned BERT model |
| 💬 **Empathetic Responses** | LangChain + Groq LLaMA 3.3-70B generates context-aware, warm replies |
| 🚨 **Crisis Safety Gate** | Hard-coded keyword detection instantly redirects to the 988 Crisis Lifeline |
| 🔒 **Privacy-First** | No data is stored; conversations clear on browser reload |
| ⚡ **Fast Inference** | Powered by Groq for near-instant response generation |
| 🔑 **User-Supplied API Key** | Secure sidebar input; key is never stored server-side |

---

## 🛠 Tech Stack

```
Frontend / UI     →  Streamlit
LLM Framework     →  LangChain (langchain-groq)
LLM Provider      →  Groq Cloud (llama-3.3-70b-versatile)
Emotion Model     →  bhadresh-savani/bert-base-uncased-emotion (HuggingFace)
ML Runtime        →  HuggingFace Transformers + PyTorch
Language          →  Python 3.9+
```

---

## 📁 Project Structure

```
mindbuddy/
│
├── app.py                  # Main Streamlit application (fixed & production-ready)
├── requirements.txt        # All Python dependencies
├── README.md               # You are here
└── .streamlit/
    └── config.toml         # Optional: Streamlit theme configuration
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.9** or higher
- A free **Groq API Key** → [Get yours here](https://console.groq.com/keys)
- `pip` package manager

---

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-username/mindbuddy.git
cd mindbuddy
```

**2. Create and activate a virtual environment** *(recommended)*

```bash
# On macOS / Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**`requirements.txt` contents:**

```txt
streamlit>=1.32.0
langchain>=0.2.0
langchain-groq>=0.1.6
langchain-core>=0.2.0
transformers>=4.40.0
torch>=2.0.0
```

---

### Running the App

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

**Enter your Groq API Key** in the sidebar to activate the chatbot, and you're all set. 🎉

---

## ⚙️ How It Works

```
User types a message
        ↓
[Safety Gate] — scans for crisis keywords
        ↓ (if safe)
[Emotion Classifier] — BERT detects emotional state
        ↓
[LangChain Prompt] — emotion injected into system context
        ↓
[Groq LLaMA 3.3-70B] — generates empathetic response
        ↓
Response displayed in chat UI
```

**Detected Emotions:** `joy`, `sadness`, `anger`, `fear`, `surprise`, `love`

---

## 🔧 Configuration

You can customize MindBuddy by modifying the following constants in `app.py`:

| Variable | Location | Default | Purpose |
|---|---|---|---|
| `model` | `init_langchain_llm()` | `llama-3.3-70b-versatile` | Groq model to use |
| `temperature` | `init_langchain_llm()` | `0.6` | Response creativity (0.0–1.0) |
| `max_tokens` | `init_langchain_llm()` | `250` | Maximum response length |
| `CRISIS_KEYWORDS` | Global list | 6 patterns | Triggers emergency response |

**To change the Groq model**, replace the model string with any [Groq-supported model](https://console.groq.com/docs/models), e.g.:
```python
model="llama-3.1-8b-instant"   # Faster, lighter
model="llama-3.3-70b-versatile" # Default, most capable
```

---

## 🛡️ Safety & Ethics

- **Not a clinical tool.** MindBuddy is an AI coping companion, not a replacement for licensed therapy or emergency mental health services.
- **Crisis detection is hard-coded** and cannot be bypassed by prompt injection.
- **No PII is collected.** The app holds only an in-memory session; reloading clears everything.
- If a user expresses suicidal ideation or self-harm intent, the app immediately surfaces the **988 Suicide & Crisis Lifeline** (US) — free, confidential, 24/7.

---

## 🐛 Bug Fixes & Code Changes

The following issues were identified in the original code and corrected in `app.py`:

### 1. `@st.cache_resource` used with a dynamic argument
**Problem:** `init_langchain_llm(api_key)` was decorated with `@st.cache_resource`, but Streamlit's cache hashing can behave unexpectedly when mutable or sensitive objects (like API keys) are passed as arguments. It also risks caching a stale LLM instance if the key changes mid-session.

**Fix:** Removed the decorator from `init_langchain_llm`. The LLM is now instantiated once per session using `st.session_state`, which is the correct Streamlit pattern for stateful objects.

```python
# BEFORE (buggy)
@st.cache_resource
def init_langchain_llm(api_key):
    ...
llm = init_langchain_llm(GROQ_KEY)

# AFTER (fixed)
if "llm" not in st.session_state:
    st.session_state.llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.6,
        max_tokens=250,
        api_key=GROQ_KEY,
    )
llm = st.session_state.llm
```

---

### 2. Emotion classifier called with wrong indexing
**Problem:** `classifier(user_text)[0]` returns a list of label-score dicts when `top_k=None`. The original code assumed a nested structure that could raise an `IndexError` depending on model version.

**Fix:** Added defensive unpacking with a fallback.

```python
# BEFORE
outputs = classifier(user_text)[0]
detected_emotion = sorted(outputs, key=lambda x: x['score'], reverse=True)[0]['label']

# AFTER (fixed)
raw_output = classifier(user_text)
outputs = raw_output[0] if isinstance(raw_output[0], list) else raw_output
detected_emotion = sorted(outputs, key=lambda x: x['score'], reverse=True)[0]['label']
```

---

### 3. No guard against empty user input
**Problem:** An empty string passed to the classifier or LLM could cause an exception or a nonsensical response.

**Fix:** Added a whitespace check before processing.

```python
if not user_input.strip():
    bot_reply = "It looks like your message was empty. Feel free to share whatever is on your mind. 💚"
else:
    bot_reply = generate_langchain_response(user_input, llm)
```

---

### 4. `generate_langchain_response` did not accept `llm` as a parameter
**Problem:** The function referenced the global `llm` variable, which is an anti-pattern. If the LLM is re-instantiated, the function would silently use a stale reference.

**Fix:** `llm` is now passed as an explicit parameter.

```python
# BEFORE
def generate_langchain_response(user_text):
    ...
    chain = prompt_template | llm | StrOutputParser()

# AFTER
def generate_langchain_response(user_text, llm):
    ...
    chain = prompt_template | llm | StrOutputParser()
```

---

## 📸 Screenshots

> *(Add your own screenshots here after running the app locally)*

| Chat Interface | Sidebar Setup |
|---|---|
| `screenshot_chat.png` | `screenshot_sidebar.png` |

---

## ⚠️ Disclaimer

MindBuddy is an **experimental AI prototype** intended for educational and supportive purposes only. It is **not** a licensed mental health service, clinical diagnostic tool, or emergency response system.

If you or someone you know is in crisis:
- 📞 **988 Suicide & Crisis Lifeline** — call or text **988** (US)
- 🌐 **Crisis Text Line** — text HOME to **741741**
- 🏥 **Emergency Services** — call **911** or go to your nearest emergency room

---

## 📄 License

This project is licensed under the **MIT License**. See `LICENSE` for details.

---

<p align="center">Built with 💚 to support student wellbeing</p>
