"""
=============================================================
  Physics Study Buddy — Streamlit UI
=============================================================
  Name    : Ridhum Mohan
  Roll No : 23051450
  Batch   : CSE 2023-2027
=============================================================
"""

import streamlit as st
import uuid
from agent import build_llm, build_embedder, build_chromadb, build_graph, ask

# ─── Page config ─────────────────────────────────────────
st.set_page_config(
    page_title="Physics Study Buddy",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}
.main { background: #0d0d1a; }
.stApp { background: linear-gradient(135deg, #0d0d1a 0%, #111128 100%); }

/* Chat bubbles */
.user-bubble {
    background: linear-gradient(135deg, #1a1a3e, #2d2d5e);
    border: 1px solid #4444aa;
    border-radius: 18px 18px 4px 18px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #e0e0ff;
    max-width: 80%;
    margin-left: auto;
    font-size: 15px;
}
.bot-bubble {
    background: linear-gradient(135deg, #0a2a1a, #0d3a24);
    border: 1px solid #22aa66;
    border-radius: 18px 18px 18px 4px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #d0ffe8;
    max-width: 85%;
    font-size: 15px;
    line-height: 1.7;
}
.meta-tag {
    font-size: 11px;
    color: #888;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 6px;
}
.header-card {
    background: linear-gradient(135deg, #1a0a2e, #0a1a2e);
    border: 1px solid #5533aa;
    border-radius: 16px;
    padding: 20px 28px;
    margin-bottom: 20px;
    text-align: center;
}
.header-card h1 { color: #aa88ff; font-size: 2rem; margin: 0; }
.header-card p  { color: #8899cc; margin: 4px 0 0 0; font-size: 0.9rem; }
.topic-pill {
    display: inline-block;
    background: #1a1a3e;
    border: 1px solid #4444aa;
    border-radius: 20px;
    padding: 4px 12px;
    margin: 3px;
    font-size: 12px;
    color: #aaaaff;
}
.faith-bar {
    height: 6px;
    border-radius: 3px;
    margin-top: 4px;
    background: linear-gradient(90deg, #22aa66, #55ff99);
}
</style>
""", unsafe_allow_html=True)


# ─── Resource caching (expensive inits run once) ─────────
@st.cache_resource
def load_agent():
    llm        = build_llm()
    embedder   = build_embedder()
    collection = build_chromadb(embedder)
    app        = build_graph(llm, embedder, collection)
    return app

# ─── Session state init ───────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

# ─── Sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:10px 0 20px 0;">
        <div style="font-size:3rem;">⚛️</div>
        <div style="color:#aa88ff; font-size:1.1rem; font-weight:700;">Physics Study Buddy</div>
        <div style="color:#666; font-size:0.75rem;">Agentic AI Capstone 2026</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("**📚 Topics Covered**")
    topics = [
        "Newton's Laws", "Kinematics", "Work & Energy",
        "Gravitation", "Thermodynamics", "Waves & Sound",
        "Electrostatics", "Current Electricity", "Optics",
        "Modern Physics"
    ]
    for t in topics:
        st.markdown(f'<span class="topic-pill">{t}</span>', unsafe_allow_html=True)

    st.divider()
    st.markdown("**💡 Sample Questions**")
    samples = [
        "Explain Newton's 3 laws with examples",
        "What is the formula for escape velocity?",
        "How does the photoelectric effect work?",
        "What is Snell's law in optics?",
        "What day is today?",
    ]
    for s in samples:
        if st.button(s, key=f"sample_{s[:15]}", use_container_width=True):
            st.session_state.pending_question = s

    st.divider()
    if st.button("🔄 New Conversation", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()

    st.divider()
    st.markdown(f"""
    <div style="font-size:11px; color:#555; text-align:center;">
        Session ID: <code>{st.session_state.thread_id}</code><br>
        <b>Ridhum Mohan</b> | 23051450<br>
        CSE 2023-2027
    </div>
    """, unsafe_allow_html=True)


# ─── Main header ─────────────────────────────────────────
st.markdown("""
<div class="header-card">
    <h1>⚛️ Physics Study Buddy</h1>
    <p>Your 24/7 AI assistant for B.Tech Physics — powered by LangGraph + ChromaDB + Groq</p>
</div>
""", unsafe_allow_html=True)

# ─── Chat history ─────────────────────────────────────────
chat_area = st.container()
with chat_area:
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center; color:#555; padding:60px 20px;">
            <div style="font-size:3rem; margin-bottom:12px;">🎓</div>
            <div style="font-size:1.1rem;">Ask me anything from your Physics syllabus!</div>
            <div style="font-size:0.85rem; margin-top:8px;">Formulas · Laws · Concepts · Derivations</div>
        </div>
        """, unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-bubble">🎓 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            faith_pct = int(msg.get("faithfulness", 1.0) * 100)
            route = msg.get("route", "")
            sources = ", ".join(msg.get("sources", [])) or "—"
            st.markdown(f"""
            <div class="bot-bubble">
                ⚛️ {msg["content"]}
                <div class="meta-tag">
                    Route: <b>{route}</b> | Sources: {sources} | Faithfulness: {faith_pct}%
                    <div class="faith-bar" style="width:{faith_pct}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ─── Input ───────────────────────────────────────────────
app = load_agent()

# Handle sample button click
question = None
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

user_input = st.chat_input("Ask a physics question...", key="chat_input")
if user_input:
    question = user_input

if question:
    # show user message immediately
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("⚛️ Thinking..."):
        result = ask(app, question, thread_id=st.session_state.thread_id)

    st.session_state.messages.append({
        "role":        "assistant",
        "content":     result["answer"],
        "route":       result["route"],
        "sources":     result["sources"],
        "faithfulness": result["faithfulness"],
    })
    st.rerun()
