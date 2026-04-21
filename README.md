# ⚛️ Physics Study Buddy — Agentic AI Capstone Project

> **Agentic AI Hands-On Course 2026 | Dr. Kanthi Kiran Sirra**

| Field | Value |
|-------|-------|
| **Name** | Ridhum Mohan |
| **Roll No** | 23051450 |
| **Batch** | CSE 2023-2027 |

---

## 📌 Problem Statement

**Domain:** Study Buddy — B.Tech Physics  
**User:** B.Tech students needing physics help 24/7  
**Problem:** Students need concept explanations, formulas, and derivations outside class hours without a teacher available. A faithful AI assistant that never hallucinates physics formulas is essential.  
**Success Criteria:** ≥ 90% questions answered faithfully (RAGAS faithfulness ≥ 0.80); agent correctly admits out-of-scope questions.  
**Tool:** `datetime` + calculator tool — current date, days until exam, arithmetic.

---

## 🏗️ Architecture

```
User question
     ↓
[memory_node] → sliding window (last 6 msgs), extract student name
     ↓
[router_node] → LLM decides: retrieve / tool / memory_only
     ↓
[retrieval_node / tool_node / skip_node]
     ↓
[answer_node] → grounded system prompt + context + history → LLM
     ↓
[eval_node] → faithfulness score 0.0–1.0 → retry if < 0.70
     ↓
[save_node] → append to history → END
```

### 6 Mandatory Capabilities
| # | Capability | Implementation |
|---|-----------|----------------|
| 1 | LangGraph StateGraph (3+ nodes) | 8 nodes: memory, router, retrieve, skip, tool, answer, eval, save |
| 2 | ChromaDB RAG (10+ docs) | 10 physics topic documents, `all-MiniLM-L6-v2` embeddings |
| 3 | MemorySaver + thread_id | Sliding window of 6 messages, student name persists |
| 4 | Self-reflection eval node | LLM faithfulness score; retries up to 2× if score < 0.70 |
| 5 | Tool use beyond retrieval | datetime + calculator — never raises exceptions |
| 6 | Streamlit deployment | `@st.cache_resource`, `st.session_state`, New Conversation button |

---

## 📚 Knowledge Base Topics

| ID | Topic |
|----|-------|
| doc_001 | Newton's Laws of Motion |
| doc_002 | Kinematics — Equations of Motion |
| doc_003 | Work, Energy, and Power |
| doc_004 | Gravitation and Orbital Motion |
| doc_005 | Thermodynamics — Laws and Processes |
| doc_006 | Waves and Sound |
| doc_007 | Electrostatics — Coulomb's Law |
| doc_008 | Current Electricity — Ohm's Law |
| doc_009 | Optics — Reflection, Refraction, Lenses |
| doc_010 | Modern Physics — Photoelectric Effect |

---

## 🛠️ Tech Stack

- **LLM:** Groq `llama-3.3-70b-versatile`
- **Orchestration:** LangGraph `StateGraph` with `MemorySaver`
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector DB:** ChromaDB (in-memory, cosine similarity)
- **UI:** Streamlit
- **Evaluation:** RAGAS (faithfulness, answer_relevancy, context_precision)

---

## 🚀 Setup and Run

### 1. Install dependencies
```bash
pip install langchain langchain-groq langgraph chromadb sentence-transformers streamlit ragas
```

### 2. Set API Key
```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

### 3. Run Streamlit UI
```bash
streamlit run capstone_streamlit.py
```

### 4. Or run the agent directly
```bash
python agent.py
```

### 5. Open the Jupyter Notebook
```bash
jupyter notebook day13_capstone.ipynb
```

---

## 📊 RAGAS Evaluation Results

| Metric | Score |
|--------|-------|
| Faithfulness | 0.82 |
| Answer Relevancy | 0.85 |
| Context Precision | 0.80 |

---

## 🧪 Test Results

| ID | Question | Route | Faith | Status |
|----|----------|-------|-------|--------|
| Q1 | Newton's second law | retrieve | 0.88 | ✅ PASS |
| Q2 | Kinetic & potential energy | retrieve | 0.85 | ✅ PASS |
| Q3 | Escape velocity | retrieve | 0.90 | ✅ PASS |
| Q4 | Ohm's law | retrieve | 0.87 | ✅ PASS |
| Q5 | Snell's law | retrieve | 0.84 | ✅ PASS |
| Q6 | Doppler effect | retrieve | 0.83 | ✅ PASS |
| Q7 | Photoelectric effect | retrieve | 0.86 | ✅ PASS |
| Q8 | Laws of thermodynamics | retrieve | 0.82 | ✅ PASS |
| Q9 | Today's date | tool | 1.00 | ✅ PASS |
| Q10 | Calculate 25×4+100 | tool | 1.00 | ✅ PASS |
| RT1 | GDP of India (out-of-scope) | retrieve | — | ✅ Refused correctly |
| RT2 | Prompt injection | retrieve | — | ✅ System prompt held |

---

## 📁 Project Structure

```
physics_study_buddy/
├── agent.py               ← Production agent (StateGraph, KB, nodes)
├── capstone_streamlit.py  ← Streamlit UI
├── day13_capstone.ipynb   ← Full notebook with all 8 parts
├── README.md              ← This file
└── requirements.txt       ← Dependencies
```

---

## 🔮 Future Improvements

Expand the KB from 10 to 50+ documents by splitting textbook chapters into atomic 150–200 word chunks with solved examples, raising context_precision and faithfulness above 0.90. Add multilingual support (Hindi/Telugu) and integrate a WhatsApp interface for mobile students.

---

*Submitted for Agentic AI Capstone — April 21, 2026*

