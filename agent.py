"""
=============================================================
  Physics Study Buddy — Agentic AI Capstone Project
=============================================================
  Name    : Ridhum Mohan
  Roll No : 23051450
  Batch   : CSE 2023-2027
  Course  : Agentic AI Hands-On Course 2026
  Trainer : Dr. Kanthi Kiran Sirra
=============================================================

Domain   : Study Buddy — Physics (B.Tech level)
User     : B.Tech students who need concept help at odd hours
Problem  : Students need physics concept help and formula explanations
           from the course syllabus faithfully without hallucinating formulas.
Success  : Agent answers 90%+ questions faithfully (RAGAS faithfulness ≥ 0.80)
           and correctly admits when a question is out of scope.
Tool     : datetime tool — to tell students how many days until exams,
           current date/time, and schedule study reminders.
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import re
import json
import math
from datetime import datetime
from typing import TypedDict, List, Optional, Annotated
import operator

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from sentence_transformers import SentenceTransformer
import chromadb

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")   # set in environment
MODEL_NAME   = "llama-3.3-70b-versatile"
EMBED_MODEL  = "all-MiniLM-L6-v2"
TOP_K        = 3
MAX_EVAL_RETRIES = 2
FAITHFULNESS_THRESHOLD = 0.70
SLIDING_WINDOW = 6          # keep last 6 messages in history

HELPLINE = "Physics Dept Helpline: +91-XXXX-XXXXXX | Email: physicshelp@college.edu"

# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE  (10 documents, each on ONE specific topic)
# ─────────────────────────────────────────────────────────────────────────────
KNOWLEDGE_BASE = [
    {
        "id": "doc_001",
        "topic": "Newton's Laws of Motion",
        "text": (
            "Newton's Laws of Motion form the foundation of classical mechanics. "
            "The First Law (Law of Inertia) states that an object remains at rest or "
            "moves in a straight line at constant speed unless acted upon by an external "
            "net force. The Second Law states F = ma, where F is net force (Newtons), "
            "m is mass (kg), and a is acceleration (m/s²). This means acceleration is "
            "directly proportional to force and inversely proportional to mass. "
            "The Third Law states that for every action there is an equal and opposite "
            "reaction. This means forces always come in pairs — if object A exerts force "
            "on B, then B exerts an equal but opposite force on A. Newton's laws apply "
            "only in inertial (non-accelerating) reference frames."
        )
    },
    {
        "id": "doc_002",
        "topic": "Kinematics — Equations of Motion",
        "text": (
            "Kinematics studies motion without considering its causes. The four equations "
            "of motion for constant acceleration are: "
            "(1) v = u + at — final velocity equals initial velocity plus acceleration times time. "
            "(2) s = ut + ½at² — displacement equals initial velocity times time plus half "
            "acceleration times time squared. "
            "(3) v² = u² + 2as — relates velocity and displacement without time. "
            "(4) s = (u+v)/2 × t — displacement using average velocity. "
            "Here u = initial velocity, v = final velocity, a = acceleration (constant), "
            "t = time, s = displacement. For free fall, a = g = 9.8 m/s² downward. "
            "Projectile motion combines horizontal uniform motion (no force) with vertical "
            "free fall. Range R = u²sin(2θ)/g, maximum height H = u²sin²θ/2g."
        )
    },
    {
        "id": "doc_003",
        "topic": "Work, Energy, and Power",
        "text": (
            "Work is done when a force causes displacement in the direction of the force. "
            "W = F × d × cos(θ), where θ is the angle between force and displacement. "
            "Work is measured in Joules (J). "
            "Kinetic Energy (KE) is the energy of motion: KE = ½mv². "
            "Potential Energy (PE) is stored energy. Gravitational PE = mgh, where h is height. "
            "Elastic PE = ½kx², where k is spring constant and x is compression/extension. "
            "The Work-Energy Theorem states: Net work done on an object equals its change in KE. "
            "Conservation of Energy: Total mechanical energy (KE + PE) is conserved when only "
            "conservative forces act. Power P = Work/Time = F×v, measured in Watts (W). "
            "1 Watt = 1 Joule per second. Efficiency = (Useful output energy / Input energy) × 100%."
        )
    },
    {
        "id": "doc_004",
        "topic": "Gravitation and Orbital Motion",
        "text": (
            "Newton's Law of Universal Gravitation: Every two masses attract each other with force "
            "F = G×m₁×m₂/r², where G = 6.674 × 10⁻¹¹ N·m²/kg² is the gravitational constant, "
            "m₁ and m₂ are masses, and r is the distance between their centres. "
            "Acceleration due to gravity on Earth's surface g = GM/R² ≈ 9.8 m/s². "
            "g decreases with altitude: g' = g(R/(R+h))². "
            "Escape velocity is the minimum velocity to escape a planet's gravity: ve = √(2gR) ≈ 11.2 km/s for Earth. "
            "Orbital velocity for a satellite at height h: vo = √(g'(R+h)). "
            "Time period of satellite: T = 2π(R+h)/vo. "
            "Geostationary satellites orbit at ~35,786 km, completing one orbit in 24 hours."
        )
    },
    {
        "id": "doc_005",
        "topic": "Thermodynamics — Laws and Processes",
        "text": (
            "Thermodynamics deals with heat, temperature, and energy. "
            "Zeroth Law: If two systems are each in thermal equilibrium with a third, they are "
            "in equilibrium with each other — this defines temperature. "
            "First Law: Energy is conserved — ΔU = Q - W, where ΔU is change in internal energy, "
            "Q is heat added to the system, W is work done by the system. "
            "Second Law: Heat naturally flows from hot to cold. No engine is 100% efficient. "
            "Entropy of an isolated system never decreases. "
            "Third Law: As temperature approaches absolute zero (0 K), entropy approaches a minimum. "
            "Thermodynamic processes: Isothermal (constant T), Adiabatic (Q=0), Isobaric (constant P), "
            "Isochoric (constant V). Carnot efficiency = 1 - T_cold/T_hot (maximum possible efficiency)."
        )
    },
    {
        "id": "doc_006",
        "topic": "Waves and Sound",
        "text": (
            "A wave transfers energy without transferring matter. "
            "Transverse waves: oscillation perpendicular to propagation (e.g., light waves). "
            "Longitudinal waves: oscillation parallel to propagation (e.g., sound waves). "
            "Wave speed v = fλ, where f is frequency (Hz) and λ is wavelength (m). "
            "Speed of sound in air at 20°C ≈ 343 m/s. Sound cannot travel through vacuum. "
            "Amplitude determines loudness; frequency determines pitch. "
            "Doppler Effect: When source and observer move relative to each other, the observed "
            "frequency changes. f' = f×(v+v_o)/(v-v_s), where v_o is observer velocity toward source, "
            "v_s is source velocity toward observer. "
            "Resonance occurs when driving frequency matches natural frequency. "
            "Beats = difference in frequencies of two similar sounds: f_beat = |f₁ - f₂|."
        )
    },
    {
        "id": "doc_007",
        "topic": "Electrostatics — Coulomb's Law and Electric Fields",
        "text": (
            "Coulomb's Law: Force between two point charges F = kq₁q₂/r², where "
            "k = 9×10⁹ N·m²/C² (Coulomb's constant), q₁ and q₂ are charges in Coulombs, "
            "r is the separation. Like charges repel; unlike charges attract. "
            "Electric Field E = F/q₀ = kq/r² (N/C or V/m). Field lines flow from positive to negative charges. "
            "Electric Potential V = kq/r (Volts). Potential energy U = qV. "
            "Gauss's Law: Electric flux through any closed surface = Q_enclosed/ε₀, where "
            "ε₀ = 8.85×10⁻¹² F/m is permittivity of free space. "
            "Capacitance C = Q/V (Farads). Parallel plate capacitor: C = ε₀A/d. "
            "Energy stored in capacitor: U = ½CV² = Q²/2C."
        )
    },
    {
        "id": "doc_008",
        "topic": "Current Electricity — Ohm's Law and Circuits",
        "text": (
            "Electric current I = Q/t (Amperes = Coulombs/second). "
            "Ohm's Law: V = IR, where V is voltage (Volts), I is current (Amperes), R is resistance (Ohms). "
            "Resistance R = ρL/A, where ρ is resistivity, L is length, A is cross-sectional area. "
            "Power P = VI = I²R = V²/R (Watts). "
            "Series circuits: same current everywhere; R_total = R₁+R₂+R₃; V divides. "
            "Parallel circuits: same voltage across each branch; 1/R_total = 1/R₁+1/R₂+1/R₃; I divides. "
            "Kirchhoff's Current Law (KCL): Sum of currents at a junction = 0. "
            "Kirchhoff's Voltage Law (KVL): Sum of voltages around a closed loop = 0. "
            "EMF (ε) of battery: ε = V_terminal + Ir, where r is internal resistance. "
            "Wheatstone bridge is balanced when R₁/R₂ = R₃/R₄."
        )
    },
    {
        "id": "doc_009",
        "topic": "Optics — Reflection, Refraction, and Lenses",
        "text": (
            "Reflection: angle of incidence = angle of reflection (both measured from normal). "
            "Plane mirror image is virtual, erect, same size, laterally inverted, and same distance behind mirror. "
            "Refraction: Snell's Law — n₁sinθ₁ = n₂sinθ₂. Refractive index n = c/v, "
            "where c = 3×10⁸ m/s (speed of light in vacuum) and v is speed in medium. "
            "Total Internal Reflection occurs when angle of incidence exceeds critical angle θ_c = sin⁻¹(1/n). "
            "Lens formula: 1/v - 1/u = 1/f. Mirror formula: 1/v + 1/u = 1/f. "
            "Magnification m = -v/u (mirrors and lenses). "
            "Convex lens: converging, positive focal length. Concave lens: diverging, negative focal length. "
            "Power of lens P = 1/f (Dioptres). Combination: P_total = P₁ + P₂."
        )
    },
    {
        "id": "doc_010",
        "topic": "Modern Physics — Photoelectric Effect and Atomic Models",
        "text": (
            "Photoelectric Effect (Einstein, 1905): Light ejects electrons from metal surface. "
            "Photon energy E = hf = hc/λ, where h = 6.626×10⁻³⁴ J·s (Planck's constant). "
            "Maximum KE of ejected electron: KE_max = hf - φ, where φ is work function of metal. "
            "Threshold frequency f₀ = φ/h — below this, no electrons are ejected regardless of intensity. "
            "de Broglie wavelength: λ = h/mv — matter exhibits wave properties. "
            "Bohr's Model of Hydrogen Atom: electrons orbit in fixed shells; radius r_n = n²×0.529 Å. "
            "Energy of nth level: E_n = -13.6/n² eV. Emission occurs when electron falls to lower shell: "
            "ΔE = hf = E_higher - E_lower. "
            "Radioactivity: N = N₀e^(-λt), half-life T½ = 0.693/λ. "
            "Mass-energy equivalence: E = mc² (Einstein)."
        )
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# STATE DESIGN  (TypedDict — defined BEFORE any node function)
# ─────────────────────────────────────────────────────────────────────────────
class PhysicsState(TypedDict):
    question:      str
    messages:      List[dict]        # conversation history {role, content}
    route:         str               # "retrieve" | "tool" | "memory_only"
    retrieved:     str               # formatted context from ChromaDB
    sources:       List[str]         # topic names retrieved
    tool_result:   str               # output of tool node
    answer:        str               # final answer from LLM
    faithfulness:  float             # RAGAS-style faithfulness score 0.0–1.0
    eval_retries:  int               # number of eval retries attempted
    user_name:     str               # extracted user name if given

# ─────────────────────────────────────────────────────────────────────────────
# MODEL & EMBEDDINGS INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────
def build_llm() -> ChatGroq:
    return ChatGroq(model=MODEL_NAME, api_key=GROQ_API_KEY, temperature=0.2)

def build_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)

def build_chromadb(embedder: SentenceTransformer) -> chromadb.Collection:
    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name="physics_kb",
        metadata={"hnsw:space": "cosine"}
    )
    docs   = [d["text"]  for d in KNOWLEDGE_BASE]
    ids    = [d["id"]    for d in KNOWLEDGE_BASE]
    metas  = [{"topic": d["topic"]} for d in KNOWLEDGE_BASE]
    embeds = embedder.encode(docs).tolist()
    collection.add(documents=docs, embeddings=embeds, ids=ids, metadatas=metas)
    return collection

# ─────────────────────────────────────────────────────────────────────────────
# NODE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def make_memory_node(sliding_window: int = SLIDING_WINDOW):
    def memory_node(state: PhysicsState) -> PhysicsState:
        msgs = list(state.get("messages", []))
        msgs.append({"role": "user", "content": state["question"]})
        # sliding window
        msgs = msgs[-sliding_window:]
        # extract name
        name = state.get("user_name", "")
        q_lower = state["question"].lower()
        if "my name is" in q_lower:
            match = re.search(r"my name is ([a-z]+)", q_lower)
            if match:
                name = match.group(1).capitalize()
        return {**state, "messages": msgs, "user_name": name,
                "eval_retries": state.get("eval_retries", 0)}
    return memory_node


def make_router_node(llm: ChatGroq):
    def router_node(state: PhysicsState) -> PhysicsState:
        history = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in state["messages"][-4:]
        )
        prompt = f"""You are a routing agent for a Physics Study Buddy assistant.
Given the user question, decide the best route. Reply with ONE word only.

Routes:
- retrieve   → question is about a physics concept, formula, law, or topic
- tool       → question asks for current date/time, how many days until an event, or a math calculation
- memory_only → question is a greeting, thanks, general chat, or asks about the conversation itself

Conversation so far:
{history}

Current question: {state['question']}

Reply ONE word: retrieve / tool / memory_only"""

        response = llm.invoke([HumanMessage(content=prompt)])
        route = response.content.strip().lower().split()[0]
        if route not in ("retrieve", "tool", "memory_only"):
            route = "retrieve"
        return {**state, "route": route}
    return router_node


def make_retrieval_node(embedder: SentenceTransformer, collection: chromadb.Collection):
    def retrieval_node(state: PhysicsState) -> PhysicsState:
        q_embed = embedder.encode([state["question"]]).tolist()
        results = collection.query(query_embeddings=q_embed, n_results=TOP_K)
        docs    = results["documents"][0]
        metas   = results["metadatas"][0]
        sources = [m["topic"] for m in metas]
        context = "\n\n".join(
            f"[{meta['topic']}]\n{doc}" for doc, meta in zip(docs, metas)
        )
        return {**state, "retrieved": context, "sources": sources}
    return retrieval_node


def skip_retrieval_node(state: PhysicsState) -> PhysicsState:
    return {**state, "retrieved": "", "sources": []}


def tool_node(state: PhysicsState) -> PhysicsState:
    """Datetime and basic calculator tool — never raises exceptions."""
    try:
        q = state["question"].lower()
        now = datetime.now()

        # date/time query
        if any(w in q for w in ["date", "time", "today", "day", "month", "year"]):
            result = (
                f"Current date and time: {now.strftime('%A, %B %d, %Y — %I:%M %p')}\n"
                f"Day of week: {now.strftime('%A')}"
            )
        # days until calculation
        elif "days until" in q or "days left" in q or "how long" in q:
            # try to find a month mentioned
            months = {
                "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
                "july":7,"august":8,"september":9,"october":10,"november":11,"december":12
            }
            found_month = None
            for m_name, m_num in months.items():
                if m_name in q:
                    found_month = m_num
                    break
            if found_month:
                target = datetime(now.year, found_month, 1)
                if target < now:
                    target = datetime(now.year + 1, found_month, 1)
                delta = (target - now).days
                result = f"Days until {target.strftime('%B %Y')}: {delta} days"
            else:
                result = f"Today is {now.strftime('%B %d, %Y')}. Please specify a target date."
        # calculator
        elif any(op in q for op in ["+", "-", "*", "/", "^", "sqrt", "calculate", "compute"]):
            # safe eval for simple math
            expr = re.sub(r"[^0-9+\-*/().^ ]", "", state["question"])
            expr = expr.replace("^", "**")
            result = f"Calculation result: {eval(expr)}"  # noqa: S307 — restricted input
        else:
            result = (
                f"Tool response: Today is {now.strftime('%B %d, %Y')}. "
                f"For physics help, ask me about any topic in the syllabus!"
            )
    except Exception as e:
        result = f"Tool could not process request: {str(e)}. {HELPLINE}"
    return {**state, "tool_result": result}


def make_answer_node(llm: ChatGroq):
    def answer_node(state: PhysicsState) -> PhysicsState:
        name_greeting = f"Hi {state['user_name']}! " if state.get("user_name") else ""
        retries = state.get("eval_retries", 0)

        # build context section
        context_section = ""
        if state.get("retrieved"):
            context_section = f"\n\nKNOWLEDGE BASE CONTEXT:\n{state['retrieved']}"
        if state.get("tool_result"):
            context_section += f"\n\nTOOL RESULT:\n{state['tool_result']}"

        escalation = ""
        if retries > 0:
            escalation = (
                "\n\nIMPORTANT: Previous answer scored low on faithfulness. "
                "Be MORE grounded in the context. Quote formulas exactly as given. "
                "Do NOT add anything not in the context."
            )

        system_prompt = f"""You are PhysicsBot — a Study Buddy for B.Tech Physics students.
Your job: Answer ONLY from the provided context. 
If the context does not contain the answer, say clearly: 
"I don't have that specific information in my knowledge base. {HELPLINE}"
Never fabricate formulas, constants, or facts.
Be clear, educational, and step-by-step when explaining.
Keep answers focused and well-structured.{escalation}"""

        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in state["messages"][-4:]
        )

        user_prompt = f"""Conversation History:
{history_text}

Current Question: {state['question']}
{context_section}

Provide a clear, grounded answer:"""

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        answer = name_greeting + response.content.strip()
        return {**state, "answer": answer}
    return answer_node


def make_eval_node(llm: ChatGroq):
    def eval_node(state: PhysicsState) -> PhysicsState:
        # skip eval if no retrieval context
        if not state.get("retrieved"):
            return {**state, "faithfulness": 1.0}

        prompt = f"""Rate the faithfulness of this answer to the given context.
Faithfulness = does the answer use ONLY information from the context (no hallucination)?

Context:
{state['retrieved']}

Answer:
{state['answer']}

Reply with a single decimal number between 0.0 and 1.0.
0.0 = completely fabricated, 1.0 = perfectly grounded.
Reply ONLY the number."""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            score_text = response.content.strip().split()[0]
            score = float(score_text)
            score = max(0.0, min(1.0, score))
        except Exception:
            score = 0.75  # default if parsing fails

        retries = state.get("eval_retries", 0)
        if score < FAITHFULNESS_THRESHOLD:
            retries += 1

        print(f"  [eval_node] faithfulness={score:.2f} | retries={retries}")
        return {**state, "faithfulness": score, "eval_retries": retries}
    return eval_node


def save_node(state: PhysicsState) -> PhysicsState:
    msgs = list(state.get("messages", []))
    msgs.append({"role": "assistant", "content": state["answer"]})
    msgs = msgs[-SLIDING_WINDOW:]
    return {**state, "messages": msgs}

# ─────────────────────────────────────────────────────────────────────────────
# CONDITIONAL EDGE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def route_decision(state: PhysicsState) -> str:
    r = state.get("route", "retrieve")
    if r == "tool":
        return "tool"
    elif r == "memory_only":
        return "skip"
    return "retrieve"


def eval_decision(state: PhysicsState) -> str:
    score   = state.get("faithfulness", 1.0)
    retries = state.get("eval_retries", 0)
    if score < FAITHFULNESS_THRESHOLD and retries < MAX_EVAL_RETRIES:
        print(f"  [eval_decision] RETRY (score={score:.2f}, retries={retries})")
        return "answer"   # retry answer node
    print(f"  [eval_decision] PASS (score={score:.2f})")
    return "save"

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(llm: ChatGroq, embedder: SentenceTransformer,
                collection: chromadb.Collection) -> object:
    graph = StateGraph(PhysicsState)

    # nodes
    graph.add_node("memory",   make_memory_node())
    graph.add_node("router",   make_router_node(llm))
    graph.add_node("retrieve", make_retrieval_node(embedder, collection))
    graph.add_node("skip",     skip_retrieval_node)
    graph.add_node("tool",     tool_node)
    graph.add_node("answer",   make_answer_node(llm))
    graph.add_node("eval",     make_eval_node(llm))
    graph.add_node("save",     save_node)

    # entry point
    graph.set_entry_point("memory")

    # fixed edges
    graph.add_edge("memory",   "router")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("skip",     "answer")
    graph.add_edge("tool",     "answer")
    graph.add_edge("answer",   "eval")
    graph.add_edge("save",     END)

    # conditional edges
    graph.add_conditional_edges("router", route_decision,
                                {"retrieve": "retrieve", "skip": "skip", "tool": "tool"})
    graph.add_conditional_edges("eval", eval_decision,
                                {"answer": "answer", "save": "save"})

    app = graph.compile(checkpointer=MemorySaver())
    print("✅ Graph compiled successfully")
    return app


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — ask()
# ─────────────────────────────────────────────────────────────────────────────

def ask(app, question: str, thread_id: str = "student_001") -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    initial_state: PhysicsState = {
        "question":     question,
        "messages":     [],
        "route":        "",
        "retrieved":    "",
        "sources":      [],
        "tool_result":  "",
        "answer":       "",
        "faithfulness": 0.0,
        "eval_retries": 0,
        "user_name":    "",
    }
    result = app.invoke(initial_state, config=config)
    return {
        "question":     question,
        "answer":       result["answer"],
        "route":        result["route"],
        "sources":      result["sources"],
        "faithfulness": result["faithfulness"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL TEST (run before graph to verify KB)
# ─────────────────────────────────────────────────────────────────────────────

def test_retrieval(embedder: SentenceTransformer, collection: chromadb.Collection):
    test_queries = [
        "What is Newton's second law?",
        "How does the photoelectric effect work?",
        "What is Ohm's law?",
    ]
    print("\n─── Retrieval Test ───")
    for q in test_queries:
        emb = embedder.encode([q]).tolist()
        res = collection.query(query_embeddings=emb, n_results=2)
        topics = [m["topic"] for m in res["metadatas"][0]]
        print(f"  Q: {q}\n  → Topics: {topics}\n")
    print("─── Retrieval Test Complete ───\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — standalone run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🔧 Initialising Physics Study Buddy...")
    llm       = build_llm()
    embedder  = build_embedder()
    collection = build_chromadb(embedder)

    test_retrieval(embedder, collection)

    app = build_graph(llm, embedder, collection)

    # quick smoke test
    r = ask(app, "What is the formula for kinetic energy?", thread_id="demo")
    print(f"\nQ: {r['question']}\nA: {r['answer']}\nRoute: {r['route']} | Faith: {r['faithfulness']:.2f}")
