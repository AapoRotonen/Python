"""
hybrid_chat.py — Hybridi AI Chatbot
- Vapaa keskustelu GPT-4o:n kanssa mistä tahansa aiheesta
- Valinnainen PDF-lataus: kun dokumentti ladattu, botti vastaa sen perusteella
- Streamlit web UI, conversation memory, session management
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ── Sivuasetukset ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered",
)

# ── Tyyli ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #0d0d0d; color: #f0ece4; }

.hero {
    text-align: center;
    padding: 2rem 0 0.5rem 0;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.2rem;
    color: #f0ece4;
    letter-spacing: -0.03em;
    margin-bottom: 0.2rem;
}
.hero p { color: #888; font-size: 0.9rem; }
.accent { color: #c8ff00; }

.mode-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 0.5rem 0 1rem 0;
}
.mode-free { background: #1a2a00; color: #c8ff00; border: 1px solid #c8ff00; }
.mode-rag  { background: #002233; color: #00c8ff; border: 1px solid #00c8ff; }

.divider { border: none; border-top: 1px solid #222; margin: 0.5rem 0 1rem 0; }

.msg-user {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 16px 16px 4px 16px;
    padding: 0.8rem 1.1rem;
    margin: 0.4rem 0 0.4rem 3rem;
    color: #f0ece4;
    font-size: 0.95rem;
    line-height: 1.6;
}
.msg-ai {
    background: #141414;
    border: 1px solid #222;
    padding: 0.8rem 1.1rem;
    margin: 0.4rem 3rem 0.4rem 0;
    color: #d8d4cc;
    font-size: 0.95rem;
    line-height: 1.6;
    border-radius: 4px 16px 16px 16px;
}
.msg-ai-free  { border-left: 3px solid #c8ff00; }
.msg-ai-rag   { border-left: 3px solid #00c8ff; }

.msg-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.label-user { color: #555; }
.label-free { color: #c8ff00; }
.label-rag  { color: #00c8ff; }

.stTextInput > div > div > input {
    background: #1a1a1a !important;
    border: 1px solid #333 !important;
    border-radius: 12px !important;
    color: #f0ece4 !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 0.75rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #c8ff00 !important;
    box-shadow: 0 0 0 2px rgba(200,255,0,0.1) !important;
}

.stButton > button {
    background: #c8ff00 !important;
    color: #0d0d0d !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 1.2rem !important;
}

section[data-testid="stFileUploader"] {
    background: #141414 !important;
    border: 1px dashed #333 !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state alustus ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "hybrid_user_1"
if "chat_store" not in st.session_state:
    st.session_state.chat_store = {}
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "chain_free" not in st.session_state:
    st.session_state.chain_free = None
if "chain_rag" not in st.session_state:
    st.session_state.chain_rag = None


# ── LLM + ketjujen rakentaminen ──────────────────────────────────────────────
@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0.7)


def get_session_history(session_id: str) -> ChatMessageHistory:
    store = st.session_state.chat_store
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def build_free_chain(llm):
    """Vapaa chat — ei kontekstia, GPT-4o vastaa omasta tiedostaan."""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful, friendly, and knowledgeable AI assistant. "
         "Answer questions clearly and conversationally. "
         "If you don't know something, say so honestly."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )


def build_rag_chain(llm, vectorstore):
    """RAG-ketju — vastaa ladatun PDF:n perusteella."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer the user's question based on the document context below. "
         "If the answer is not in the context, say so clearly — don't make things up. "
         "You can also use your general knowledge to complement the document when relevant.\n\n"
         "Document context:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    def retrieve_context(inputs):
        docs = retriever.invoke(inputs["input"])
        return {**inputs, "context": "\n\n".join(d.page_content for d in docs)}

    chain = RunnableLambda(retrieve_context) | prompt | llm | StrOutputParser()
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )


def process_pdf(uploaded_file):
    """Prosessoi ladattu PDF vektoritietokantaan."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(docs, embeddings)
    os.unlink(tmp_path)
    return vectorstore


# ── Rakenna ketjut ───────────────────────────────────────────────────────────
llm = get_llm()
if st.session_state.chain_free is None:
    st.session_state.chain_free = build_free_chain(llm)


# ── UI ───────────────────────────────────────────────────────────────────────
mode = "rag" if st.session_state.vectorstore else "free"
mode_label = f"📄 PDF mode: {st.session_state.pdf_name}" if mode == "rag" else "💬 Free chat mode"
mode_class = "mode-rag" if mode == "rag" else "mode-free"

st.markdown(f"""
<div class="hero">
    <h1>AI <span class="accent">Chat</span></h1>
    <p>Ask me anything — or upload a PDF to chat with your document</p>
    <div class="mode-badge {mode_class}">{mode_label}</div>
</div>
<hr class="divider">
""", unsafe_allow_html=True)

# ── Sivupalkki: PDF-lataus ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📄 Document Mode")
    st.markdown("Upload a PDF to switch from free chat to document Q&A.")

    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded_file and uploaded_file.name != st.session_state.pdf_name:
        with st.spinner("Processing PDF..."):
            st.session_state.vectorstore = process_pdf(uploaded_file)
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.chain_rag = build_rag_chain(llm, st.session_state.vectorstore)
            st.session_state.messages.append({
                "role": "system",
                "content": f"📄 Document **{uploaded_file.name}** loaded. You can now ask questions about it."
            })
        st.rerun()

    if st.session_state.pdf_name:
        st.success(f"Active: {st.session_state.pdf_name}")
        if st.button("❌ Remove document"):
            st.session_state.vectorstore = None
            st.session_state.pdf_name = None
            st.session_state.chain_rag = None
            st.session_state.messages.append({
                "role": "system",
                "content": "📄 Document removed. Back to free chat mode."
            })
            st.rerun()

    st.markdown("---")
    if st.button("🗑 Clear conversation"):
        st.session_state.messages = []
        st.session_state.chat_store = {}
        st.session_state.session_id = "hybrid_user_" + str(os.urandom(4).hex())
        st.rerun()

# ── Viestit ──────────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="msg-user">
            <div class="msg-label label-user">You</div>
            {msg["content"]}
        </div>""", unsafe_allow_html=True)
    elif msg["role"] == "assistant":
        accent_class = "msg-ai-rag" if msg.get("mode") == "rag" else "msg-ai-free"
        label_class  = "label-rag"  if msg.get("mode") == "rag" else "label-free"
        label_text   = "AI · Document" if msg.get("mode") == "rag" else "AI · Free chat"
        st.markdown(f"""
        <div class="msg-ai {accent_class}">
            <div class="msg-label {label_class}">{label_text}</div>
            {msg["content"]}
        </div>""", unsafe_allow_html=True)
    elif msg["role"] == "system":
        st.info(msg["content"])

# ── Input ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([5, 1])
with col1:
    placeholder = "Ask about the document..." if mode == "rag" else "Ask me anything..."
    user_input = st.text_input("input", placeholder=placeholder,
                               label_visibility="collapsed", key="chat_input")
with col2:
    send = st.button("Send")

# ── Lähetys ──────────────────────────────────────────────────────────────────
if send and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input})

    current_mode = "rag" if st.session_state.vectorstore else "free"
    chain = st.session_state.chain_rag if current_mode == "rag" else st.session_state.chain_free

    with st.spinner("Thinking..."):
        response = chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": st.session_state.session_id}},
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "mode": current_mode,
    })
    st.rerun()