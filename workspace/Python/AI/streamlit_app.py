import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

# ── Sivu-asetukset ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aapo Rotonen — AI Assistant",
    page_icon="🤖",
    layout="centered",
)

# ── Tyyli ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Tausta */
.stApp {
    background: #0d0d0d;
    color: #f0ece4;
}

/* Otsikko */
.hero {
    text-align: center;
    padding: 2.5rem 0 1rem 0;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    color: #f0ece4;
    letter-spacing: -0.03em;
    margin-bottom: 0.2rem;
}
.hero p {
    color: #888;
    font-size: 0.95rem;
    font-weight: 300;
}
.accent { color: #c8ff00; }

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #222;
    margin: 1rem 0 1.5rem 0;
}

/* Chat-viestit */
.msg-user {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 16px 16px 4px 16px;
    padding: 0.8rem 1.1rem;
    margin: 0.5rem 0 0.5rem 3rem;
    color: #f0ece4;
    font-size: 0.95rem;
    line-height: 1.6;
}
.msg-ai {
    background: #141414;
    border: 1px solid #222;
    border-left: 3px solid #c8ff00;
    border-radius: 4px 16px 16px 16px;
    padding: 0.8rem 1.1rem;
    margin: 0.5rem 3rem 0.5rem 0;
    color: #d8d4cc;
    font-size: 0.95rem;
    line-height: 1.6;
}
.msg-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.label-user { color: #555; }
.label-ai { color: #c8ff00; }

/* Input-alue */
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

/* Nappi */
.stButton > button {
    background: #c8ff00 !important;
    color: #0d0d0d !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.4rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
}

/* Tyhjennä-nappi */
.clear-btn > button {
    background: transparent !important;
    color: #444 !important;
    border: 1px solid #222 !important;
    font-size: 0.78rem !important;
}
.clear-btn > button:hover {
    color: #888 !important;
    border-color: #444 !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #c8ff00 !important;
}

/* Piilota Streamlitin oletuskomponentit */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── RAG-ketjun alustus (vain kerran) ────────────────────────────────────────
@st.cache_resource
def load_rag_chain():
    loader = PyPDFLoader("Rotonen_Aapo_CV_2026_Full.pdf")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Käytä seuraavaa kontekstia vastataksesi kysymykseen. "
         "Jos et tiedä vastausta, sano ettet tiedä. Älä keksi omasta päästäsi.\n\n"
         "Konteksti:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    def retrieve_context(inputs):
        retrieved = retriever.invoke(inputs["input"])
        return {**inputs, "context": "\n\n".join(d.page_content for d in retrieved)}

    chain = (
        RunnableLambda(retrieve_context)
        | prompt
        | llm
        | StrOutputParser()
    )

    store = {}

    def get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    chain_with_memory = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return chain_with_memory


# ── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit_user"


# ── UI ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Aapo Rotonen <span class="accent">AI</span></h1>
    <p>Kysy mitä tahansa Aapon CV:stä ja osaamisesta</p>
</div>
<hr class="divider">
""", unsafe_allow_html=True)

# Lataa ketju
chain = load_rag_chain()

# Näytä viestit
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="msg-user">
            <div class="msg-label label-user">Sinä</div>
            {msg["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="msg-ai">
            <div class="msg-label label-ai">AI</div>
            {msg["content"]}
        </div>
        """, unsafe_allow_html=True)

# Input + lähetä
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input(
        "kysymys",
        placeholder="Esim. Mikä on Aapon kokemus AI-projekteista?",
        label_visibility="collapsed",
        key="input_field"
    )
with col2:
    send = st.button("Lähetä")

# Tyhjennä historia
st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
if st.button("Tyhjennä keskustelu"):
    st.session_state.messages = []
    st.session_state.session_id = "streamlit_user_" + str(len(st.session_state.messages))
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# Käsittele lähetys
if send and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Ajatellaan..."):
        response = chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": st.session_state.session_id}},
        )

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()