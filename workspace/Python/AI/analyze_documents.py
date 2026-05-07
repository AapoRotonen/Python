import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Lataa API-avain .env-tiedostosta
load_dotenv()

# 1. LATAA PDF
loader = PyPDFLoader("Rotonen_Aapo_CV_2026_Full.pdf")
documents = loader.load()

# 2. PILKO TEKSTI
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# 3. LUO VEKTORITIETOKANTA
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# 4. LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 5. PROMPT — sisältää chat-historian
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Käytä seuraavaa kontekstia vastataksesi kysymykseen. "
     "Jos et tiedä vastausta, sano ettet tiedä. Älä keksi omasta päästäsi.\n\n"
     "Konteksti:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),  # muisti tänne
    ("human", "{input}"),
])

# 6. SESSIOKOHTAINEN MUISTI (dict pitää historian session_id:n mukaan)
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 7. RAKENNA KETJU
def retrieve_context(inputs):
    docs = retriever.invoke(inputs["input"])
    return {**inputs, "context": "\n\n".join(d.page_content for d in docs)}

chain = (
    RunnableLambda(retrieve_context)
    | prompt
    | llm
    | StrOutputParser()
)

# 8. KÄÄRI MUISTILLA
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# 9. TESTAA — käytä samaa session_id:tä, jotta muisti säilyy
SESSION = "kayttaja_1"

def ask(question: str):
    response = chain_with_memory.invoke(
        {"input": question},
        config={"configurable": {"session_id": SESSION}},
    )
    print(f"\nKysymys: {question}")
    print(f"Vastaus: {response}")
    return response

# Ensimmäinen kysymys
ask("Miksi Aapo olisi hyvä valinta AI engineerin rooliin?")

# Toinen kysymys viittaa edelliseen — muisti toimii!
ask("Mitä muuta hänestä voi sanoa?")

# Tulosta koko historia halutessasi
print("\n--- Keskusteluhistoria ---")
for msg in store[SESSION].messages:
    role = "Human" if msg.type == "human" else "AI"
    print(f"{role}: {msg.content}")