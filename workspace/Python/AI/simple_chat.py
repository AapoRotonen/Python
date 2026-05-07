import os
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

print("Ladataan dokumenttia...")

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

store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def retrieve_context(inputs):
    docs = retriever.invoke(inputs["input"])
    return {**inputs, "context": "\n\n".join(d.page_content for d in docs)}

chain = (
    RunnableLambda(retrieve_context)
    | prompt
    | llm
    | StrOutputParser()
)

chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

SESSION = "kayttaja_1"

print("\n✅ Valmis! Voit nyt kysyä Aapon CV:stä.")
print("Kirjoita 'lopeta' tai 'exit' poistuaksesi.\n")
print("-" * 50)

while True:
    user_input = input("\nSinä: ").strip()

    if not user_input:
        continue

    if user_input.lower() in ["lopeta", "exit", "quit"]:
        print("Heippa!")
        break

    response = chain_with_memory.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": SESSION}},
    )

    print(f"\nAI: {response}")