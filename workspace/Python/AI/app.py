import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

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

# 4. RAKENNA RAG-KETJU (Uusi tapa)
llm = ChatOpenAI(model="gpt-4o", temperature=0)

system_prompt = (
    "Käytä seuraavaa kontekstia vastataksesi kysymykseen. "
    "Jos et tiedä vastausta, sano ettet tiedä. Älä keksi omasta päästäsi."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(vectorstore.as_retriever(), question_answer_chain)

# 5. TESTAA
query = "Miksi Aapo olisi hyvä valinta AI engineerin rooliin?"
response = rag_chain.invoke({"input": query})
print("\nVastaus PDF:n perusteella:")
print(response["answer"])