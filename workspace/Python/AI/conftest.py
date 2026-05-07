# conftest.py — juuritason fixtures, näkyvät kaikille testeille

import sys
import os

# Lisää AI-kansio Python-polkuun jotta app.py, chat.py jne. löytyvät
sys.path.insert(0, os.path.dirname(__file__))

import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from langchain_community.chat_message_histories import ChatMessageHistory


@pytest.fixture
def sample_docs():
    return [
        Document(page_content="Aapo Rotonen on ohjelmistoinsinööri Oulusta.", metadata={"source": "cv.pdf", "page": 0}),
        Document(page_content="Hän on työskennellyt Qt Groupilla vuodesta 2021.", metadata={"source": "cv.pdf", "page": 0}),
        Document(page_content="Tekninen osaaminen: Python, C++, LangChain, ChromaDB.", metadata={"source": "cv.pdf", "page": 1}),
        Document(page_content="AI-projekti: 250M+ näyttökertaa YouTubessa ja TikTokissa.", metadata={"source": "cv.pdf", "page": 2}),
    ]


@pytest.fixture
def mock_retriever(sample_docs):
    retriever = MagicMock()
    retriever.invoke.return_value = sample_docs
    return retriever


@pytest.fixture
def mock_chain():
    """Mock RAG-ketju joka palauttaa vakiovastauksen."""
    chain = MagicMock()
    chain.invoke.return_value = "Aapo on kokenut AI-insinööri."
    return chain


@pytest.fixture
def chat_history():
    return ChatMessageHistory()


@pytest.fixture
def session_store():
    """Tyhjä session store sanakirja."""
    return {}


def make_retrieve_context(retriever):
    """Apufunktio retrieve_context-funktion luomiseen testejä varten."""
    def retrieve_context(inputs):
        docs = retriever.invoke(inputs["input"])
        return {**inputs, "context": "\n\n".join(d.page_content for d in docs)}
    return retrieve_context
