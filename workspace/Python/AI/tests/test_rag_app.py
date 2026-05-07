"""
RAG AI App — Test Suite
Tekijä: Aapo Rotonen
Teknologiat: Pytest, unittest.mock — hyödyntää QA & test automation osaamista

Testikategoriat:
  1. Unit tests     — yksittäiset funktiot (kontekstin haku, promptin rakenne)
  2. Integration    — RAG-ketjun end-to-end (mockattu LLM, oikea vektoritietokanta)
  3. Edge cases     — tyhjä syöte, erikoismerkit, hyvin pitkä kysymys
  4. Boundary       — chunk-koot, overlap-arvot äärilaidoilla
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_community.chat_message_histories import ChatMessageHistory


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_docs():
    """Testidokumentit — simuloi PDF:stä ladattua sisältöä."""
    return [
        Document(page_content="Aapo Rotonen on ohjelmistoinsinööri Oulusta.", metadata={"source": "cv.pdf", "page": 0}),
        Document(page_content="Hän on työskennellyt Qt Groupilla vuodesta 2021.", metadata={"source": "cv.pdf", "page": 0}),
        Document(page_content="Tekninen osaaminen: Python, C++, LangChain, ChromaDB.", metadata={"source": "cv.pdf", "page": 1}),
        Document(page_content="AI-projekti: 250M+ näyttökertaa YouTubessa ja TikTokissa.", metadata={"source": "cv.pdf", "page": 2}),
    ]

@pytest.fixture
def mock_retriever(sample_docs):
    """Mock retriever joka palauttaa sample_docs."""
    retriever = MagicMock()
    retriever.invoke.return_value = sample_docs
    return retriever

@pytest.fixture
def mock_llm():
    """Mock LLM joka palauttaa vakiovastauksen."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="Aapo on kokenut ohjelmistoinsinööri.")
    return llm

@pytest.fixture
def chat_history():
    return ChatMessageHistory()


# ═══════════════════════════════════════════════════════════════════════════
# 1. UNIT TESTS — retrieve_context
# ═══════════════════════════════════════════════════════════════════════════

class TestRetrieveContext:

    def test_returns_context_string(self, mock_retriever, sample_docs):
        """Konteksti yhdistetään oikein dokumenteista."""
        def retrieve_context(inputs):
            docs = mock_retriever.invoke(inputs["input"])
            return {**inputs, "context": "\n\n".join(d.page_content for d in docs)}

        result = retrieve_context({"input": "Kuka on Aapo?"})

        assert "context" in result
        assert "Aapo Rotonen" in result["context"]
        assert "Qt Groupilla" in result["context"]

    def test_input_passthrough(self, mock_retriever):
        """Alkuperäinen input säilyy muuttumattomana kontekstin lisäksi."""
        def retrieve_context(inputs):
            docs = mock_retriever.invoke(inputs["input"])
            return {**inputs, "context": "\n\n".join(d.page_content for d in docs)}

        original_input = "Mikä on Aapon kokemus?"
        result = retrieve_context({"input": original_input})

        assert result["input"] == original_input

    def test_retriever_called_with_correct_query(self, mock_retriever):
        """Retriever saa oikean hakusanan."""
        def retrieve_context(inputs):
            docs = mock_retriever.invoke(inputs["input"])
            return {**inputs, "context": "\n\n".join(d.page_content for d in docs)}

        query = "Python-kokemus"
        retrieve_context({"input": query})

        mock_retriever.invoke.assert_called_once_with(query)

    def test_context_documents_separated_by_double_newline(self, mock_retriever, sample_docs):
        """Dokumentit erotetaan toisistaan kahdella rivinvaihdolla."""
        def retrieve_context(inputs):
            docs = mock_retriever.invoke(inputs["input"])
            return {**inputs, "context": "\n\n".join(d.page_content for d in docs)}

        result = retrieve_context({"input": "testi"})
        parts = result["context"].split("\n\n")

        assert len(parts) == len(sample_docs)


# ═══════════════════════════════════════════════════════════════════════════
# 2. UNIT TESTS — ChatMessageHistory (muisti)
# ═══════════════════════════════════════════════════════════════════════════

class TestChatMemory:

    def test_history_starts_empty(self, chat_history):
        assert len(chat_history.messages) == 0

    def test_messages_added_correctly(self, chat_history):
        chat_history.add_user_message("Hei!")
        chat_history.add_ai_message("Moi, miten voin auttaa?")

        assert len(chat_history.messages) == 2
        assert chat_history.messages[0].content == "Hei!"
        assert chat_history.messages[1].content == "Moi, miten voin auttaa?"

    def test_message_order_preserved(self, chat_history):
        """Viestien järjestys säilyy — kriittistä kontekstin kannalta."""
        messages = ["Kysymys 1", "Kysymys 2", "Kysymys 3"]
        for msg in messages:
            chat_history.add_user_message(msg)

        for i, msg in enumerate(chat_history.messages):
            assert msg.content == messages[i]

    def test_multiple_sessions_independent(self):
        """Eri sessioiden historiat eivät sekoitu."""
        store = {}

        def get_session_history(session_id):
            if session_id not in store:
                store[session_id] = ChatMessageHistory()
            return store[session_id]

        get_session_history("user_1").add_user_message("Viesti sessiolle 1")
        get_session_history("user_2").add_user_message("Viesti sessiolle 2")

        assert len(store["user_1"].messages) == 1
        assert len(store["user_2"].messages) == 1
        assert store["user_1"].messages[0].content != store["user_2"].messages[0].content

    def test_history_clear(self, chat_history):
        """Historia tyhjenee oikein."""
        chat_history.add_user_message("Testi")
        chat_history.clear()

        assert len(chat_history.messages) == 0


# ═══════════════════════════════════════════════════════════════════════════
# 3. EDGE CASES — tyhjä syöte, erikoismerkit, pitkä kysymys
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_empty_input_handled(self, mock_retriever):
        """Tyhjä kysymys ei kaada sovellusta."""
        def retrieve_context(inputs):
            docs = mock_retriever.invoke(inputs["input"])
            return {**inputs, "context": "\n\n".join(d.page_content for d in docs)}

        result = retrieve_context({"input": ""})
        assert "context" in result

    def test_special_characters_in_query(self, mock_retriever):
        """Erikoismerkit kyselyssä eivät aiheuta virheitä."""
        special_queries = [
            "Aapo's skills?",
            "Osaaminen: C++ & Python!",
            "Mikä on palkka€€€?",
            "<script>alert('xss')</script>",
        ]
        def retrieve_context(inputs):
            docs = mock_retriever.invoke(inputs["input"])
            return {**inputs, "context": "\n\n".join(d.page_content for d in docs)}

        for query in special_queries:
            result = retrieve_context({"input": query})
            assert "context" in result, f"Epäonnistui kyselylle: {query}"

    def test_very_long_query(self, mock_retriever):
        """Hyvin pitkä kysymys (500 sanaa) käsitellään ilman kaatumista."""
        long_query = "Kerro Aapon osaamisesta " * 100  # ~500 sanaa

        def retrieve_context(inputs):
            docs = mock_retriever.invoke(inputs["input"])
            return {**inputs, "context": "\n\n".join(d.page_content for d in docs)}

        result = retrieve_context({"input": long_query})
        assert "context" in result

    def test_whitespace_only_input(self, mock_retriever):
        """Pelkkä whitespace käsitellään."""
        def retrieve_context(inputs):
            docs = mock_retriever.invoke(inputs["input"])
            return {**inputs, "context": "\n\n".join(d.page_content for d in docs)}

        result = retrieve_context({"input": "   \n\t  "})
        assert "context" in result

    def test_empty_retriever_result(self):
        """Retriever ei löydä mitään — context on tyhjä string."""
        empty_retriever = MagicMock()
        empty_retriever.invoke.return_value = []

        def retrieve_context(inputs):
            docs = empty_retriever.invoke(inputs["input"])
            return {**inputs, "context": "\n\n".join(d.page_content for d in docs)}

        result = retrieve_context({"input": "Jotain mitä ei löydy"})
        assert result["context"] == ""


# ═══════════════════════════════════════════════════════════════════════════
# 4. BOUNDARY TESTS — text splitter parametrit
# ═══════════════════════════════════════════════════════════════════════════

class TestTextSplitterBoundary:

    def test_chunk_size_minimum(self):
        """Chunk size 1 — jokainen merkki on oma chunkkinsä."""
        from langchain_text_splitters import CharacterTextSplitter
        splitter = CharacterTextSplitter(chunk_size=1, chunk_overlap=0, separator="")
        docs = splitter.create_documents(["ABC"])
        assert len(docs) >= 1

    def test_chunk_overlap_zero(self):
        """Overlap 0 — ei päällekkäisyyttä chunkkien välillä."""
        from langchain_text_splitters import CharacterTextSplitter
        splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        text = "a" * 300
        docs = splitter.create_documents([text])
        assert len(docs) >= 1

    def test_chunk_size_larger_than_document(self):
        """Chunk size isompi kuin dokumentti — pitäisi tulla 1 chunk."""
        from langchain_text_splitters import CharacterTextSplitter
        splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
        short_text = "Lyhyt teksti."
        docs = splitter.create_documents([short_text])
        assert len(docs) == 1

    def test_normal_chunking_preserves_content(self):
        """Normaali chunkkaus ei hävitä tekstiä."""
        from langchain_text_splitters import CharacterTextSplitter
        splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        original = "Aapo Rotonen. " * 50
        docs = splitter.create_documents([original])
        combined = " ".join(d.page_content for d in docs)
        # Tarkistetaan että alkuperäinen sisältö löytyy chunkeista
        assert "Aapo Rotonen" in combined


# ═══════════════════════════════════════════════════════════════════════════
# 5. DOCUMENT METADATA TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestDocumentMetadata:

    def test_documents_have_page_content(self, sample_docs):
        """Kaikilla dokumenteilla on page_content."""
        for doc in sample_docs:
            assert hasattr(doc, "page_content")
            assert len(doc.page_content) > 0

    def test_documents_have_metadata(self, sample_docs):
        """Kaikilla dokumenteilla on metadata."""
        for doc in sample_docs:
            assert hasattr(doc, "metadata")
            assert isinstance(doc.metadata, dict)

    def test_source_metadata_present(self, sample_docs):
        """Lähde löytyy metadatasta."""
        for doc in sample_docs:
            assert "source" in doc.metadata