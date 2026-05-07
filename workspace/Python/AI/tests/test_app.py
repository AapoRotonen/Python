"""
test_app.py — Testit app.py:n RAG-logiikalle

Kattaa:
  - retrieve_context (unit)
  - ChatMessageHistory / session store (unit)
  - CharacterTextSplitter boundary-arvot
  - Document metadata validointi
  - Edge caset: tyhjä syöte, erikoismerkit, pitkä kysymys
  - RAG-ketjun end-to-end (mockattu LLM)
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import CharacterTextSplitter
from conftest import make_retrieve_context  # root conftest löytyy automaattisesti


# ═══════════════════════════════════════════════════════════════════════════
# 1. UNIT — retrieve_context
# ═══════════════════════════════════════════════════════════════════════════

class TestRetrieveContext:

    def test_context_contains_document_content(self, mock_retriever):
        fn = make_retrieve_context(mock_retriever)
        result = fn({"input": "Kuka on Aapo?"})
        assert "Aapo Rotonen" in result["context"]
        assert "Qt Groupilla" in result["context"]

    def test_input_passthrough_unchanged(self, mock_retriever):
        fn = make_retrieve_context(mock_retriever)
        original = "Mikä on Aapon kokemus?"
        result = fn({"input": original})
        assert result["input"] == original

    def test_retriever_receives_correct_query(self, mock_retriever):
        fn = make_retrieve_context(mock_retriever)
        fn({"input": "Python-kokemus"})
        mock_retriever.invoke.assert_called_once_with("Python-kokemus")

    def test_documents_joined_with_double_newline(self, mock_retriever, sample_docs):
        fn = make_retrieve_context(mock_retriever)
        result = fn({"input": "testi"})
        parts = result["context"].split("\n\n")
        assert len(parts) == len(sample_docs)

    def test_empty_retriever_returns_empty_context(self):
        empty_retriever = MagicMock()
        empty_retriever.invoke.return_value = []
        fn = make_retrieve_context(empty_retriever)
        result = fn({"input": "kysymys"})
        assert result["context"] == ""

    def test_single_document_no_separator(self):
        single_retriever = MagicMock()
        single_retriever.invoke.return_value = [
            Document(page_content="Yksi dokumentti.", metadata={})
        ]
        fn = make_retrieve_context(single_retriever)
        result = fn({"input": "kysymys"})
        assert result["context"] == "Yksi dokumentti."
        assert "\n\n" not in result["context"]


# ═══════════════════════════════════════════════════════════════════════════
# 2. UNIT — Session store ja muisti
# ═══════════════════════════════════════════════════════════════════════════

class TestSessionStore:

    def get_session_history(self, store, session_id):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    def test_new_session_starts_empty(self, session_store):
        history = self.get_session_history(session_store, "user_1")
        assert len(history.messages) == 0

    def test_same_session_id_returns_same_history(self, session_store):
        h1 = self.get_session_history(session_store, "user_1")
        h1.add_user_message("Hei")
        h2 = self.get_session_history(session_store, "user_1")
        assert len(h2.messages) == 1

    def test_different_sessions_are_independent(self, session_store):
        self.get_session_history(session_store, "user_1").add_user_message("Sessio 1")
        self.get_session_history(session_store, "user_2").add_user_message("Sessio 2")
        assert session_store["user_1"].messages[0].content == "Sessio 1"
        assert session_store["user_2"].messages[0].content == "Sessio 2"

    def test_message_order_preserved(self, session_store):
        history = self.get_session_history(session_store, "user_1")
        messages = ["Kysymys 1", "Vastaus 1", "Kysymys 2", "Vastaus 2"]
        history.add_user_message(messages[0])
        history.add_ai_message(messages[1])
        history.add_user_message(messages[2])
        history.add_ai_message(messages[3])
        for i, msg in enumerate(history.messages):
            assert msg.content == messages[i]

    def test_history_clear_empties_messages(self, chat_history):
        chat_history.add_user_message("Testi")
        chat_history.add_ai_message("Vastaus")
        chat_history.clear()
        assert len(chat_history.messages) == 0

    def test_multiple_messages_accumulate(self, chat_history):
        for i in range(10):
            chat_history.add_user_message(f"Viesti {i}")
        assert len(chat_history.messages) == 10


# ═══════════════════════════════════════════════════════════════════════════
# 3. BOUNDARY — CharacterTextSplitter
# ═══════════════════════════════════════════════════════════════════════════

class TestTextSplitterBoundary:

    def test_chunk_size_larger_than_text_gives_one_chunk(self):
        splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
        docs = splitter.create_documents(["Lyhyt teksti."])
        assert len(docs) == 1

    def test_overlap_zero_no_duplicate_content(self):
        splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=0, separator=" ")
        text = " ".join([f"sana{i}" for i in range(100)])
        docs = splitter.create_documents([text])
        assert len(docs) > 1

    def test_content_preserved_across_chunks(self):
        splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        original = "Aapo Rotonen. " * 50
        docs = splitter.create_documents([original])
        combined = " ".join(d.page_content for d in docs)
        assert "Aapo Rotonen" in combined

    def test_empty_text_produces_no_chunks(self):
        splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        docs = splitter.create_documents([""])
        assert len(docs) == 0

    def test_chunk_overlap_smaller_than_chunk_size(self):
        """Overlap pitää aina olla pienempi kuin chunk_size."""
        with pytest.raises(Exception):
            splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=20)
            splitter.create_documents(["Testiteksti joka on tarpeeksi pitkä."])


# ═══════════════════════════════════════════════════════════════════════════
# 4. EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_empty_input_does_not_crash(self, mock_retriever):
        fn = make_retrieve_context(mock_retriever)
        result = fn({"input": ""})
        assert "context" in result

    def test_whitespace_only_input(self, mock_retriever):
        fn = make_retrieve_context(mock_retriever)
        result = fn({"input": "   \n\t  "})
        assert "context" in result

    def test_special_characters(self, mock_retriever):
        fn = make_retrieve_context(mock_retriever)
        for query in ["C++ & Python!", "palkka €€€?", "<script>alert('xss')</script>", "SQL; DROP TABLE--"]:
            result = fn({"input": query})
            assert "context" in result, f"Kaatui kyselylle: {query}"

    def test_very_long_query(self, mock_retriever):
        fn = make_retrieve_context(mock_retriever)
        long_query = "Kerro Aapon osaamisesta " * 200
        result = fn({"input": long_query})
        assert "context" in result

    def test_unicode_input(self, mock_retriever):
        fn = make_retrieve_context(mock_retriever)
        result = fn({"input": "日本語テスト 🤖 тест"})
        assert "context" in result


# ═══════════════════════════════════════════════════════════════════════════
# 5. DOCUMENT METADATA
# ═══════════════════════════════════════════════════════════════════════════

class TestDocumentMetadata:

    def test_all_docs_have_page_content(self, sample_docs):
        for doc in sample_docs:
            assert hasattr(doc, "page_content")
            assert len(doc.page_content) > 0

    def test_all_docs_have_metadata(self, sample_docs):
        for doc in sample_docs:
            assert isinstance(doc.metadata, dict)

    def test_source_present_in_metadata(self, sample_docs):
        for doc in sample_docs:
            assert "source" in doc.metadata

    def test_page_number_in_metadata(self, sample_docs):
        for doc in sample_docs:
            assert "page" in doc.metadata
            assert isinstance(doc.metadata["page"], int)


# ═══════════════════════════════════════════════════════════════════════════
# 6. END-TO-END — mockattu RAG-ketju
# ═══════════════════════════════════════════════════════════════════════════

class TestRAGChainEndToEnd:

    def test_chain_returns_answer(self, mock_chain):
        response = mock_chain.invoke({"input": "Kuka on Aapo?"})
        assert isinstance(response, str)
        assert len(response) > 0

    def test_chain_called_with_input_key(self, mock_chain):
        mock_chain.invoke({"input": "Testi"})
        mock_chain.invoke.assert_called_once_with({"input": "Testi"})

    def test_chain_handles_multiple_questions(self, mock_chain):
        questions = [
            "Mikä on Aapon kokemus?",
            "Mitä teknologioita hän osaa?",
            "Missä hän on töissä?",
        ]
        for q in questions:
            response = mock_chain.invoke({"input": q})
            assert response is not None
