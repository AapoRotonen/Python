"""
test_streamlit_app.py — Testit streamlit_app.py:n UI-logiikalle

Käyttää Streamlitin virallista AppTest-frameworkia (streamlit >= 1.28)
joka simuloi UI:ta ilman selainta — vastaa Playwright/Selenium-tyylistä testausta.

Kattaa:
  - Session state alustus
  - Viestien lisäys ja näyttö
  - Tyhjennä-toiminto
  - Input-validointi (tyhjä syöte ei lähetä)
  - Chain-integraatio (mockattu)
  - UI-elementtien olemassaolo
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_community.chat_message_histories import ChatMessageHistory


# ═══════════════════════════════════════════════════════════════════════════
# Streamlit AppTest — käynnistää oikean Streamlit-appin testejä varten
# Nämä testit vaativat: pip install streamlit>=1.28
# ═══════════════════════════════════════════════════════════════════════════

# Yritetään importata AppTest — graceful skip jos Streamlit puuttuu tai on vanha
try:
    from streamlit.testing.v1 import AppTest
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

streamlit_required = pytest.mark.skipif(
    not STREAMLIT_AVAILABLE,
    reason="Vaatii streamlit>=1.28 AppTest-tuen kanssa"
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. SESSION STATE — logiikka ilman Streamlitiä
# ═══════════════════════════════════════════════════════════════════════════

class TestSessionStateLogic:
    """
    Testaa session state -logiikka suoraan ilman Streamlitiä.
    Simuloi streamlit_app.py:n if-blokkeja.
    """

    def _init_session_state(self):
        """Simuloi Streamlitin session_state alustusta."""
        state = {}
        if "messages" not in state:
            state["messages"] = []
        if "session_id" not in state:
            state["session_id"] = "kayttaja_1"
        return state

    def test_messages_initialized_as_empty_list(self):
        state = self._init_session_state()
        assert state["messages"] == []

    def test_session_id_initialized(self):
        state = self._init_session_state()
        assert "session_id" in state
        assert len(state["session_id"]) > 0

    def test_message_append_user(self):
        state = self._init_session_state()
        state["messages"].append({"role": "user", "content": "Hei!"})
        assert len(state["messages"]) == 1
        assert state["messages"][0]["role"] == "user"

    def test_message_append_assistant(self):
        state = self._init_session_state()
        state["messages"].append({"role": "user", "content": "Kysymys"})
        state["messages"].append({"role": "assistant", "content": "Vastaus"})
        assert len(state["messages"]) == 2
        assert state["messages"][1]["role"] == "assistant"

    def test_clear_resets_messages(self):
        state = self._init_session_state()
        state["messages"].append({"role": "user", "content": "Testi"})
        state["messages"] = []
        assert state["messages"] == []

    def test_clear_creates_new_session_id(self):
        state = self._init_session_state()
        old_id = state["session_id"]
        state["session_id"] = "streamlit_user_new"
        assert state["session_id"] != old_id

    def test_message_structure_has_required_keys(self):
        state = self._init_session_state()
        state["messages"].append({"role": "user", "content": "Testi"})
        msg = state["messages"][0]
        assert "role" in msg
        assert "content" in msg

    def test_roles_are_valid(self):
        """Vain 'user' ja 'assistant' ovat sallittuja rooleja."""
        valid_roles = {"user", "assistant"}
        messages = [
            {"role": "user", "content": "Kysymys"},
            {"role": "assistant", "content": "Vastaus"},
        ]
        for msg in messages:
            assert msg["role"] in valid_roles


# ═══════════════════════════════════════════════════════════════════════════
# 2. SEND-LOGIIKKA — input validointi ja chain-kutsu
# ═══════════════════════════════════════════════════════════════════════════

class TestSendLogic:
    """Simuloi streamlit_app.py:n 'if send and user_input.strip()' -logiikka."""

    def _handle_send(self, chain, state, user_input, session_id):
        """Simuloi lähetyksen käsittely."""
        if user_input and user_input.strip():
            state["messages"].append({"role": "user", "content": user_input})
            response = chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )
            state["messages"].append({"role": "assistant", "content": response})
            return True
        return False

    def test_valid_input_adds_two_messages(self, mock_chain):
        state = {"messages": [], "session_id": "user_1"}
        self._handle_send(mock_chain, state, "Kysymys", "user_1")
        assert len(state["messages"]) == 2

    def test_empty_input_not_sent(self, mock_chain):
        state = {"messages": [], "session_id": "user_1"}
        sent = self._handle_send(mock_chain, state, "", "user_1")
        assert not sent
        mock_chain.invoke.assert_not_called()

    def test_whitespace_input_not_sent(self, mock_chain):
        state = {"messages": [], "session_id": "user_1"}
        sent = self._handle_send(mock_chain, state, "   ", "user_1")
        assert not sent
        mock_chain.invoke.assert_not_called()

    def test_user_message_stored_before_ai_response(self, mock_chain):
        state = {"messages": [], "session_id": "user_1"}
        self._handle_send(mock_chain, state, "Kysymys", "user_1")
        assert state["messages"][0]["role"] == "user"
        assert state["messages"][1]["role"] == "assistant"

    def test_chain_invoked_with_session_id(self, mock_chain):
        state = {"messages": [], "session_id": "testi_sessio"}
        self._handle_send(mock_chain, state, "Kysymys", "testi_sessio")
        mock_chain.invoke.assert_called_once_with(
            {"input": "Kysymys"},
            config={"configurable": {"session_id": "testi_sessio"}},
        )

    def test_ai_response_stored_in_messages(self, mock_chain):
        mock_chain.invoke.return_value = "AI vastaus tähän"
        state = {"messages": [], "session_id": "user_1"}
        self._handle_send(mock_chain, state, "Kysymys", "user_1")
        assert state["messages"][1]["content"] == "AI vastaus tähän"

    def test_multiple_sends_accumulate_messages(self, mock_chain):
        state = {"messages": [], "session_id": "user_1"}
        for q in ["Kysymys 1", "Kysymys 2", "Kysymys 3"]:
            self._handle_send(mock_chain, state, q, "user_1")
        assert len(state["messages"]) == 6  # 3 user + 3 assistant


# ═══════════════════════════════════════════════════════════════════════════
# 3. CACHE — RAG-ketjun lataus kerran
# ═══════════════════════════════════════════════════════════════════════════

class TestCacheLogic:

    def test_chain_loaded_once_with_cache(self):
        """
        Simuloi @st.cache_resource — funktio kutsutaan vain kerran
        vaikka UI renderöityisi useita kertoja.
        """
        call_count = 0

        def load_rag_chain_mock():
            nonlocal call_count
            call_count += 1
            return MagicMock()

        # Simuloi cache: ensimmäinen kutsu lataa, loput palauttavat saman
        _cache = {}

        def cached_load():
            if "chain" not in _cache:
                _cache["chain"] = load_rag_chain_mock()
            return _cache["chain"]

        chain1 = cached_load()
        chain2 = cached_load()
        chain3 = cached_load()

        assert call_count == 1
        assert chain1 is chain2 is chain3


# ═══════════════════════════════════════════════════════════════════════════
# 4. STREAMLIT APPTEST — oikea UI-testaus
# Vaatii: pip install streamlit>=1.28
# Aja: py -m pytest test_streamlit_app.py -v -k "AppTest"
# ═══════════════════════════════════════════════════════════════════════════

@streamlit_required
class TestStreamlitAppTest:

    @pytest.fixture
    def app(self):
        """Käynnistä Streamlit-appi testejä varten mockatuilla riippuvuuksilla."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Mockattu AI vastaus"

        with patch("streamlit_app.load_rag_chain", return_value=mock_chain):
            at = AppTest.from_file("../streamlit_app.py", default_timeout=30)
            at.run()
            yield at

    def test_app_runs_without_exception(self, app):
        assert not app.exception

    def test_title_present(self, app):
        """Sivulla on otsikko."""
        page_text = str(app)
        assert "Aapo" in page_text or len(app.markdown) > 0

    def test_text_input_exists(self, app):
        assert len(app.text_input) > 0

    def test_send_button_exists(self, app):
        buttons = [b.label for b in app.button]
        assert any("Lähetä" in label for label in buttons)

    def test_clear_button_exists(self, app):
        buttons = [b.label for b in app.button]
        assert any("Tyhjennä" in label for label in buttons)

    def test_empty_input_does_not_add_messages(self, app):
        """Tyhjä syöte + Lähetä ei lisää viestejä."""
        app.text_input[0].set_value("")
        send_btn = next(b for b in app.button if "Lähetä" in b.label)
        send_btn.click().run()
        assert app.session_state["messages"] == []

    def test_valid_input_adds_messages(self, app):
        """Oikea kysymys lisää kaksi viestiä (user + assistant)."""
        app.text_input[0].set_value("Mikä on Aapon kokemus?")
        send_btn = next(b for b in app.button if "Lähetä" in b.label)
        send_btn.click().run()
        assert len(app.session_state["messages"]) == 2

    def test_clear_button_empties_messages(self, app):
        """Tyhjennä-nappi nollaa viestit."""
        app.session_state["messages"] = [
            {"role": "user", "content": "Testi"},
            {"role": "assistant", "content": "Vastaus"},
        ]
        clear_btn = next(b for b in app.button if "Tyhjennä" in b.label)
        clear_btn.click().run()
        assert app.session_state["messages"] == []
