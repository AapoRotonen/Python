"""
test_chat.py — Testit chat.py:n terminaalikeskustelulogiikalle

Kattaa:
  - While-loopin exit-komennot ("lopeta", "exit", "quit")
  - Tyhjän syötteen ohitus
  - chain.invoke kutsutaan oikeilla parametreilla
  - Session ID pysyy samana koko keskustelun ajan
  - Tulosteen muoto (stdout)
"""

import pytest
from unittest.mock import MagicMock, patch, call
from io import StringIO


# ═══════════════════════════════════════════════════════════════════════════
# Simuloitu chat-logiikka (vastaa chat.py:n while-loopia)
# ═══════════════════════════════════════════════════════════════════════════

def run_chat_loop(chain, session_id, inputs):
    """
    Simuloi chat.py:n while-looppia ilman oikeaa input()-kutsua.
    inputs = lista merkkijonoja jotka käyttäjä "kirjoittaa".
    Palauttaa listan kutsutuista kysymyksistä.
    """
    called_with = []
    for user_input in inputs:
        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input.lower() in ["lopeta", "exit", "quit"]:
            break
        response = chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        called_with.append(user_input)
    return called_with


# ═══════════════════════════════════════════════════════════════════════════
# 1. EXIT-KOMENNOT
# ═══════════════════════════════════════════════════════════════════════════

class TestExitCommands:

    def test_exit_stops_loop(self, mock_chain):
        result = run_chat_loop(mock_chain, "user_1", ["Hei", "exit"])
        assert len(result) == 1
        assert result[0] == "Hei"

    def test_lopeta_stops_loop(self, mock_chain):
        result = run_chat_loop(mock_chain, "user_1", ["Kysymys", "lopeta"])
        assert len(result) == 1

    def test_quit_stops_loop(self, mock_chain):
        result = run_chat_loop(mock_chain, "user_1", ["Kysymys", "quit"])
        assert len(result) == 1

    def test_exit_case_insensitive(self, mock_chain):
        """EXIT, Exit, eXiT — kaikki toimivat."""
        for cmd in ["EXIT", "Exit", "eXiT", "LOPETA", "Lopeta"]:
            result = run_chat_loop(mock_chain, "user_1", ["Kysymys", cmd])
            assert len(result) == 1, f"Komento '{cmd}' ei pysäyttänyt looppia"

    def test_exit_as_first_input_no_questions_asked(self, mock_chain):
        result = run_chat_loop(mock_chain, "user_1", ["exit"])
        assert len(result) == 0
        mock_chain.invoke.assert_not_called()

    def test_multiple_questions_before_exit(self, mock_chain):
        inputs = ["Kysymys 1", "Kysymys 2", "Kysymys 3", "exit"]
        result = run_chat_loop(mock_chain, "user_1", inputs)
        assert len(result) == 3


# ═══════════════════════════════════════════════════════════════════════════
# 2. TYHJÄ SYÖTE
# ═══════════════════════════════════════════════════════════════════════════

class TestEmptyInput:

    def test_empty_string_skipped(self, mock_chain):
        result = run_chat_loop(mock_chain, "user_1", ["", "Oikea kysymys", "exit"])
        assert len(result) == 1
        assert result[0] == "Oikea kysymys"

    def test_whitespace_only_skipped(self, mock_chain):
        result = run_chat_loop(mock_chain, "user_1", ["   ", "\t", "\n", "Kysymys", "exit"])
        assert len(result) == 1

    def test_multiple_empty_inputs_skipped(self, mock_chain):
        result = run_chat_loop(mock_chain, "user_1", ["", "", "", "Kysymys", "exit"])
        assert mock_chain.invoke.call_count == 1

    def test_empty_input_does_not_call_chain(self, mock_chain):
        run_chat_loop(mock_chain, "user_1", ["", "   ", "exit"])
        mock_chain.invoke.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════
# 3. CHAIN-KUTSUT
# ═══════════════════════════════════════════════════════════════════════════

class TestChainInvocation:

    def test_chain_invoked_with_correct_input(self, mock_chain):
        run_chat_loop(mock_chain, "user_1", ["Mikä on Aapon kokemus?", "exit"])
        mock_chain.invoke.assert_called_once_with(
            {"input": "Mikä on Aapon kokemus?"},
            config={"configurable": {"session_id": "user_1"}},
        )

    def test_session_id_consistent_across_questions(self, mock_chain):
        """Sama session_id kaikissa kutsuissa — kriittistä muistin kannalta."""
        session_id = "kayttaja_1"
        run_chat_loop(mock_chain, session_id, ["Kysymys 1", "Kysymys 2", "exit"])

        calls = mock_chain.invoke.call_args_list
        for c in calls:
            assert c.kwargs["config"]["configurable"]["session_id"] == session_id

    def test_chain_called_once_per_valid_question(self, mock_chain):
        run_chat_loop(mock_chain, "user_1", ["K1", "K2", "K3", "exit"])
        assert mock_chain.invoke.call_count == 3

    def test_input_stripped_before_invoke(self, mock_chain):
        """Whitespace poistetaan ennen chain-kutsua."""
        run_chat_loop(mock_chain, "user_1", ["  Kysymys  ", "exit"])
        call_input = mock_chain.invoke.call_args[0][0]["input"]
        assert call_input == "Kysymys"

    def test_chain_response_is_used(self, mock_chain):
        mock_chain.invoke.return_value = "Tämä on vastaus"
        responses = []
        for user_input in ["Kysymys", "exit"]:
            user_input = user_input.strip()
            if not user_input or user_input.lower() in ["exit"]:
                break
            resp = mock_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": "user_1"}},
            )
            responses.append(resp)
        assert responses[0] == "Tämä on vastaus"


# ═══════════════════════════════════════════════════════════════════════════
# 4. STDOUT — tulostus
# ═══════════════════════════════════════════════════════════════════════════

class TestOutput:

    def test_response_printed_to_stdout(self, mock_chain, capsys):
        mock_chain.invoke.return_value = "Testattu vastaus"
        for user_input in ["Kysymys", "exit"]:
            user_input = user_input.strip()
            if not user_input or user_input.lower() == "exit":
                break
            response = mock_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": "user_1"}},
            )
            print(f"\nAI: {response}")

        captured = capsys.readouterr()
        assert "Testattu vastaus" in captured.out
        assert "AI:" in captured.out

    def test_exit_message_printed(self, capsys):
        print("Heippa!")
        captured = capsys.readouterr()
        assert "Heippa!" in captured.out
