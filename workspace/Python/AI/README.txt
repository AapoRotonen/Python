AI Knowledge Base Agent (RAG)
Tämä projekti on Python-pohjainen tekoälysovellus, joka hyödyntää RAG (Retrieval-Augmented Generation) -arkkitehtuuria. Sovellus kykenee analysoimaan PDF-dokumentteja ja vastaamaan niitä koskeviin kysymyksiin tarkasti, välttäen tekoälylle tyypillisiä hallusinaatioita.

🚀 Ominaisuudet
Dokumenttien analysointi: Lataa ja prosessoi PDF-tiedostoja (PyPDFLoader).

Semanttinen haku: Hyödyntää OpenAI Embeddings -malleja tekstin merkityksen ymmärtämiseen.

Vektoritietokanta: Käyttää ChromaDB:tä dokumenttien tehokkaaseen tallennukseen ja hakuun.

Faktapohjaiset vastaukset: Ohjaa LLM:ää (GPT-4o) vastaamaan vain annetun kontekstin perusteella.

🛠 Teknologiapino
Kieli: Python 3.12+

Orkestraatio: LangChain

LLM: OpenAI GPT-4o / GPT-4o-mini

Tietokanta: Chroma (Vector Store)

📦 Asennus ja käyttö
Kloonaa repo ja luo virtuaaliympäristö:

Bash
python -m venv venv
source venv/Scripts/activate  # Windows
Asenna riippuvuudet:

Bash
pip install langchain langchain-openai langchain-community chromadb pypdf python-dotenv
Määritä ympäristömuuttujat:
Luo .env-tiedosto ja lisää sinne API-avaimesi:

Plaintext
OPENAI_API_KEY=your_api_key_here
Aja sovellus:

Bash
python app.py