AI-Powered Resume Ranker

Setup:
1. Copy .env.example to .env and fill PINECONE_API_KEY and PINECONE_ENV.
2. python -m venv .venv && source .venv/bin/activate
3. pip install -r requirements.txt

Local dev:
1. Put sample PDF resumes into data/resumes/
2. python scripts/seed_resumes.py
3. streamlit run app.py

Notes:
- The pipeline uses a pre-trained SentenceTransformer for embeddings (no fine-tuning).
- Pinecone stores vectors; Streamlit provides UI.
- embed cache saved at data/embed_cache.pkl to avoid re-embedding same files.

