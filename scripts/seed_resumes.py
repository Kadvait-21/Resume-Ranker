import os
from sentence_transformers import SentenceTransformer
from utils.pdf_utils import extract_text_from_filelike
from utils.pinecone_client import init_pinecone, upsert_documents

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "resume-ranker")
DATA_DIR = "data/resumes"

model = SentenceTransformer(MODEL_NAME)
index = init_pinecone(PINECONE_API_KEY, PINECONE_ENV, INDEX_NAME)

vectors = []
for fname in os.listdir(DATA_DIR):
    if not fname.lower().endswith(".pdf"):
        continue
    path = os.path.join(DATA_DIR, fname)
    with open(path, "rb") as f:
        text = extract_text_from_filelike(f)
    if not text:
        continue
    emb = model.encode(text).tolist()
    vectors.append((fname, emb, {"filename": fname}))

if vectors:
    upsert_documents(index, vectors)
    print(f"Upserted {len(vectors)} resumes to Pinecone index '{INDEX_NAME}'.")
else:
    print("No vectors to upsert.")
