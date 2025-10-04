import os
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from utils.pdf_utils import extract_text_from_filelike
from utils.pinecone_client import init_pinecone, upsert_documents, query_index
from utils.embed_cache import load_cache, save_cache
from dotenv import load_dotenv
import re

load_dotenv()

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "resume-ranker")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CACHE_PATH = "data/embed_cache.pkl"

@st.cache_resource
def get_model():
    return SentenceTransformer(MODEL_NAME)


def extract_skills(text, jd_text):
    """Find overlapping skills between JD and resume"""
    jd_keywords = re.findall(r"\b[A-Za-z]+\b", jd_text.lower())
    jd_keywords = list(set([w for w in jd_keywords if len(w) > 2]))  # basic filter
    resume_words = set(re.findall(r"\b[A-Za-z]+\b", text.lower()))
    matched = [kw for kw in jd_keywords if kw in resume_words]
    return matched

def simple_summary(text, max_words=30):
    """Naive extractive summary: first N words"""
    words = text.split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")

def store_uploaded_resumes(index, files, model, cache):
    vectors = []
    texts = {}
    updated = False
    for file in files:
        name = file.name
        if name in cache:
            print(f"Skipping {name}, already in cache")
            continue
        text = extract_text_from_filelike(file)
        if not text:
            print(f"No text extracted from {name}")
            continue
        emb = model.encode(text).tolist()
        cache[name] = emb
        vectors.append((name, emb, {"filename": name, "raw_text": text}))
        texts[name] = text
        updated = True
    if vectors:
        print(f"Upserting {len(vectors)} vectors into Pinecone")
        upsert_documents(index, vectors)
    if updated:
        save_cache(CACHE_PATH, cache)
    return len(vectors), texts


def main():
    st.set_page_config(page_title="Resume Ranker", layout="wide")
    st.title("AI-Powered Resume Ranker")
    st.write("Upload resumes, store embeddings to Pinecone, and rank by semantic match to a job description.")

    if not all([PINECONE_API_KEY, INDEX_NAME]):
        st.error("Pinecone API key or index name not set. Check your .env file.")
        return

    # Initialize model & Pinecone index
    model = get_model()
    index = init_pinecone(PINECONE_API_KEY, INDEX_NAME)

    # Initialize session state to store last matches
    if "last_matches" not in st.session_state:
        st.session_state["last_matches"] = []

    tabs = st.tabs(["Upload & Store", "Rank", "Dashboard"])

    with tabs[0]:
        st.subheader("Upload & Manage Resumes")

        uploaded = st.file_uploader("Upload resumes (PDF)", type="pdf", accept_multiple_files=True)
        if st.button("Store uploaded resumes"):
            if not uploaded:
                st.warning("Upload at least one PDF.")
            else:
                cache = load_cache(CACHE_PATH)
                count, texts = store_uploaded_resumes(index, uploaded, model, cache)
                st.success(f"Stored {count} new resume embeddings in Pinecone.")

        st.markdown("---")
        st.subheader("Clear All Resumes")
        clear_confirm = st.checkbox("Yes, I really want to delete all resumes from Pinecone and local cache")
        if st.button("Clear All Resumes") and clear_confirm:
            try:
                index.delete(delete_all=True)
                if os.path.exists(CACHE_PATH):
                    os.remove(CACHE_PATH)
                st.session_state["last_matches"] = []
                st.success("All resumes removed from Pinecone and local cache cleared.")
            except Exception as e:
                st.error(f"Failed to clear resumes: {e}")


    with tabs[1]:
        job = st.text_area("Paste job description here")
        top_k = st.slider("Top K results", 1, 20, 5)
        if st.button("Rank"):
            if not job or not job.strip():
                st.warning("Enter a job description.")
            else:
                q_emb = model.encode(job).tolist()
                matches = query_index(index, q_emb, top_k=top_k)
                if not matches:
                    st.info("No resumes found in Pinecone. Upload and store some first.")
                else:
                    matches = sorted(matches, key=lambda x: x.score, reverse=True)
                    st.session_state["last_matches"] = matches  

                    rows = []
                    for i, m in enumerate(matches, start=1):
                        score_pct = round(m.score * 100, 2)
                        raw_text = m.metadata.get("raw_text", "")
                        matched_skills = extract_skills(raw_text, job) if raw_text else []
                        summary = simple_summary(raw_text) if raw_text else "N/A"

                        rows.append({
                            "Rank": i,
                            "Resume": m.metadata.get("filename", m.id),
                            "Match Score (%)": score_pct,
                            "Matched Skills": ", ".join(matched_skills[:10]),
                            "Summary": summary
                        })

                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True)
                    csv = df.to_csv(index=False)
                    st.download_button("Download Results as CSV", csv, "ranking.csv", "text/csv")


    with tabs[2]:
        st.subheader("Recruiter Dashboard")
        if st.session_state["last_matches"]:
            matches = st.session_state["last_matches"]
            avg_score = sum([m.score * 100 for m in matches]) / len(matches)
            best = matches[0]
            best_name = best.metadata.get("filename", best.id)

            st.metric("Total Resumes", len(matches))
            st.metric("Avg Match Score", f"{avg_score:.2f}%")
            st.metric("Best Candidate", f"{best_name} → {best.score*100:.2f}%")
        else:
            st.info("No rankings yet. Upload resumes and perform a ranking first.")


if __name__ == "__main__":
    main()
