# AI-Powered Resume Ranker

A **Streamlit app** that ranks resumes by semantic similarity to a job description using **SentenceTransformer embeddings** and **Pinecone vector database**.  

---

## Features

- Upload PDF resumes and store embeddings in Pinecone.  
- Rank resumes by semantic match to a job description.  
- Extract skills overlapping between JD and resumes.  
- Provide quick summaries of resumes.  
- Dashboard shows top candidates, average match score, and total resumes.  
- Clear all resumes from Pinecone and local cache with a single click.  
- Local caching (`data/embed_cache.pkl`) avoids re-embedding the same files.  

---

## Setup

1. Copy `.env.example` to `.env` and fill in your Pinecone credentials:

```bash
PINECONE_API_KEY=<your_api_key>
PINECONE_INDEX_NAME=resume-ranker
```

2. Create a virtual environment and activate it:
   `python -m venv .venv`
   `.venv\Scripts\activate `    # Windows
   `source .venv/bin/activate`  # macOS / Linux


3. Install required packages:
   `pip install -r requirements.txt`


## Local Development

1. Put sample PDF resumes into data/resumes/.
2. (Optional) Preload resumes using your seed script:  `python scripts/seed_resumes.py`
3. Run the Streamlit app:  `streamlit run app.py`
4. Use the UI to upload new resumes, rank them against a job description, and view the dashboard. 




   


