from pinecone import Pinecone

def init_pinecone(api_key, index_name: str):
    pc = Pinecone(api_key=api_key)
    return pc.Index(index_name)


def upsert_documents(index, vectors):
    formatted = [
        {"id": vid, "values": vals, "metadata": meta}
        for vid, vals, meta in vectors
    ]
    index.upsert(vectors=formatted)


def query_index(index, vector, top_k=5):
    res = index.query(
        vector=[vector],          
        top_k=top_k,
        include_metadata=True
    )
    return res.matches   
