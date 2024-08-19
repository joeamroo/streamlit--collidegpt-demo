import streamlit as st
import os
from haystack import Pipeline
from haystack.utils import Secret
from haystack.dataclasses import Document
from haystack.document_stores import QdrantDocumentStore
from haystack.components.retrievers import QdrantEmbeddingRetriever
from haystack.components.generators import OpenAIGenerator
from haystack.components.embedders import SentenceTransformersTextEmbedder

# Set environment variables from Streamlit secrets
os.environ["QDRANT_API_KEY"] = st.secrets["QDRANT_API_KEY"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Configuration
QDRANT_URL = "https://326191f2-80a7-4787-9bdd-ef46a5f88987.us-east4-0.gcp.cloud.qdrant.io:6333"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

@st.cache_resource
def initialize_document_stores():
    common_params = {
        "url": QDRANT_URL,
        "api_key": Secret.from_token(os.environ["QDRANT_API_KEY"]),
        "embedding_dim": 384,
        "return_embedding": True,
    }
    
    return {
        "books": QdrantDocumentStore(index="books_haystack_improved", **common_params),
        "documents": QdrantDocumentStore(index="documents_haystack_improved", **common_params),
        "muse_videos": QdrantDocumentStore(index="muse_videos_haystack", **common_params),
        "glossaries": QdrantDocumentStore(index="glossaries_haystack", **common_params)
    }

@st.cache_resource
def initialize_embedder():
    return SentenceTransformersTextEmbedder(model_name=EMBEDDING_MODEL)

def rag_pipeline_run(query: str, document_stores, embedder):
    st.info("Generating embeddings...")
    query_embedding = embedder.embed(query)

    all_documents = []
    for collection, document_store in document_stores.items():
        retriever = QdrantEmbeddingRetriever(document_store=document_store, top_k=10)
        results = retriever.run(query_embedding=query_embedding)
        for doc in results['documents']:
            doc.meta["collection"] = collection
        all_documents.extend(results['documents'])

    st.info(f"Retrieved {len(all_documents)} documents")

    if not all_documents:
        return "No relevant information found.", [], []

    context = "\n".join([
        f"[{doc.meta.get('document_type', doc.meta.get('content_type', 'Unknown'))} - "
        f"{doc.meta.get('title', doc.meta.get('term', 'Untitled'))}] "
        f"(Chunk {doc.meta.get('chunk_index', 'N/A')}/{doc.meta.get('total_chunks', 'N/A')}) {doc.content}"
        for doc in all_documents
    ])

    prompt = f"""
    You are an expert in various industries. Answer the question based on the provided documents.
    Use information from the documents primarily. If needed, supplement with your expert knowledge, but indicate when doing so.
    Cite sources as: (Source: [document_type/content_type] - [title/term], Chunk [chunk_index]/[total_chunks]).
    Describe relevant images if present.
    Explain technical concepts for a general audience.

    Documents:
    {context}

    Question: {query}

    Expert Answer:
    """

    st.info("Generating answer...")
    generator = OpenAIGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"), model="gpt-4o-mini")
    response = generator.run(prompt=prompt)
    answer = response["replies"][0]

    sources = [
        {
            "document_type": doc.meta.get('document_type', doc.meta.get('content_type', 'Unknown')),
            "title": doc.meta.get('title', doc.meta.get('term', 'Untitled')),
            "chunk_index": doc.meta.get('chunk_index', 'N/A'),
            "total_chunks": doc.meta.get('total_chunks', 'N/A'),
            "collection": doc.meta.get('collection', 'Unknown'),
            "link": doc.meta.get('link', '#')
        }
        for doc in all_documents
        if doc.meta.get('document_type') not in ['Unknown', 'glossary_term']
    ]

    images = []
    for doc in all_documents:
        img_links = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', doc.content)
        images.extend([link for link in img_links if link.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))])

    return answer, sources, images

st.title("RAG Pipeline Demo")

document_stores = initialize_document_stores()
embedder = initialize_embedder()

query = st.text_input("Enter your query:")
if st.button("Generate"):
    with st.spinner("Processing your query..."):
        answer, sources, images = rag_pipeline_run(query, document_stores, embedder)
    
    st.write("Answer:", answer)
    
    st.subheader("Sources:")
    for source in sources:
        st.write(f"- [{source['document_type']}] {source['title']} (Chunk {source['chunk_index']}/{source['total_chunks']}) from {source['collection']} collection")
    
    if images:
        st.subheader("Relevant Images:")
        for img in images:
            st.image(img)

if __name__ == "__main__":
    st.run()
