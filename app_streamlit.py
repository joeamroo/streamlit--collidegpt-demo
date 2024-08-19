import streamlit as st
import os
import sys
from haystackragtest import rag_pipeline_run, initialize_document_stores, FastembedTextEmbedder

# Set API keys using Streamlit secrets
os.environ["QDRANT_API_KEY"] = st.secrets["QDRANT_API_KEY"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Initialize Haystack components
document_stores = initialize_document_stores()
embedder = FastembedTextEmbedder(model="BAAI/bge-small-en-v1.5")
embedder.warm_up()

st.title("RAG Pipeline Demo")

query = st.text_input("Enter your query:")

if query:
    response, sources, images = rag_pipeline_run(query, document_stores, embedder)
    
    st.write("Expert Answer:")
    st.write(response)
    
    st.write("Sources:")
    for source in sources:
        st.write(f"- [{source['document_type']}] {source['title']} (Chunk {source['chunk_index']}/{source['total_chunks']}) from {source['collection']} collection")
    
    if images:
        st.write("Relevant Images:")
        for img in images:
            st.image(img)
