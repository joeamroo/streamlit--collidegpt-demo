import streamlit as st
from haystackragtest import rag_pipeline_run, initialize_document_stores, FastembedTextEmbedder
import os

# Set API keys (consider using Streamlit secrets for production)
os.environ["QDRANT_API_KEY"] = "your_qdrant_api_key"
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

# Initialize components
document_stores = initialize_document_stores()
embedder = FastembedTextEmbedder(model="BAAI/bge-small-en-v1.5")

st.title("RAG Pipeline Demo")

query = st.text_input("Enter your query:")
if st.button("Generate"):
    answer, sources, images = rag_pipeline_run(query, document_stores, embedder)
    st.write("Answer:", answer)
    st.write("Sources:")
    for source in sources:
        st.write(f"- {source['document_type']} - {source['title']}")
    if images:
        st.write("Images:")
        for image in images:
            st.image(image)

if __name__ == "__main__":
    st.run()
