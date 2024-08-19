import streamlit as st
import os
import sys
from PIL import Image
import requests
from io import BytesIO
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
    st.latex(response)  # This will render LaTeX in the response
    
    # Display images
    for img_url in images:
        try:
            response = requests.get(img_url)
            img = Image.open(BytesIO(response.content))
            st.image(img, caption="Retrieved Image")
        except Exception as e:
            st.warning(f"Failed to load image from {img_url}: {str(e)}")
    
    # Display used sources as footnotes
    st.write("Sources:")
    for idx, source in enumerate(sources, start=1):
        st.write(f"[{idx}] {source['title']} (Chunk {source['chunk_index']}/{source['total_chunks']}) from {source['collection']} collection")

# You may need to add additional styling or formatting to make the footnotes look better
