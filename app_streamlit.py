import streamlit as st
import os
import sys
import re
from PIL import Image
import requests
from io import BytesIO

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from haystackragtest import rag_pipeline_run, initialize_document_stores, FastembedTextEmbedder

# Set API keys using Streamlit secrets
os.environ["QDRANT_API_KEY"] = st.secrets["QDRANT_API_KEY"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Initialize Haystack components
document_stores = initialize_document_stores()
embedder = FastembedTextEmbedder(model="BAAI/bge-small-en-v1.5")
embedder.warm_up()

def render_latex_selectively(text):
    parts = re.split(r'(\$\$.*?\$\$|\$.*?\$)', text)
    for i, part in enumerate(parts):
        if part.startswith('$') and part.endswith('$'):
            if part.startswith('$$') and part.endswith('$$'):
                st.latex(part[2:-2])
            else:
                st.latex(part[1:-1])
        else:
            st.write(part)

st.title("RAG Pipeline Demo")

query = st.text_input("Enter your query:")

if query:
    
    response, sources, images = rag_pipeline_run(query, document_stores, embedder)    
    st.write("Expert Answer:")
    render_latex_selectively(response)
    
    # Display images
    for img_url in images:
        try:
            response = requests.get(img_url)
            img = Image.open(BytesIO(response.content))
            st.image(img, caption="Retrieved Image", use_column_width=True)
        except Exception as e:
            st.warning(f"Failed to load image from {img_url}: {str(e)}")
