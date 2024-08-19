import sys
import os
sys.path.append(os.path.expanduser('~'))
import logging
from flask import Flask, request, jsonify
from haystackragtest import rag_pipeline_run, initialize_document_stores, FastembedTextEmbedder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the home directory to the Python path
sys.path.append(os.path.expanduser('~'))

# Set API keys directly in the script
os.environ["QDRANT_API_KEY"] = "btv2xEL02mrCKHGzrxYG_AbGtcBXpNM9WM6atghoVGEUpe-4W9cy1g"
os.environ["OPENAI_API_KEY"] = "sk-proj-IG2OmdeEKKZZaRtM11jaT3BlbkFJs3koscRRi4rDpAlw6gJW"

# Instantiate the Flask application
app = Flask(__name__)

# Initialize Haystack components
document_stores = initialize_document_stores()
embedder = FastembedTextEmbedder(model="BAAI/bge-small-en-v1.5")
embedder.warm_up()

@app.route('/api/generate', methods=['POST'])
def generate():
    query = request.json.get('query')

    # Capture the raw output from rag_pipeline_run
    raw_output = rag_pipeline_run(query, document_stores, embedder)

    # Log the raw output
    logger.info(f"Raw output from rag_pipeline_run: {raw_output}")

    # Assuming the raw output is a tuple, check its length and contents
    if isinstance(raw_output, tuple):
        logger.info(f"Length of raw output: {len(raw_output)}")
        for i, item in enumerate(raw_output):
            logger.info(f"Item {i}: {item}")
    
        # If the output matches the expected three values, unpack it
        if len(raw_output) == 3:
            response, sources, images = raw_output
            return jsonify({
                "response": response,
                "sources": sources,
                "images": images
            })
        else:
            # Handle unexpected return values
            logger.error("Unexpected number of return values from rag_pipeline_run")
            return jsonify({"error": "Unexpected number of return values from rag_pipeline_run"}), 500
    else:
        # Handle the case where the output is not a tuple
        logger.error("rag_pipeline_run did not return a tuple as expected")
        return jsonify({"error": "Unexpected return type from rag_pipeline_run"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
