import os
import logging
import re
from typing import List, Dict, Tuple
from haystack import Pipeline
from haystack.utils import Secret
from haystack.dataclasses import Document
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack.components.generators import OpenAIGenerator
from haystack_integrations.components.embedders.fastembed import FastembedTextEmbedder
from haystack import component
import tiktoken

# Disable telemetry
os.environ["HAYSTACK_TELEMETRY_ENABLED"] = "False"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Qdrant configuration
QDRANT_URL = "https://326191f2-80a7-4787-9bdd-ef46a5f88987.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
if not QDRANT_API_KEY:
    raise ValueError("QDRANT_API_KEY environment variable is not set")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

def initialize_document_stores():
    common_params = {
        "url": QDRANT_URL,
        "api_key": Secret.from_token(QDRANT_API_KEY),
        "embedding_dim": 384,
        "return_embedding": True,
        "use_sparse_embeddings": True
    }
    
    return {
        "books": QdrantDocumentStore(index="books_haystack_improved", **common_params),
        "documents": QdrantDocumentStore(index="documents_haystack_improved", **common_params),
        "muse_videos": QdrantDocumentStore(index="muse_videos_haystack", **common_params),
        "glossaries": QdrantDocumentStore(index="glossaries_haystack", **common_params)
    }

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def truncate_prompt(prompt: str, max_tokens: int = 128000) -> str:
    current_tokens = count_tokens(prompt)
    if current_tokens > max_tokens:
        excess_tokens = current_tokens - max_tokens
        truncated_prompt = prompt[:-excess_tokens]
        logger.warning(f"Prompt truncated to fit within {max_tokens} tokens.")
        return truncated_prompt
    return prompt

@component
class CustomPromptTemplate:
    @component.output_types(prompt=str)
    def run(self, documents: List[Document], query: str):
        context = "\n".join([
            f"[{doc.meta.get('document_type', doc.meta.get('content_type', 'Unknown'))} - "
            f"{doc.meta.get('title', doc.meta.get('term', 'Untitled'))}] "
            f"(Chunk {doc.meta.get('chunk_index', 'N/A')}/{doc.meta.get('total_chunks', 'N/A')}) {doc.content}"
            for doc in documents
        ])
        prompt = f"""
        You are an expert in various industries. Your task is to answer questions based on the provided documents.

        Answer the question primarily using the information from the retrieved documents. If the documents don't contain enough information to fully answer the question, you may use your expert knowledge to supplement the answer. Clearly indicate when you're using information beyond what's provided in the documents.

        For each piece of information you use from the documents, provide a citation using the following format: (Source:  - title/term) If source/title/term is unknown don't provide a citation.

        If there are any image links in the content, Show them in the answer and describe them if they are relevant to answering the question. These images may contain important diagrams, charts, or visual information related to the topic.

        When discussing technical concepts, briefly explain them in a way that would be understandable to someone with a general knowledge of the topic.

        Retrieved Documents:
        {context}

        Question: {query}

        Expert Answer:
        """
        return {"prompt": prompt}

def rag_pipeline_run(
    query: str,
    document_stores: Dict[str, QdrantDocumentStore],
    embedder: FastembedTextEmbedder,
    debug: bool = False
) -> Tuple[str, List[Dict], List[str]]:
    logger.info(f"Running RAG pipeline for query: '{query}'")
    
    try:
        # Step 1: Generate embeddings
        embedding_result = embedder.run(text=query)
        query_embedding = embedding_result["embedding"]

        all_documents = []
        for collection, document_store in document_stores.items():
            try:
                retriever = QdrantEmbeddingRetriever(
                    document_store=document_store,
                    top_k=10,
                    return_embedding=False
                )
                result = retriever.run(
                    query_embedding=query_embedding,
                    filters=None,
                    top_k=10,
                    scale_score=False
                )
                
                for doc in result["documents"]:
                    doc.meta["collection"] = collection
                all_documents.extend(result["documents"])
                
                # Debugging: Log retrieved documents
                logger.info(f"Retrieved {len(result['documents'])} documents from '{collection}' collection")
            except Exception as e:
                logger.error(f"Error retrieving documents from '{collection}': {str(e)}")

        if not all_documents:
            logger.warning("No documents were retrieved from any collection.")
            return "I'm sorry, but I couldn't find any relevant information to answer your query. Could you please rephrase or ask a different question?", [], []

        # Step 2: Generate prompt
        prompt_template = CustomPromptTemplate()
        prompt_result = prompt_template.run(documents=all_documents, query=query)
        prompt = prompt_result["prompt"]
        
        # Debugging: Log generated prompt
        logger.info(f"Generated prompt: {prompt}")

        prompt = truncate_prompt(prompt, max_tokens=128000)

        # Step 3: Call OpenAI API
        generator = OpenAIGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"), model="gpt-4o-mini")
        response = generator.run(prompt=prompt)
        
        # Debugging: Log OpenAI API response
        logger.info(f"OpenAI API response: {response}")
        
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

        # Extract image URLs
        images = extract_image_urls(all_documents)

        # Debugging: Log before returning
        logger.info(f"Generated answer: {answer}")
        logger.info(f"Sources: {sources}")
        logger.info(f"Images: {images}")

        return answer, sources, images
    except Exception as e:
        logger.error(f"Error in RAG pipeline: {str(e)}")
        return f"An error occurred while processing your query: {str(e)}", [], []


def generate_response(prompt: str) -> str:
    generator = OpenAIGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"), model="gpt-4o-mini")
    response = generator.run(prompt=prompt)
    return response["replies"][0]
        
def extract_image_urls(documents: List[Document]) -> List[str]:
    image_urls = []
    for doc in documents:
        # Check if there's an image URL in the document metadata
        if 'image_url' in doc.meta:
            image_urls.append(doc.meta['image_url'])
        
        # Also check the document content for image URLs
        content_urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', doc.content)
        image_urls.extend([url for url in content_urls if url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))])
    
    return list(set(image_urls))  # Remove duplicates


def main():
    try:
        embedder = FastembedTextEmbedder(model=EMBEDDING_MODEL)
        
        # Warm up the embedder
        logger.info("Warming up embedder...")
        embedder.warm_up()
        
        document_stores = initialize_document_stores()
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        return

    print("Welcome to the RAG Pipeline. Type 'exit' to quit.")
    while True:
        query = input("\nEnter your query: ")
        if query.lower() == 'exit':
            break
        try:
            answer, sources, images = rag_pipeline_run(query, document_stores, embedder, debug=True)
            print("\nExpert Answer:")
            print(answer)
            print("\nSources:")
            for source in sources:
                print(f"- [{source['document_type']}] {source['title']} (Chunk {source['chunk_index']}/{source['total_chunks']}) from {source['collection']} collection")
            if images:
                print("\nRelevant Images:")
                for img in images:
                    print(f"- {img}")
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()
