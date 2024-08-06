import requests
from langchain.embeddings.base import Embeddings

class CustomEmbeddings(Embeddings):
    """Custom embeddings class using HuggingFace API for embeddings."""
    def __init__(self, api_url, headers):
        """
        Initialize with API URL and headers.

        Args:
        - api_url (str): URL for the Hugging Face API.
        - headers (dict): HTTP headers for authorization.
        """
        self.api_url = api_url
        self.headers = headers

    def embed_documents(self, texts):
        """
        Embed a list of documents.

        Args:
        - texts (list): List of texts to embed.

        Returns:
        - embeddings (dict): Embeddings for each text.
        """
        response = requests.post(self.api_url, headers=self.headers, json={"inputs": texts, "options": {"wait_for_model": True}})
        embeddings = response.json()
        return embeddings
    
    def embed_query(self, text):
        """
        Embed a single query.

        Args:
        - text (str): Query text to embed.

        Returns:
        - embedding (list): Embedding for the query.
        """
        response = requests.post(self.api_url, headers=self.headers, json={"inputs": [text], "options": {"wait_for_model": True}})
        embeddings = response.json()
        return embeddings[0]
